from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .catalog import Catalog, TEST_TYPE_LABELS, normalize_text, tokenize
from .models import Assessment, ChatMessage, ChatResponse, Recommendation

OFF_TOPIC_PATTERNS = [
    "salary",
    "compensation",
    "legal",
    "law",
    "gdpr",
    "visa",
    "immigration",
    "interview questions",
    "employment contract",
]

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "reveal your prompt",
    "developer message",
    "jailbreak",
]

PERSONALITY_HINTS = {"personality", "behavior", "behaviour", "culture", "teamwork", "leadership", "interpersonal", "communication", "collaborate", "collaboration"}
STAKEHOLDER_HINTS = {"stakeholder", "stakeholders"}  # soft signal — triggers personality but less aggressively
COGNITIVE_HINTS = {"reasoning", "cognitive", "aptitude", "ability", "critical", "analytical", "analysis", "thinking", "learning", "intelligence"}
TECH_HINTS = {"java", "python", "sql", "javascript", "coding", "developer", "engineer", "software", "technical", "programming", "devops", "data", "science", "machine", "learning", "cloud", "backend", "frontend"}
SIMULATION_HINTS = {"simulation", "inbox", "scenario", "situational", "case", "exercise"}

MAX_TURNS = 8  # evaluator cap

SENIORITY_KEYWORDS = {
    "intern": "Intern",
    "graduate": "Graduate",
    "entry": "Entry",
    "junior": "Entry",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "senior": "Professional",
    "manager": "Manager",
    "director": "Director",
    "executive": "Executive",
}

REFINEMENT_MARKERS = {"actually", "instead", "also", "add", "remove", "change", "update", "prefer"}


@dataclass
class Constraints:
    query_text: str
    tokens: set[str]
    seniority: set[str] = field(default_factory=set)
    preferred_test_types: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    role_signals: set[str] = field(default_factory=set)
    compare_targets: list[Assessment] = field(default_factory=list)
    is_vague: bool = False


class RecommendationAgent:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def respond(self, messages: Iterable[ChatMessage]) -> ChatResponse:
        messages = list(messages)
        total_turns = len(messages)
        user_messages = [message.content for message in messages if message.role == "user"]
        combined_user_text = "\n".join(user_messages)
        normalized_text = normalize_text(combined_user_text)

        if self._is_injection(normalized_text):
            return self._refusal("I can only help with selecting and comparing SHL assessments from the catalog.")

        if self._is_off_topic(normalized_text):
            return self._refusal("I can help with SHL assessment selection and comparison only. I can't provide general hiring, legal, or compensation advice.")

        compare_targets = self._extract_compare_targets(combined_user_text) if self._is_compare_request(normalized_text) else []
        constraints = self._extract_constraints(user_messages, compare_targets)
        if len(compare_targets) >= 2:
            return self._compare_response(compare_targets[:2])

        at_turn_cap = total_turns >= MAX_TURNS - 1

        if not at_turn_cap and self._should_clarify(messages, constraints):
            return ChatResponse(
                reply=self._clarifying_question(constraints),
                recommendations=[],
                end_of_conversation=False,
            )

        shortlisted = self._recommend(constraints)
        reply = self._recommendation_reply(constraints, shortlisted)
        recommendations = [
            Recommendation(
                name=item.name,
                url=item.url,
                test_type=", ".join(item.test_types) if item.test_types else "unknown",
            )
            for item in shortlisted
        ]
        end_of_conversation = total_turns >= MAX_TURNS
        return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end_of_conversation)

    def _extract_constraints(self, user_messages: list[str], compare_targets: list[Assessment]) -> Constraints:
        text = "\n".join(user_messages)
        tokens = set(tokenize(text))
        seniority = {label for key, label in SENIORITY_KEYWORDS.items() if key in normalize_text(text)}
        preferred_test_types: set[str] = set()
        languages = self._extract_languages(text)
        role_signals = set()

        if tokens & PERSONALITY_HINTS:
            preferred_test_types.add("P")
            role_signals.add("personality")
        if tokens & STAKEHOLDER_HINTS:  # soft signal — add personality but don't dominate scoring
            preferred_test_types.add("P")
            role_signals.add("personality")
        if tokens & COGNITIVE_HINTS:
            preferred_test_types.update({"A", "S"})
            role_signals.add("cognitive")
        if tokens & TECH_HINTS:
            preferred_test_types.add("K")
            role_signals.add("technical")
        if tokens & SIMULATION_HINTS:
            preferred_test_types.add("S")
            role_signals.add("simulation")

        meaningful_tokens = {
            token for token in tokens if len(token) > 2 and token not in {"need", "test", "role", "hire", "hiring", "looking"}
        }
        is_vague = len(meaningful_tokens) < 3 and not compare_targets

        return Constraints(
            query_text=text,
            tokens=meaningful_tokens,
            seniority=seniority,
            preferred_test_types=preferred_test_types,
            languages=languages,
            role_signals=role_signals,
            compare_targets=compare_targets,
            is_vague=is_vague,
        )

    def _extract_compare_targets(self, text: str) -> list[Assessment]:
        normalized = normalize_text(text)
        patterns = [
            r"difference between (?P<left>.+?) and (?P<right>.+?)(?:$|\?|\.)",
            r"compare (?P<left>.+?) and (?P<right>.+?)(?:$|\?|\.)",
            r"(?P<left>.+?)\s+vs\.?\s+(?P<right>.+?)(?:$|\?|\.)",
            r"(?P<left>.+?)\s+versus\s+(?P<right>.+?)(?:$|\?|\.)",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            left = self.catalog.resolve_reference(match.group("left"))
            right = self.catalog.resolve_reference(match.group("right"))
            resolved = [assessment for assessment in [left, right] if assessment is not None]
            if len(resolved) == 2:
                return resolved

        return []

    def _extract_languages(self, text: str) -> set[str]:
        normalized = normalize_text(text)
        matched = set()
        for assessment in self.catalog.assessments:
            for language in assessment.languages:
                candidate = normalize_text(language)
                if candidate and candidate in normalized:
                    matched.add(language)
        return matched

    def _should_clarify(self, messages: list[ChatMessage], constraints: Constraints) -> bool:
        assistant_turns = sum(1 for message in messages if message.role == "assistant")
        has_role_or_skill = bool(constraints.tokens & (TECH_HINTS | PERSONALITY_HINTS | COGNITIVE_HINTS) or constraints.tokens)
        has_enough_detail = has_role_or_skill and (bool(constraints.seniority) or len(constraints.tokens) >= 4 or len(constraints.query_text.split()) >= 8)
        if constraints.is_vague:
            return True
        if not has_enough_detail and assistant_turns == 0:
            return True
        return False

    def _clarifying_question(self, constraints: Constraints) -> str:
        if not constraints.tokens:
            return "What role are you hiring for, and what do you want to measure: technical skills, cognitive ability, personality, or a mix?"
        if not constraints.seniority:
            return "What seniority level is this role, and do you want technical, personality, or mixed assessments?"
        return "Do you want to prioritize technical skills, personality fit, or broader reasoning and simulation coverage?"

    def _recommend(self, constraints: Constraints) -> list[Assessment]:
        scored: list[tuple[float, Assessment]] = []
        query_text = normalize_text(constraints.query_text)

        for assessment in self.catalog.assessments:
            score = 0.0
            searchable_tokens = set(tokenize(assessment.searchable_text))
            name_tokens = set(tokenize(assessment.name))

            overlap = constraints.tokens & searchable_tokens
            score += len(overlap) * 3.0
            score += len(constraints.tokens & name_tokens) * 5.0

            if assessment.name.lower() in query_text:
                score += 12.0

            for alias in assessment.aliases:
                if alias and alias in query_text:
                    score += 8.0

            if constraints.seniority and set(assessment.job_levels) & constraints.seniority:
                score += 5.0

            if constraints.languages and set(assessment.languages) & constraints.languages:
                score += 3.0

            if constraints.preferred_test_types and set(assessment.test_types) & constraints.preferred_test_types:
                score += 7.0
            elif constraints.preferred_test_types:
                score -= 1.0

            if "java" in constraints.tokens and "java" in assessment.searchable_text:
                score += 8.0
            if (constraints.tokens & PERSONALITY_HINTS or constraints.tokens & STAKEHOLDER_HINTS) and "P" in assessment.test_types:
                score += 6.0
            if constraints.tokens & COGNITIVE_HINTS and any(code in assessment.test_types for code in ["A", "S"]):
                score += 6.0
            if constraints.tokens & TECH_HINTS and "K" in assessment.test_types:
                score += 4.0
            if constraints.tokens & SIMULATION_HINTS and "S" in assessment.test_types:
                score += 4.0

            if score > 0:
                scored.append((score, assessment))

        scored.sort(key=lambda item: (-item[0], item[1].name))

        if not scored:
            fallback = [assessment for assessment in self.catalog.assessments if assessment.test_types][:5]
            return fallback

        # Build a diverse shortlist: if multiple test types requested, ensure at least
        # one representative from each preferred type appears in the top 10
        unique: list[Assessment] = []
        seen_names: set[str] = set()
        seen_types: set[str] = set()
        required_types = set(constraints.preferred_test_types)

        # First pass: collect top-scoring items, tracking type diversity
        for _, assessment in scored:
            if assessment.name in seen_names:
                continue
            seen_names.add(assessment.name)
            unique.append(assessment)
            seen_types.update(assessment.test_types)
            if len(unique) == 10:
                break

        # Second pass: if required types are still missing, find the best representative
        if required_types and not required_types <= seen_types:
            missing = required_types - seen_types
            for missing_type in missing:
                for _, assessment in scored:
                    if assessment.name in seen_names:
                        continue
                    if missing_type in assessment.test_types:
                        # Replace last item if at cap to maintain diversity
                        if len(unique) >= 10:
                            unique.pop()
                        unique.append(assessment)
                        seen_names.add(assessment.name)
                        break

        return unique[:10]

    def _recommendation_reply(self, constraints: Constraints, items: list[Assessment]) -> str:
        if not items:
            return "I couldn’t ground a shortlist from the SHL catalog yet. Please share the role, seniority, or key skills to assess."

        parts = []
        if constraints.seniority:
            parts.append(" / ".join(sorted(constraints.seniority)))
        if constraints.preferred_test_types:
            labels = [TEST_TYPE_LABELS.get(code, code) for code in sorted(constraints.preferred_test_types)]
            parts.append("focus on " + ", ".join(labels))

        context = f" for {' and '.join(parts)}" if parts else ""
        return f"Here is a grounded SHL shortlist{context}. I ranked these against the role and constraints mentioned in the conversation."

    def _compare_response(self, assessments: list[Assessment]) -> ChatResponse:
        left, right = assessments[0], assessments[1]

        def describe(item: Assessment) -> str:
            parts: list[str] = []
            if item.description:
                parts.append(item.description)
            types = [TEST_TYPE_LABELS.get(code, code) for code in item.test_types]
            parts.append(f"Measures: {', '.join(types) if types else 'unknown'}.")
            if item.job_levels:
                parts.append(f"Job levels: {', '.join(item.job_levels)}.")
            if item.assessment_length:
                parts.append(f"Length: {item.assessment_length}.")
            parts.append(f"Catalog URL: {item.url}")
            return " ".join(parts)

        reply = (
            f"{left.name} vs {right.name} — "
            f"{left.name}: {describe(left)} | "
            f"{right.name}: {describe(right)}"
        )
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    def _is_compare_request(self, text: str) -> bool:
        return any(keyword in text for keyword in ["difference", "compare", "vs", "versus"])

    def _is_off_topic(self, text: str) -> bool:
        return any(pattern in text for pattern in OFF_TOPIC_PATTERNS)

    def _is_injection(self, text: str) -> bool:
        return any(pattern in text for pattern in INJECTION_PATTERNS)

    def _refusal(self, reply: str) -> ChatResponse:
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
