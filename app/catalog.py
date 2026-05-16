from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .models import Assessment

CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog_enriched.json"
FALLBACK_CATALOG_PATH = Path(__file__).resolve().parent.parent / "shl_catalog.json"

STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "new",
    "report",
    "test",
    "assessment",
    "questionnaire",
    "interactive",
    "shl",
}

TEST_TYPE_LABELS = {
    "A": "ability",
    "B": "biodata",
    "C": "competency",
    "D": "development",
    "E": "simulation exercise",
    "K": "knowledge",
    "P": "personality",
    "S": "simulation",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def build_aliases(name: str) -> list[str]:
    aliases: set[str] = {name.lower()}
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", name) if token.lower() not in STOPWORDS]
    if tokens:
        aliases.add(" ".join(token.lower() for token in tokens))
        acronym = "".join(token[0].lower() for token in tokens if token)
        if len(acronym) >= 3:
            aliases.add(acronym)

    uppercase_chunks = re.findall(r"\b[A-Z]{2,}\d*[a-z]*\b", name)
    for chunk in uppercase_chunks:
        if len(chunk) >= 3:
            aliases.add(chunk.lower())

    if "General Ability Screen" in name:
        aliases.discard("gsa")
        aliases.update({"gas", "general ability screen", "verify general ability screen", "vgas"})
    if "Occupational Personality Questionnaire" in name:
        aliases.update({"opq", "opq32", "opq32r"})
    if name == "Global Skills Assessment":
        aliases.add("gsa")

    return sorted(alias for alias in aliases if alias)


def build_searchable_text(item: Assessment) -> str:
    parts: list[str] = [
        item.name,
        item.description,
        " ".join(item.job_levels),
        " ".join(item.languages),
        " ".join(item.test_types),
        " ".join(TEST_TYPE_LABELS.get(code, "") for code in item.test_types),
        " ".join(item.aliases),
    ]
    return normalize_text(" ".join(part for part in parts if part))


class Catalog:
    def __init__(self, assessments: Iterable[Assessment]):
        self.assessments = list(assessments)
        self.by_name = {assessment.name.lower(): assessment for assessment in self.assessments}

    @classmethod
    def load(cls) -> "Catalog":
        path = CATALOG_PATH if CATALOG_PATH.exists() else FALLBACK_CATALOG_PATH
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        assessments: list[Assessment] = []
        for raw in raw_items:
            aliases = raw.get("aliases") or build_aliases(raw["name"])
            assessment = Assessment(
                **raw,
                aliases=aliases,
                searchable_text=raw.get("searchable_text") or "",
            )
            if not assessment.searchable_text:
                assessment.searchable_text = build_searchable_text(assessment)
            assessments.append(assessment)
        return cls(assessments)

    def find_named_assessments(self, text: str) -> list[Assessment]:
        normalized = normalize_text(text)
        matches: list[Assessment] = []
        for assessment in self.assessments:
            if any(alias and alias in normalized for alias in assessment.aliases):
                matches.append(assessment)
        return matches

    def resolve_reference(self, reference: str) -> Assessment | None:
        normalized_reference = normalize_text(reference)
        if not normalized_reference:
            return None

        reference_tokens = set(tokenize(normalized_reference))
        scored: list[tuple[float, Assessment]] = []
        for assessment in self.assessments:
            score = 0.0
            normalized_name = normalize_text(assessment.name)
            alias_set = {normalize_text(alias) for alias in assessment.aliases}

            if normalized_name == normalized_reference:
                score += 100.0
            if normalized_reference in alias_set:
                score += 80.0
            if normalized_reference in normalized_name:
                score += 40.0

            overlap = reference_tokens & set(tokenize(assessment.name))
            score += len(overlap) * 8.0

            if reference_tokens and reference_tokens <= set(tokenize(assessment.name)):
                score += 20.0

            if assessment.name == "Occupational Personality Questionnaire OPQ32r" and normalized_reference in {"opq", "opq32", "opq32r"}:
                score += 50.0
            if assessment.name == "Global Skills Assessment" and normalized_reference == "gsa":
                score += 50.0

            if score > 0:
                scored.append((score, assessment))

        if not scored:
            return None

        scored.sort(key=lambda item: (-item[0], item[1].name))
        return scored[0][1]
