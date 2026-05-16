from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

from app.catalog import build_aliases, build_searchable_text
from app.models import Assessment

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "shl_catalog.json"
OUTPUT_PATH = ROOT / "catalog_enriched.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SECTION_HEADERS = {"Description", "Job levels", "Languages", "Assessment length", "Downloads"}


def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def split_csv_text(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def extract_section_map(soup: BeautifulSoup) -> dict[str, Tag]:
    section_map: dict[str, Tag] = {}
    for heading in soup.find_all(["h3", "h4"]):
        title = clean_text(heading.get_text(" ", strip=True))
        if title in SECTION_HEADERS:
            section_map[title] = heading
    return section_map


def collect_section_text(heading: Tag) -> str:
    parts: list[str] = []
    sibling = heading.find_next_sibling()
    while sibling and sibling.name not in {"h3", "h4"}:
        text = clean_text(sibling.get_text(" ", strip=True))
        if text:
            parts.append(text)
        sibling = sibling.find_next_sibling()
    return " ".join(parts)


def collect_downloads(heading: Tag) -> list[dict[str, str]]:
    downloads: list[dict[str, str]] = []
    sibling = heading.find_next_sibling()
    current_language = ""
    while sibling and sibling.name not in {"h3", "h4"}:
        if sibling.name == "p":
            text = clean_text(sibling.get_text(" ", strip=True))
            link = sibling.find("a")
            if link:
                downloads.append(
                    {
                        "label": text or clean_text(link.get_text(" ", strip=True)),
                        "url": urljoin("https://www.shl.com", link.get("href", "")),
                        "language": current_language,
                    }
                )
            elif text:
                current_language = text
        sibling = sibling.find_next_sibling()
    return downloads


def enrich_assessment(summary: dict) -> Assessment:
    response = requests.get(summary["url"], headers=HEADERS, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    sections = extract_section_map(soup)

    description = collect_section_text(sections["Description"]) if "Description" in sections else ""
    job_levels = split_csv_text(collect_section_text(sections["Job levels"])) if "Job levels" in sections else []
    languages = split_csv_text(collect_section_text(sections["Languages"])) if "Languages" in sections else []
    assessment_length = collect_section_text(sections["Assessment length"]) if "Assessment length" in sections else ""
    downloads = collect_downloads(sections["Downloads"]) if "Downloads" in sections else []

    assessment = Assessment(
        **summary,
        description=description,
        job_levels=job_levels,
        languages=languages,
        assessment_length=assessment_length,
        downloads=downloads,
        aliases=build_aliases(summary["name"]),
    )
    assessment.searchable_text = build_searchable_text(assessment)
    return assessment


def main() -> None:
    summaries = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    enriched: list[dict] = []

    for index, summary in enumerate(summaries, start=1):
        try:
            assessment = enrich_assessment(summary)
            enriched.append(assessment.model_dump())
            print(f"[{index}/{len(summaries)}] {assessment.name}")
            time.sleep(0.15)
        except Exception as exc:
            print(f"[{index}/{len(summaries)}] FAILED {summary['name']}: {exc}")
            fallback = Assessment(**summary, aliases=build_aliases(summary["name"]))
            fallback.searchable_text = build_searchable_text(fallback)
            enriched.append(fallback.model_dump())

    OUTPUT_PATH.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(enriched)} items to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
