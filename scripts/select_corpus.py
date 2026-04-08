from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPILED_DIR = PROJECT_ROOT / "data" / "compiled"
CATALOG_PATH = COMPILED_DIR / "corpus_catalog.csv"
ACTIVE_PATH = COMPILED_DIR / "consultation_compiled.csv"


def load_catalog() -> list[dict[str, str]]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_record(records: list[dict[str, str]], query: str) -> dict[str, str]:
    normalized = query.strip().lower()
    for record in records:
        filename = record.get("filename", "")
        candidates = {
            str(record.get("corpus_id", "")).strip().lower(),
            filename.lower(),
            Path(filename).stem.lower(),
        }
        if normalized in candidates:
            return record
    available = ", ".join(record.get("corpus_id", "?") for record in records) or "none"
    raise SystemExit(f"Unknown corpus '{query}'. Available corpus ids: {available}")


def print_catalog(records: list[dict[str, str]]) -> None:
    if not records:
        print("No bundled corpora found.")
        return
    print("Bundled corpora:")
    for record in records:
        corpus_id = record.get("corpus_id", "?")
        title = record.get("title", "").strip()
        topics = record.get("topics", "").strip()
        total_feedback_count = record.get("total_feedback_count", "").strip()
        filename = record.get("filename", "").strip()
        print(
            f"- {corpus_id}: {title} "
            f"[{topics}] "
            f"responses={total_feedback_count} "
            f"file={filename}"
        )


def copy_active_corpus(record: dict[str, str]) -> None:
    filename = record.get("filename", "").strip()
    if not filename:
        raise SystemExit("Catalog entry is missing filename.")
    source_path = COMPILED_DIR / filename
    if not source_path.exists():
        raise SystemExit(f"Bundled corpus file not found: {source_path}")
    shutil.copyfile(source_path, ACTIVE_PATH)
    print(
        f"Selected corpus {record.get('corpus_id', '?')}: "
        f"{record.get('title', '').strip()}"
    )
    print(f"Active file updated: {ACTIVE_PATH.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List bundled compiled corpora or select one as the active consultation corpus."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the bundled compiled corpora and exit.",
    )
    parser.add_argument(
        "--corpus",
        help="Corpus id or filename stem to activate, for example 16213 or 14031_compiled.",
    )
    args = parser.parse_args()

    records = load_catalog()

    if args.list or not args.corpus:
        print_catalog(records)
        if not args.corpus:
            return 0

    record = resolve_record(records, args.corpus)
    copy_active_corpus(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
