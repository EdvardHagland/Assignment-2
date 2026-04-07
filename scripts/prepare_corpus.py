from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd
import yaml


# This script prepares the text units that will later be embedded.
#
# The most important choice here is the unit of analysis.
#
# We now support three modes:
# - `document`: one response row becomes one embedding vector
# - `pseudo_paragraph`: sentence-window chunks from flattened response text
# - `attachment_aware_paragraph`: use real markdown paragraph breaks from
#   `attachments_markdown` when available, and a sentence-window fallback
#   for the flattened response body when attachments are absent
#
# The third mode is the preferred setup for this consultation, because many of
# the most substantive submissions were uploaded as markdown-converted
# attachments that still preserve headings and paragraph separators.


def load_config(config_path: Path) -> dict:
    """Read the project configuration file."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an output file if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def clean_markdown_text(text: str) -> str:
    """Remove the noisiest markdown/HTML artifacts while keeping the wording."""

    text = html.unescape(text or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[*_#>`~\-]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    """Create sentence-like units for pseudo-paragraph segmentation.

    The source field in this corpus does not preserve real paragraph breaks.
    So if we want a smaller unit than a full response, the most honest option
    is to build sentence windows rather than pretending that true paragraphs
    still exist in the compiled text.
    """

    normalized = clean_markdown_text(text)
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def split_markdown_blocks(text: str) -> list[str]:
    """Split markdown text on blank lines.

    This is the attachment-aware paragraph splitter. Unlike the flattened main
    response field, attachment markdown still preserves blank lines, headings,
    and list structure in many cases.
    """

    if not text or not str(text).strip():
        return []
    return [block.strip() for block in re.split(r"\n\s*\n+", str(text)) if block.strip()]


def build_pseudo_paragraphs(text: str, target_words: int, min_words: int) -> list[str]:
    """Group neighboring sentences into paragraph-like windows.

    Why do this:
    - document-level embeddings give us one vector per whole response
    - smaller windows can surface finer argumentative themes
    - this is useful when a single response mixes several points

    Why not call them real paragraphs:
    - the compiled corpus flattened line breaks
    - so these are reconstructed chunks, not author-preserved paragraphs
    """

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)

        # If adding the next sentence would overshoot the target and we already
        # have enough text, we close the current chunk and start a new one.
        if current_sentences and current_words >= min_words and current_words + sentence_words > target_words:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = []
            current_words = 0

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    # Merge a very short last chunk into the previous chunk so we do not create
    # tiny fragments that are too weak for semantic embedding.
    if len(chunks) >= 2 and word_count(chunks[-1]) < min_words:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return [chunk for chunk in chunks if word_count(chunk) >= min_words]


def is_heading_or_metadata_block(raw_block: str, cleaned_block: str) -> bool:
    """Identify short structural blocks that should be merged with a paragraph.

    Examples:
    - markdown headings such as `## Introduction`
    - filename/title blocks from converted attachments
    - short reference lines like `Ref. Ares(...)`
    """

    raw = (raw_block or "").strip()
    cleaned = (cleaned_block or "").strip()
    lowered = cleaned.lower()

    if not raw or not cleaned:
        return True

    if re.match(r"^\s{0,3}#{1,6}\s", raw):
        return True

    if lowered.startswith("ref.") or lowered.startswith("written by") or lowered.startswith("submission to:"):
        return True

    if re.search(r"\.(pdf|docx?|odt|txt)\b", lowered):
        return True

    return False


def build_attachment_markdown_chunks(text: str, target_words: int, min_words: int) -> list[str]:
    """Create paragraph-like chunks from attachment markdown.

    Strategy:
    - split on blank lines to recover markdown paragraph blocks
    - merge short heading/metadata blocks into the following paragraph
    - keep substantive blocks as their own units
    - split exceptionally long blocks into sentence windows

    This preserves real paragraph structure where the data still gives it to us.
    """

    blocks = split_markdown_blocks(text)
    if not blocks:
        return []

    chunks: list[str] = []
    pending_context: list[str] = []

    for raw_block in blocks:
        cleaned_block = clean_markdown_text(raw_block)
        if not cleaned_block:
            continue

        if is_heading_or_metadata_block(raw_block, cleaned_block) or word_count(cleaned_block) < min_words:
            pending_context.append(cleaned_block)
            continue

        chunk_text = " ".join(pending_context + [cleaned_block]).strip() if pending_context else cleaned_block
        pending_context = []

        # Very long paragraphs are still too broad for one embedding vector, so
        # we split only the outliers while keeping ordinary markdown paragraphs
        # intact as the default unit.
        if word_count(chunk_text) > target_words * 1.75:
            split_chunks = build_pseudo_paragraphs(chunk_text, target_words=target_words, min_words=min_words)
            if split_chunks:
                chunks.extend(split_chunks)
                continue

        chunks.append(chunk_text)

    # If the attachment ends with dangling headings or very short context lines,
    # fold them into the previous chunk when possible.
    if pending_context:
        trailing_context = " ".join(pending_context).strip()
        if chunks:
            candidate = f"{chunks[-1]} {trailing_context}".strip()
            if word_count(candidate) <= target_words * 2.2:
                chunks[-1] = candidate
        elif word_count(trailing_context) >= min_words:
            chunks.append(trailing_context)

    return [chunk for chunk in chunks if word_count(chunk) >= min_words]


def build_response_fallback_chunks(
    text: str,
    target_words: int,
    min_words: int,
    fallback_min_words: int,
) -> list[str]:
    """Chunk the flattened response body when no attachment markdown exists.

    We initially hoped non-attachment responses would be short enough to treat
    as one paragraph, but in this corpus many still run several hundred words.
    So we keep short bodies intact and only split the longer flattened ones.

    Important safeguard:
    - `min_words` is the preferred lower bound for paragraph-like chunks
    - `fallback_min_words` is the smaller response-level threshold used to keep
      short but still meaningful submissions in the analysis
    """

    cleaned = clean_markdown_text(text)
    if not cleaned:
        return []

    cleaned_word_count = word_count(cleaned)
    if cleaned_word_count < fallback_min_words:
        return []

    if cleaned_word_count <= target_words * 1.25 or cleaned_word_count < min_words:
        return [cleaned]

    chunks = build_pseudo_paragraphs(cleaned, target_words=target_words, min_words=min_words)
    if chunks:
        return chunks

    # If sentence-window chunking fails to produce paragraph-sized units, keep
    # the full response instead of losing the submission entirely.
    return [cleaned]


def build_attachment_aware_chunks(
    response_text: str,
    attachments_markdown: str,
    target_words: int,
    min_words: int,
    fallback_min_words: int,
) -> list[tuple[str, str]]:
    """Return `(chunk_source, chunk_text)` pairs for the preferred hybrid mode.

    Priority:
    1. Use attachment markdown when it exists, because it preserves paragraph
       separators and richer structure.
    2. Fall back to the flattened response body otherwise.
    """

    attachment_chunks = build_attachment_markdown_chunks(
        attachments_markdown,
        target_words=target_words,
        min_words=min_words,
    )
    if attachment_chunks:
        return [("attachment_markdown", chunk) for chunk in attachment_chunks]

    response_chunks = build_response_fallback_chunks(
        response_text,
        target_words=target_words,
        min_words=min_words,
        fallback_min_words=fallback_min_words,
    )
    return [("response_markdown", chunk) for chunk in response_chunks]


def word_count(text: str) -> int:
    """Count whitespace-separated tokens for lightweight filtering."""

    return len(re.findall(r"\S+", text or ""))


def nonempty(value: object) -> str:
    """Convert nullable values to a clean string."""

    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def summarize_metadata(prepared_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict:
    """Build a compact summary for the report and quick QA.

    Important distinction:
    - response-level metadata should be counted on the original response rows
    - unit-level metadata should be counted on the prepared chunk table
    """

    top_n = 10
    response_word_counts = raw_df["source_text"].map(lambda text: word_count(clean_markdown_text(text)))
    units_per_response = prepared_df.groupby("parent_document_id").size()

    avg_unit_word_count = round(float(prepared_df["word_count"].mean()), 2)
    median_unit_word_count = float(prepared_df["word_count"].median())

    return {
        "n_units": int(len(prepared_df)),
        "n_documents": int(len(prepared_df)),  # backward-compatible alias for older report code
        "n_responses": int(len(raw_df)),
        "n_retained_responses": int(prepared_df["parent_document_id"].nunique()),
        "avg_unit_word_count": avg_unit_word_count,
        "median_unit_word_count": median_unit_word_count,
        "avg_word_count": avg_unit_word_count,  # backward-compatible alias
        "median_word_count": median_unit_word_count,  # backward-compatible alias
        "avg_response_word_count": round(float(response_word_counts.mean()), 2),
        "median_response_word_count": float(response_word_counts.median()),
        "attachment_rate": round(float(raw_df["has_attachment"].mean()), 4),
        "avg_units_per_response": round(float(units_per_response.mean()), 2),
        "median_units_per_response": float(units_per_response.median()),
        "chunk_source_counts": prepared_df["chunk_source"].value_counts().to_dict() if "chunk_source" in prepared_df.columns else {},
        "top_user_types": raw_df["user_type"].fillna("UNKNOWN").value_counts().head(top_n).to_dict(),
        "top_languages": raw_df["language"].fillna("UNKNOWN").value_counts().head(top_n).to_dict(),
        "top_countries": raw_df["country"].fillna("UNKNOWN").value_counts().head(top_n).to_dict(),
        "unit_of_analysis": str(prepared_df["unit_of_analysis"].iloc[0]) if not prepared_df.empty else "",
    }


def expand_to_analysis_units(df: pd.DataFrame, unit_of_analysis: str, min_words: int, target_words: int, paragraph_min_words: int) -> pd.DataFrame:
    """Transform the compiled response-level corpus into analysis units.

    `document`:
    - one response becomes one embedding

    `pseudo_paragraph`:
    - one response may become several sentence-window chunks
    - this is the closest thing we can do to paragraph-level analysis with the
      current flattened text source

    `attachment_aware_paragraph`:
    - use real markdown paragraph breaks from attachments when available
    - use sentence-window fallback chunks for the flattened response body
    """

    records: list[dict] = []
    parent_document_id = 0

    for _, row in df.iterrows():
        parent_document_id += 1
        raw_text = row["source_text"]

        if unit_of_analysis == "document":
            clean_text = clean_markdown_text(raw_text)
            if word_count(clean_text) < min_words:
                continue

            records.append(
                {
                    **row.to_dict(),
                    "parent_document_id": parent_document_id,
                    "segment_index": 1,
                    "segment_count": 1,
                    "unit_of_analysis": "document",
                    "chunk_source": "response_markdown",
                    "clean_text": clean_text,
                    "word_count": word_count(clean_text),
                }
            )
            continue

        if unit_of_analysis == "attachment_aware_paragraph":
            chunk_pairs = build_attachment_aware_chunks(
                response_text=raw_text,
                attachments_markdown=row.get("attachments_markdown", ""),
                target_words=target_words,
                min_words=paragraph_min_words,
                fallback_min_words=min_words,
            )

            for segment_index, (chunk_source, chunk_text) in enumerate(chunk_pairs, start=1):
                records.append(
                    {
                        **row.to_dict(),
                        "parent_document_id": parent_document_id,
                        "segment_index": segment_index,
                        "segment_count": len(chunk_pairs),
                        "unit_of_analysis": "attachment_aware_paragraph",
                        "chunk_source": chunk_source,
                        "clean_text": chunk_text,
                        "word_count": word_count(chunk_text),
                    }
                )
            continue

        if unit_of_analysis != "pseudo_paragraph":
            raise ValueError(f"Unsupported unit_of_analysis: {unit_of_analysis}")

        chunks = build_pseudo_paragraphs(
            raw_text,
            target_words=target_words,
            min_words=paragraph_min_words,
        )

        for segment_index, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    **row.to_dict(),
                    "parent_document_id": parent_document_id,
                    "segment_index": segment_index,
                    "segment_count": len(chunks),
                    "unit_of_analysis": "pseudo_paragraph",
                    "chunk_source": "response_markdown",
                    "clean_text": chunk,
                    "word_count": word_count(chunk),
                }
            )

    prepared = pd.DataFrame.from_records(records)
    if prepared.empty:
        raise RuntimeError("Preparation produced zero analysis units. Check filtering settings.")

    prepared["document_id"] = range(1, len(prepared) + 1)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the compiled consultation corpus.")
    parser.add_argument("--config", default="config/project_config.yml", help="Path to YAML config.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    config = load_config(config_path)

    raw_corpus = (project_root / config["paths"]["raw_corpus"]).resolve()
    prepared_corpus = (project_root / config["paths"]["prepared_corpus"]).resolve()
    metadata_summary_path = (project_root / config["paths"]["metadata_summary"]).resolve()

    analysis_cfg = config["analysis"]
    min_words = int(analysis_cfg["min_words"])
    text_column = analysis_cfg["text_column"]
    unit_of_analysis = str(analysis_cfg.get("unit_of_analysis", "document"))
    paragraph_target_words = int(analysis_cfg.get("paragraph_target_words", 120))
    paragraph_min_words = int(analysis_cfg.get("paragraph_min_words", 40))

    # Read the compiled corpus produced by Assignment 1.
    df = pd.read_csv(raw_corpus)

    required_columns = {
        "response_key",
        "actor",
        "user_type",
        "country",
        "language",
        "attachment_count",
        "date_feedback",
        text_column,
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep the raw response row, but add a few cleaned helper fields that are
    # useful downstream for both metadata summaries and cluster inspection.
    df = df.copy()
    df["source_text"] = df[text_column].fillna("").astype(str)
    if "attachments_markdown" in df.columns:
        df["attachments_markdown"] = df["attachments_markdown"].fillna("").astype(str)
    else:
        df["attachments_markdown"] = ""
    df["has_attachment"] = pd.to_numeric(df["attachment_count"], errors="coerce").fillna(0).astype(int) > 0
    df["organization_clean"] = df["organization"].map(nonempty) if "organization" in df.columns else ""
    df["actor_label"] = df["organization_clean"]
    df.loc[df["actor_label"] == "", "actor_label"] = df["actor"].map(nonempty)
    df["submission_date"] = pd.to_datetime(df["date_feedback"], format="%Y/%m/%d %H:%M:%S", errors="coerce").dt.date

    before = len(df)

    # This is the most important conceptual step:
    # we choose the unit that will become one embedding vector.
    prepared = expand_to_analysis_units(
        df=df,
        unit_of_analysis=unit_of_analysis,
        min_words=min_words,
        target_words=paragraph_target_words,
        paragraph_min_words=paragraph_min_words,
    )

    keep_columns = [
        "document_id",
        "parent_document_id",
        "segment_index",
        "segment_count",
        "unit_of_analysis",
        "chunk_source",
        "response_key",
        "actor",
        "actor_label",
        "organization",
        "user_type",
        "country",
        "language",
        "company_size",
        "date_feedback",
        "submission_date",
        "attachment_count",
        "has_attachment",
        "word_count",
        "clean_text",
        "feedback_content_english",
    ]
    keep_columns = [column for column in keep_columns if column in prepared.columns]
    prepared = prepared[keep_columns]

    ensure_parent(prepared_corpus)
    prepared.to_csv(prepared_corpus, index=False, encoding="utf-8")

    metadata_summary = summarize_metadata(prepared, df)
    metadata_summary["n_raw_documents"] = int(before)
    metadata_summary["n_filtered_out"] = int(before - prepared["parent_document_id"].nunique())
    metadata_summary["min_words_threshold"] = min_words
    metadata_summary["paragraph_target_words"] = paragraph_target_words
    metadata_summary["paragraph_min_words"] = paragraph_min_words

    ensure_parent(metadata_summary_path)
    with metadata_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_summary, handle, ensure_ascii=False, indent=2)

    print(f"Prepared corpus written to: {prepared_corpus}")
    print(f"Raw responses available: {before}")
    print(f"Analysis units retained: {len(prepared)}")
    print(f"Unit of analysis: {unit_of_analysis}")


if __name__ == "__main__":
    main()
