from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer


# This script turns the clustering output into something interpretable.
#
# The embedding map tells us that semantic neighborhoods exist.
# This script helps answer the harder question:
# "What is each cluster actually about?"
#
# It does that with four inspection layers:
# - top cluster terms from average TF-IDF
# - metadata summaries
# - central representative examples
# - optional Gemini labels if they already exist


# A light multilingual stopword list so the cluster-term output is more
# interpretive and less dominated by articles, auxiliaries, and generic corpus
# words like "digital" or "software".
#
# This is intentionally not exhaustive. The goal is not perfect linguistic
# normalization; it is to make the inspection tables easier to read.
INSPECTION_STOPWORDS = {
    "and", "are", "but", "can", "for", "from", "have", "into", "its", "more", "not", "our", "should", "that", "the", "their", "there", "these", "this", "use", "using", "with",
    "des", "les", "une", "est", "que", "pour", "dans", "par", "sur", "aux", "plus", "pas", "nous", "qui", "ses", "son", "elle", "ils", "leurs",
    "der", "die", "das", "und", "mit", "von", "den", "ein", "eine", "ist", "nicht", "auch", "auf", "für", "als", "sind",
    "che", "del", "della", "delle", "degli", "dei", "con", "per", "una", "sono", "anche", "non", "nel", "nella", "all", "dell",
    "los", "las", "para", "unos", "unas", "por", "como", "más", "son", "sus", "este", "esta",
    "het", "een", "van", "voor", "dat", "zijn", "niet", "ook", "met", "door", "meer", "als", "wij", "ons",
    "digital", "europe", "european", "ecosystem", "ecosystems", "eu", "open", "source", "software", "open-source", "opensource",
    "added", "value", "call", "evidence", "commission", "union", "ares", "2026",
    "strengths", "weaknesses", "barriers", "priority", "priorities", "recommended", "measures",
    "sector", "sectors", "please",
}

CATEGORICAL_SIGNAL_FIELDS = ["user_type", "language", "country", "has_attachment", "chunk_source"]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_optional_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_cluster_terms(df: pd.DataFrame, top_terms_per_cluster: int, min_df: int, max_df: float) -> pd.DataFrame:
    """Find terms that are characteristic of each cluster.

    Method:
    - fit a document-frequency vectorizer on the prepared text units
    - estimate how common each term is inside the cluster versus the corpus
    - rank terms by a contrastive document-share score

    Why this helps:
    - embeddings tell us which texts are semantically similar
    - term-share contrast helps explain that similarity in human-readable vocabulary
    - this table is a post hoc interpretability aid, not something produced by
      the embedding model itself
    """

    texts = df["clean_text"].fillna("").astype(str).tolist()
    effective_min_df = min(max(1, int(min_df)), max(1, len(texts) // 2))

    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=effective_min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[\w-]{3,}\b",
        stop_words=sorted(INSPECTION_STOPWORDS),
    )

    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    binary_matrix = (matrix > 0).astype(int)
    corpus_share = np.asarray(binary_matrix.mean(axis=0)).ravel()

    records: list[dict] = []
    for cluster_id, group in df.groupby("cluster_id"):
        cluster_binary = binary_matrix[group.index]
        cluster_doc_share = np.asarray(cluster_binary.mean(axis=0)).ravel()
        cluster_doc_counts = np.asarray(cluster_binary.sum(axis=0)).ravel()
        min_cluster_docs = max(effective_min_df, int(np.ceil(0.005 * len(group))))
        score = cluster_doc_share * np.log((cluster_doc_share + 1e-9) / (corpus_share + 1e-9))
        eligible = np.where(cluster_doc_counts >= min_cluster_docs)[0]
        if len(eligible) == 0:
            eligible = np.arange(len(feature_names))

        ranked_eligible = eligible[np.lexsort((-cluster_doc_share[eligible], -score[eligible]))]
        top_indices = ranked_eligible[:top_terms_per_cluster]

        for rank, feature_index in enumerate(top_indices, start=1):
            salience = float(score[feature_index])
            if salience <= 0:
                continue
            records.append(
                {
                    "cluster_id": int(cluster_id),
                    "term_rank": rank,
                    "term": feature_names[feature_index],
                    "cluster_doc_count": int(cluster_doc_counts[feature_index]),
                    "cluster_doc_share": round(float(cluster_doc_share[feature_index]), 6),
                    "corpus_doc_share": round(float(corpus_share[feature_index]), 6),
                    "contrast_score": round(
                        float(np.log((cluster_doc_share[feature_index] + 1e-9) / (corpus_share[feature_index] + 1e-9))),
                        6,
                    ),
                    "salience_score": round(salience, 6),
                }
            )

    return pd.DataFrame.from_records(records)


def normalize_category(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text if text else "UNKNOWN"


def build_metadata_signals(df: pd.DataFrame, fields: list[str], top_n: int) -> dict[str, dict[str, list[dict]]]:
    """Estimate which metadata values are overrepresented in each cluster.

    Dominant categories alone can be misleading. Lift-style summaries compare a
    cluster's local share to the corpus baseline, which is a cleaner way to say
    that a category is unusually common in one cluster.
    """

    output: dict[str, dict[str, list[dict]]] = {}
    for field in fields:
        if field not in df.columns:
            continue

        normalized = df[field].map(normalize_category)
        baseline = normalized.value_counts(normalize=True)
        field_payload: dict[str, list[dict]] = {}

        for cluster_id, group in df.groupby("cluster_id", sort=True):
            local = group[field].map(normalize_category)
            local_counts = local.value_counts()
            local_share = local.value_counts(normalize=True)
            rows: list[dict] = []
            for value, share in local_share.items():
                corpus_share = float(baseline.get(value, 0.0))
                rows.append(
                    {
                        "value": value,
                        "cluster_count": int(local_counts[value]),
                        "cluster_share": round(float(share), 6),
                        "corpus_share": round(corpus_share, 6),
                        "lift": round(float(share / max(corpus_share, 1e-9)), 6),
                    }
                )
            rows.sort(key=lambda item: (-item["lift"], -item["cluster_count"], item["value"]))
            field_payload[str(int(cluster_id))] = rows[:top_n]
        output[field] = field_payload

    return output


def flatten_representative_examples(payload: list[dict]) -> pd.DataFrame:
    """Convert the nested representative-examples JSON into a flat table."""

    rows: list[dict] = []
    for cluster_item in payload:
        cluster_id = int(cluster_item["cluster_id"])
        for example in cluster_item.get("examples", []):
            rows.append({"cluster_id": cluster_id, **example})
    return pd.DataFrame.from_records(rows)


def build_gemini_lookup(payload: dict | list | None) -> dict[int, dict]:
    """Create a quick lookup from cluster id to Gemini interpretation."""

    if not isinstance(payload, dict):
        return {}

    interpretations = payload.get("interpretations", [])
    lookup: dict[int, dict] = {}
    for item in interpretations:
        try:
            lookup[int(item["cluster_id"])] = item
        except Exception:
            continue
    return lookup


def build_inspection_records(
    cluster_summary: pd.DataFrame,
    cluster_terms: pd.DataFrame,
    representative_examples_flat: pd.DataFrame,
    metadata_signals: dict[str, dict[str, list[dict]]],
    gemini_lookup: dict[int, dict],
) -> list[dict]:
    """Build one human-readable inspection record per cluster."""

    records: list[dict] = []
    for _, summary_row in cluster_summary.sort_values("cluster_id").iterrows():
        cluster_id = int(summary_row["cluster_id"])

        terms = (
            cluster_terms.loc[cluster_terms["cluster_id"] == cluster_id]
            .sort_values("term_rank")["term"]
            .tolist()
        )

        examples_df = representative_examples_flat.loc[
            representative_examples_flat["cluster_id"] == cluster_id
        ].sort_values("distance_to_centroid")

        examples = []
        for _, row in examples_df.head(5).iterrows():
            examples.append(
                {
                    "document_id": int(row["document_id"]),
                    "response_key": str(row.get("response_key", "")),
                    "role": str(row.get("role", "")),
                    "segment_index": int(row.get("segment_index", 0)),
                    "segment_count": int(row.get("segment_count", 0)),
                    "chunk_source": str(row.get("chunk_source", "")),
                    "actor_label": str(row.get("actor_label", "")),
                    "user_type": str(row.get("user_type", "")),
                    "language": str(row.get("language", "")),
                    "country": str(row.get("country", "")),
                    "distance_to_centroid": float(row.get("distance_to_centroid", 0.0)),
                    "text_preview": str(row.get("clean_text", ""))[:350],
                }
            )

        gemini_item = gemini_lookup.get(cluster_id, {})

        records.append(
            {
                "cluster_id": cluster_id,
                "n_units": int(summary_row.get("n_units", summary_row["n_documents"])),
                "n_documents": int(summary_row["n_documents"]),
                "pct_of_units": float(summary_row.get("pct_of_units", summary_row["pct_of_corpus"])),
                "pct_of_corpus": float(summary_row["pct_of_corpus"]),
                "n_parent_responses": int(summary_row.get("n_parent_responses", summary_row["n_documents"])),
                "pct_of_responses": float(summary_row.get("pct_of_responses", summary_row["pct_of_corpus"])),
                "dominant_user_type": str(summary_row["dominant_user_type"]),
                "dominant_language": str(summary_row["dominant_language"]),
                "dominant_country": str(summary_row["dominant_country"]),
                "avg_word_count": float(summary_row["avg_word_count"]),
                "pct_with_attachment": float(summary_row["pct_with_attachment"]),
                "pct_attachment_sourced_units": float(summary_row.get("pct_attachment_sourced_units", 0.0)),
                "dominant_chunk_source": str(summary_row.get("dominant_chunk_source", "")),
                "top_terms": terms,
                "metadata_signals": {
                    field: metadata_signals.get(field, {}).get(str(cluster_id), [])
                    for field in CATEGORICAL_SIGNAL_FIELDS
                },
                "medoid_actor": str(summary_row["medoid_actor"]),
                "medoid_chunk_source": str(summary_row.get("medoid_chunk_source", "")),
                "medoid_text_preview": str(summary_row["medoid_text_preview"]),
                "gemini_frame_label": gemini_item.get("frame_label", ""),
                "gemini_summary": gemini_item.get("summary", ""),
                "examples": examples,
            }
        )

    return records


def render_markdown(records: list[dict]) -> str:
    """Render a readable markdown note for cluster interpretation."""

    lines: list[str] = []
    lines.append("# Cluster Inspection Notes")
    lines.append("")
    lines.append("These notes are meant to answer the question: what do the clusters seem to be about?")
    lines.append("")

    for item in records:
        lines.append(f"## Cluster {item['cluster_id']}")
        lines.append("")
        if item["gemini_frame_label"]:
            lines.append(f"Suggested label: {item['gemini_frame_label']}")
            lines.append("")
        if item["gemini_summary"]:
            lines.append(item["gemini_summary"])
            lines.append("")

        lines.append(f"- Size: {item['n_units']} text units ({item['pct_of_units']}% of all units)")
        lines.append(f"- Response coverage: {item['n_parent_responses']} parent responses ({item['pct_of_responses']}% of all retained responses)")
        lines.append(f"- Dominant actor type: {item['dominant_user_type']}")
        lines.append(f"- Dominant language: {item['dominant_language']}")
        lines.append(f"- Dominant country: {item['dominant_country']}")
        lines.append(f"- Average words per unit: {item['avg_word_count']}")
        lines.append(f"- Parent-response attachment rate: {item['pct_with_attachment']}%")
        lines.append(f"- Units sourced from attachment markdown: {item['pct_attachment_sourced_units']}%")
        if item["dominant_chunk_source"]:
            lines.append(f"- Dominant chunk source: {item['dominant_chunk_source']}")
        lines.append(f"- Top characteristic terms: {', '.join(item['top_terms'])}")
        for field, rows in item["metadata_signals"].items():
            if not rows:
                continue
            preview = "; ".join(
                f"{row['value']} (lift={row['lift']:.2f}, share={row['cluster_share']:.2f})"
                for row in rows[:3]
            )
            lines.append(f"- {field.replace('_', ' ').title()} signals: {preview}")
        lines.append(f"- Medoid actor: {item['medoid_actor']}")
        if item["medoid_chunk_source"]:
            lines.append(f"- Medoid chunk source: {item['medoid_chunk_source']}")
        lines.append("")
        lines.append("Medoid preview:")
        lines.append(item["medoid_text_preview"])
        lines.append("")
        lines.append("Representative examples:")
        for example in item["examples"]:
            lines.append(
                f"- [{example['document_id']}] role={example['role']} | {example['actor_label']} | "
                f"{example['user_type']} | {example['language']} | {example['country']} | "
                f"source={example['chunk_source']} | seg={example['segment_index']}/{example['segment_count']} | "
                f"distance={example['distance_to_centroid']:.4f}"
            )
            lines.append(f"  {example['text_preview']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cluster contents and characteristic terms.")
    parser.add_argument("--config", default="config/project_config.yml", help="Path to YAML config.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    config = load_config(config_path)

    paths = config["paths"]
    analysis_cfg = config["analysis"]

    prepared_corpus = (project_root / paths["prepared_corpus"]).resolve()
    cluster_assignments_path = (project_root / paths["cluster_assignments"]).resolve()
    cluster_summary_path = (project_root / paths["cluster_summary"]).resolve()
    representative_examples_path = (project_root / paths["representative_examples"]).resolve()
    representative_examples_flat_path = (project_root / paths["representative_examples_flat"]).resolve()
    cluster_terms_path = (project_root / paths["cluster_terms"]).resolve()
    cluster_metadata_signals_path = (project_root / paths["cluster_metadata_signals"]).resolve()
    cluster_inspection_json_path = (project_root / paths["cluster_inspection_json"]).resolve()
    cluster_inspection_markdown_path = (project_root / paths["cluster_inspection_markdown"]).resolve()
    qualitative_path = (project_root / paths["qualitative_interpretation"]).resolve()

    prepared_df = pd.read_csv(prepared_corpus)
    cluster_df = pd.read_csv(cluster_assignments_path)
    cluster_summary = pd.read_csv(cluster_summary_path)

    representative_examples_payload = load_optional_json(representative_examples_path)
    if not isinstance(representative_examples_payload, list):
        raise RuntimeError("Representative examples JSON is missing or malformed.")

    representative_examples_flat = flatten_representative_examples(representative_examples_payload)

    # Use the clustered dataframe, not just the prepared dataframe, because that
    # file already contains the cluster assignments next to the cleaned text.
    top_terms_per_cluster = int(analysis_cfg.get("top_terms_per_cluster", 12))
    tfidf_min_df = int(analysis_cfg.get("tfidf_min_df", 5))
    tfidf_max_df = float(analysis_cfg.get("tfidf_max_df", 0.4))
    top_categories = int(analysis_cfg.get("top_categories_per_cluster", 5))
    cluster_terms = build_cluster_terms(
        df=cluster_df,
        top_terms_per_cluster=top_terms_per_cluster,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
    )
    metadata_signals = build_metadata_signals(
        cluster_df,
        fields=CATEGORICAL_SIGNAL_FIELDS,
        top_n=top_categories,
    )

    gemini_lookup = build_gemini_lookup(load_optional_json(qualitative_path))
    inspection_records = build_inspection_records(
        cluster_summary=cluster_summary,
        cluster_terms=cluster_terms,
        representative_examples_flat=representative_examples_flat,
        metadata_signals=metadata_signals,
        gemini_lookup=gemini_lookup,
    )

    ensure_parent(representative_examples_flat_path)
    representative_examples_flat.to_csv(representative_examples_flat_path, index=False, encoding="utf-8")

    ensure_parent(cluster_terms_path)
    cluster_terms.to_csv(cluster_terms_path, index=False, encoding="utf-8")

    ensure_parent(cluster_metadata_signals_path)
    with cluster_metadata_signals_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_signals, handle, ensure_ascii=False, indent=2)

    ensure_parent(cluster_inspection_json_path)
    with cluster_inspection_json_path.open("w", encoding="utf-8") as handle:
        json.dump(inspection_records, handle, ensure_ascii=False, indent=2)

    ensure_parent(cluster_inspection_markdown_path)
    cluster_inspection_markdown_path.write_text(render_markdown(inspection_records), encoding="utf-8")

    print(f"Prepared units available: {len(prepared_df)}")
    print(f"Cluster inspection markdown written to: {cluster_inspection_markdown_path}")
    print(f"Cluster terms written to: {cluster_terms_path}")


if __name__ == "__main__":
    main()
