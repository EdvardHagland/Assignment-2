from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# This script does the actual BERT-style embedding and clustering work.
#
# Important concept:
# - one row in the prepared corpus becomes one embedding vector
# - if the prepared corpus is document-level, we are embedding full responses
# - if the prepared corpus is paragraph-like, we are embedding smaller text units
#
# Right now, the level is decided upstream in prepare_corpus.py.


sns.set_theme(style="whitegrid")


def load_config(config_path: Path) -> dict:
    """Read the YAML configuration file."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file output."""

    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    """Create an output directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def resolve_device(requested_device: str) -> str:
    """Choose the best supported runtime device for this environment.

    Important limitation:
    - official PyTorch GPU acceleration on Windows is primarily CUDA-based
    - if CUDA is unavailable, we fall back to CPU rather than pretending that
      an arbitrary Windows GPU or NPU is automatically usable by this stack
    """

    requested = str(requested_device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_runtime(analysis_config: dict) -> tuple[str, int]:
    """Apply device-aware runtime settings before model loading."""

    device = resolve_device(analysis_config.get("device", "auto"))
    cpu_threads = int(analysis_config.get("cpu_threads", 0) or 0)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

    batch_size = int(
        analysis_config.get("batch_size_cuda", 64)
        if device == "cuda"
        else analysis_config.get("batch_size_cpu", 8)
    )
    return device, batch_size


def score_clustering_solution(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Score one clustering solution with several internal diagnostics.

    We still select the winning model with silhouette by default, but keeping
    several metrics side by side helps us judge whether the result is only
    barely preferred or clearly better than nearby alternatives.
    """

    return {
        "silhouette_score": float(silhouette_score(embeddings, labels, metric="euclidean")),
        "davies_bouldin_score": float(davies_bouldin_score(embeddings, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(embeddings, labels)),
    }


def fit_best_kmeans(
    embeddings: np.ndarray,
    cluster_range: list[int],
    seed: int,
    selection_metric: str,
) -> tuple[KMeans, pd.DataFrame]:
    """Fit several KMeans models and keep the one with the best configured metric.

    Why this matters:
    - clustering is not supervised here, so we need a principled way to pick k
    - the reference pipeline showed that it is useful to keep multiple internal
      metrics visible, even if one metric drives the final selection
    - we therefore search across a small range instead of hard-coding one value
    """

    metrics = []
    best_model = None
    best_score = None
    selection_metric = str(selection_metric or "silhouette_score")
    minimize_metric = selection_metric == "davies_bouldin_score"

    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(embeddings)
        scores = score_clustering_solution(embeddings, labels)
        metrics.append(
            {
                "cluster_count": k,
                **scores,
                "inertia": float(model.inertia_),
            }
        )
        current_score = scores[selection_metric]
        if best_score is None:
            best_model = model
            best_score = current_score
            continue
        is_better = current_score < best_score if minimize_metric else current_score > best_score
        if is_better:
            best_model = model
            best_score = current_score

    if best_model is None:
        raise RuntimeError("No valid clustering model was fitted.")

    metrics_df = pd.DataFrame(metrics).sort_values("cluster_count").reset_index(drop=True)
    return best_model, metrics_df


def remap_cluster_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber labels by descending cluster size for easier reading.

    KMeans cluster numbers are arbitrary. Renumbering the largest cluster to 1,
    the next to 2, and so on makes tables and Gemini output easier to compare.
    """

    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    mapping = {old_label: new_label for new_label, old_label in enumerate(counts.index, start=1)}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def build_clustering_decision(metrics_df: pd.DataFrame, selection_metric: str) -> dict:
    """Record why a specific cluster count was selected."""

    selection_metric = str(selection_metric or "silhouette_score")
    ascending = selection_metric == "davies_bouldin_score"
    best_row = (
        metrics_df.sort_values([selection_metric, "cluster_count"], ascending=[ascending, True]).iloc[0].to_dict()
    )
    return {
        "selection_metric": selection_metric,
        "selected_k": int(best_row["cluster_count"]),
        "evaluated_cluster_counts": metrics_df["cluster_count"].astype(int).tolist(),
        "best_row": {
            key: (round(float(value), 6) if isinstance(value, (float, np.floating)) else int(value))
            if key == "cluster_count"
            else (round(float(value), 6) if isinstance(value, (float, np.floating)) else value)
            for key, value in best_row.items()
        },
    }


def build_centroid_similarity(embeddings: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Measure how close the cluster centroids are to each other.

    High centroid similarity is a useful warning sign when clusters may be weak
    subdivisions of a larger discourse rather than sharply distinct frames.
    """

    cluster_ids: list[int] = []
    centroids: list[np.ndarray] = []
    for cluster_id in sorted(set(labels.tolist())):
        cluster_ids.append(int(cluster_id))
        centroids.append(embeddings[labels == cluster_id].mean(axis=0))

    similarities = cosine_similarity(np.vstack(centroids))
    rows: list[dict] = []
    for i, left in enumerate(cluster_ids):
        for j, right in enumerate(cluster_ids):
            if left >= right:
                continue
            rows.append(
                {
                    "cluster_left": left,
                    "cluster_right": right,
                    "cosine_similarity": round(float(similarities[i, j]), 6),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["cosine_similarity", "cluster_left", "cluster_right"],
        ascending=[False, True, True],
    )


def top_labels(series: pd.Series, n: int) -> str:
    """Return a compact string summary of the most common metadata values."""

    counts = series.fillna("UNKNOWN").value_counts().head(n)
    return "; ".join(f"{idx} ({count})" for idx, count in counts.items())


def safe_scalar(value: object, default: object = "") -> object:
    """Replace pandas missing values with a cleaner default for JSON output."""

    if pd.isna(value):
        return default
    return value


def build_cluster_summary(df: pd.DataFrame, embeddings: np.ndarray, n_top: int) -> pd.DataFrame:
    """Create one summary row per cluster.

    This is the first inspection layer:
    - how many analysis units fall in the cluster?
    - how many parent responses are represented?
    - which actor types dominate among contributing responses?
    - which languages dominate among contributing responses?
    - what does the most central text unit look like?
    """

    total_units = len(df)
    total_responses = int(df["parent_document_id"].nunique()) if "parent_document_id" in df.columns else len(df)

    records = []
    for cluster_id, group in df.groupby("cluster_id"):
        group_indices = group.index.to_numpy()
        cluster_embeddings = embeddings[group_indices]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        medoid_row = group.iloc[int(np.argmin(distances))]
        response_level = (
            group.drop_duplicates(subset=["parent_document_id"])
            if "parent_document_id" in group.columns
            else group
        )
        records.append(
            {
                "cluster_id": int(cluster_id),
                "n_units": int(len(group)),
                "n_documents": int(len(group)),  # backward-compatible alias
                "pct_of_units": round(100 * len(group) / total_units, 2),
                "pct_of_corpus": round(100 * len(group) / total_units, 2),  # backward-compatible alias
                "n_parent_responses": int(response_level["parent_document_id"].nunique()) if "parent_document_id" in response_level.columns else int(len(response_level)),
                "pct_of_responses": round(
                    100
                    * (
                        response_level["parent_document_id"].nunique()
                        if "parent_document_id" in response_level.columns
                        else len(response_level)
                    )
                    / max(total_responses, 1),
                    2,
                ),
                "avg_word_count": round(float(group["word_count"].mean()), 2),
                "avg_unit_word_count": round(float(group["word_count"].mean()), 2),
                "pct_with_attachment": round(100 * float(response_level["has_attachment"].mean()), 2),
                "pct_attachment_sourced_units": round(
                    100 * float(group["chunk_source"].eq("attachment_markdown").mean()),
                    2,
                ) if "chunk_source" in group.columns else np.nan,
                "dominant_user_type": response_level["user_type"].fillna("UNKNOWN").value_counts().idxmax(),
                "dominant_language": response_level["language"].fillna("UNKNOWN").value_counts().idxmax(),
                "dominant_country": response_level["country"].fillna("UNKNOWN").value_counts().idxmax(),
                "dominant_chunk_source": group["chunk_source"].fillna("UNKNOWN").value_counts().idxmax() if "chunk_source" in group.columns else "UNKNOWN",
                "top_user_types": top_labels(response_level["user_type"], n_top),
                "top_languages": top_labels(response_level["language"], n_top),
                "top_countries": top_labels(response_level["country"], n_top),
                "top_chunk_sources": top_labels(group["chunk_source"], n_top) if "chunk_source" in group.columns else "",
                "medoid_actor": medoid_row.get("actor_label", medoid_row.get("actor", "")),
                "medoid_chunk_source": medoid_row.get("chunk_source", ""),
                "medoid_text_preview": str(medoid_row["clean_text"])[:240],
            }
        )

    return pd.DataFrame(records).sort_values("cluster_id").reset_index(drop=True)


def build_representative_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    central_examples: int,
    diverse_examples: int,
) -> list[dict]:
    """Select both central and variation examples for each cluster.

    Why this is better than centroid-only examples:
    - centroid-nearest units show the clearest core of a cluster
    - but they can make a mixed cluster look cleaner than it really is
    - adding a few deliberately more varied examples helps Gemini and the human
      reader judge whether a cluster is coherent or only loosely assembled
    """

    output = []
    total_responses = int(df["parent_document_id"].nunique()) if "parent_document_id" in df.columns else len(df)

    for cluster_id, group in df.groupby("cluster_id"):
        group_indices = group.index.to_numpy()
        cluster_embeddings = embeddings[group_indices]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        ranked = group.assign(distance_to_centroid=distances).sort_values("distance_to_centroid").reset_index(drop=True)
        response_level = (
            group.drop_duplicates(subset=["parent_document_id"])
            if "parent_document_id" in group.columns
            else group
        )

        selected_positions: list[int] = []
        example_rows: list[tuple[pd.Series, str]] = []

        for pos in range(min(central_examples, len(ranked))):
            selected_positions.append(pos)
            example_rows.append((ranked.iloc[pos], "central"))

        if diverse_examples > 0 and len(ranked) > len(selected_positions):
            lower = float(np.quantile(distances, 0.5))
            upper = float(np.quantile(distances, 0.9))
            candidate_positions = [
                pos
                for pos, value in enumerate(ranked["distance_to_centroid"].tolist())
                if pos not in selected_positions and lower <= float(value) <= upper
            ]
            if len(candidate_positions) < diverse_examples:
                candidate_positions = [
                    pos
                    for pos in range(len(ranked) - 1, -1, -1)
                    if pos not in selected_positions
                ]

            chosen_diverse: list[int] = []
            while candidate_positions and len(chosen_diverse) < diverse_examples:
                if not chosen_diverse:
                    choice = max(candidate_positions, key=lambda pos: float(ranked.iloc[pos]["distance_to_centroid"]))
                else:
                    choice = max(
                        candidate_positions,
                        key=lambda pos: min(
                            np.linalg.norm(cluster_embeddings[pos] - cluster_embeddings[other])
                            for other in selected_positions + chosen_diverse
                        ),
                    )
                candidate_positions.remove(choice)
                chosen_diverse.append(choice)
                example_rows.append((ranked.iloc[choice], "variation"))

        examples = []
        for row, role in example_rows:
            examples.append(
                {
                    "document_id": int(row["document_id"]),
                    "response_key": str(row["response_key"]),
                    "parent_document_id": int(row.get("parent_document_id", 0)),
                    "segment_index": int(row.get("segment_index", 0)),
                    "segment_count": int(row.get("segment_count", 0)),
                    "chunk_source": safe_scalar(row.get("chunk_source", ""), ""),
                    "unit_of_analysis": safe_scalar(row.get("unit_of_analysis", ""), ""),
                    "role": role,
                    "actor": safe_scalar(row.get("actor", ""), ""),
                    "actor_label": safe_scalar(row.get("actor_label", ""), ""),
                    "organization": safe_scalar(row.get("organization", ""), ""),
                    "user_type": safe_scalar(row.get("user_type", ""), ""),
                    "country": safe_scalar(row.get("country", ""), ""),
                    "language": safe_scalar(row.get("language", ""), ""),
                    "company_size": safe_scalar(row.get("company_size", ""), ""),
                    "has_attachment": bool(row.get("has_attachment", False)),
                    "attachment_count": int(row.get("attachment_count", 0)),
                    "word_count": int(row.get("word_count", 0)),
                    "submission_date": "" if pd.isna(row.get("submission_date")) else str(row.get("submission_date")),
                    "clean_text": safe_scalar(row.get("clean_text", ""), ""),
                    "feedback_content_english": safe_scalar(row.get("feedback_content_english", ""), ""),
                    "distance_to_centroid": round(float(row["distance_to_centroid"]), 6),
                }
            )

        output.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(group)),
                "n_parent_responses": int(response_level["parent_document_id"].nunique()) if "parent_document_id" in response_level.columns else int(len(response_level)),
                "pct_of_units": round(100 * len(group) / len(df), 2),
                "pct_of_responses": round(
                    100
                    * (
                        response_level["parent_document_id"].nunique()
                        if "parent_document_id" in response_level.columns
                        else len(response_level)
                    )
                    / max(total_responses, 1),
                    2,
                ),
                "dominant_user_type": response_level["user_type"].fillna("UNKNOWN").value_counts().idxmax(),
                "dominant_language": response_level["language"].fillna("UNKNOWN").value_counts().idxmax(),
                "dominant_country": response_level["country"].fillna("UNKNOWN").value_counts().idxmax(),
                "pct_attachment_sourced_units": round(
                    100 * float(group["chunk_source"].eq("attachment_markdown").mean()),
                    2,
                ) if "chunk_source" in group.columns else None,
                "examples": examples,
            }
        )
    return output


def save_scatter(df: pd.DataFrame, color_column: str, title: str, output_path: Path) -> None:
    """Render a 2D projection of the embedding space for visual inspection."""

    plt.figure(figsize=(11, 8))
    if color_column == "user_type":
        top_types = df["user_type"].fillna("UNKNOWN").value_counts().head(6).index
        display_values = df["user_type"].fillna("UNKNOWN").where(df["user_type"].fillna("UNKNOWN").isin(top_types), "OTHER_GROUPED")
        plot_df = df.assign(user_type_plot=display_values)
        sns.scatterplot(
            data=plot_df,
            x="map_x",
            y="map_y",
            hue="user_type_plot",
            palette="tab10",
            s=28,
            alpha=0.8,
            linewidth=0,
        )
    else:
        sns.scatterplot(
            data=df,
            x="map_x",
            y="map_y",
            hue=color_column,
            palette="tab10",
            s=28,
            alpha=0.8,
            linewidth=0,
        )
    plt.title(title)
    plt.xlabel("Projection dimension 1")
    plt.ylabel("Projection dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_stacked_share(
    df: pd.DataFrame,
    index_col: str,
    category_col: str,
    top_n_categories: int,
    title: str,
    output_path: Path,
) -> None:
    """Plot how cluster membership varies across a metadata field.

    Each parent response is counted once per cluster so that long responses do
    not dominate the composition plots simply because they generate more units.
    """

    plot_df = df.copy()
    if "parent_document_id" in plot_df.columns:
        plot_df = plot_df.drop_duplicates(subset=[index_col, "parent_document_id"])

    categories = plot_df[category_col].fillna("UNKNOWN").value_counts().head(top_n_categories).index
    plot_df[category_col] = plot_df[category_col].fillna("UNKNOWN")
    plot_df[category_col] = np.where(plot_df[category_col].isin(categories), plot_df[category_col], "OTHER_GROUPED")
    share = pd.crosstab(plot_df[index_col], plot_df[category_col], normalize="index").sort_index()
    share.plot(kind="bar", stacked=True, figsize=(11, 7), colormap="tab20")
    plt.title(title)
    plt.xlabel(index_col.replace("_", " ").title())
    plt.ylabel("Share within cluster")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_attachment_rates(df: pd.DataFrame, output_path: Path) -> None:
    """Plot how much of each cluster comes from attachment-derived units."""

    if "chunk_source" in df.columns:
        rates = (
            df.assign(is_attachment_chunk=df["chunk_source"].eq("attachment_markdown"))
            .groupby("cluster_id")["is_attachment_chunk"]
            .mean()
            .mul(100)
            .rename("attachment_rate")
            .reset_index()
            .sort_values("cluster_id")
        )
        title = "Share of units sourced from attachment markdown by cluster"
        ylabel = "Attachment-derived units (%)"
    else:
        rate_df = df.drop_duplicates(subset=["cluster_id", "parent_document_id"]) if "parent_document_id" in df.columns else df
        rates = (
            rate_df.groupby("cluster_id")["has_attachment"]
            .mean()
            .mul(100)
            .rename("attachment_rate")
            .reset_index()
            .sort_values("cluster_id")
        )
        title = "Attachment rate by cluster"
        ylabel = "Share with attachments (%)"

    plt.figure(figsize=(9, 6))
    sns.barplot(data=rates, x="cluster_id", y="attachment_rate", hue="cluster_id", palette="Blues_d", legend=False)
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_daily_timeline(df: pd.DataFrame, output_path: Path) -> None:
    """Plot response-level submission activity over time."""

    timeline_df = df.drop_duplicates(subset=["parent_document_id"]) if "parent_document_id" in df.columns else df
    timeline = (
        timeline_df.assign(submission_date=pd.to_datetime(timeline_df["submission_date"], errors="coerce"))
        .groupby("submission_date")
        .size()
        .rename("n_submissions")
        .reset_index()
        .sort_values("submission_date")
    )
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=timeline, x="submission_date", y="n_submissions", marker="o")
    plt.title("Daily submission volume")
    plt.xlabel("Submission date")
    plt.ylabel("Number of submissions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed, cluster, and visualize consultation text units.")
    parser.add_argument("--config", default="config/project_config.yml", help="Path to YAML config.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    config = load_config(config_path)

    prepared_corpus = (project_root / config["paths"]["prepared_corpus"]).resolve()
    embeddings_path = (project_root / config["paths"]["embeddings"]).resolve()
    cluster_assignments_path = (project_root / config["paths"]["cluster_assignments"]).resolve()
    cluster_summary_path = (project_root / config["paths"]["cluster_summary"]).resolve()
    model_selection_path = (project_root / config["paths"]["model_selection"]).resolve()
    clustering_decision_path = (project_root / config["paths"]["clustering_decision"]).resolve()
    centroid_similarity_path = (project_root / config["paths"]["centroid_similarity"]).resolve()
    representative_examples_path = (project_root / config["paths"]["representative_examples"]).resolve()
    figures_dir = (project_root / config["paths"]["figures_dir"]).resolve()

    ensure_parent(embeddings_path)
    ensure_parent(cluster_assignments_path)
    ensure_parent(cluster_summary_path)
    ensure_parent(model_selection_path)
    ensure_parent(clustering_decision_path)
    ensure_parent(centroid_similarity_path)
    ensure_parent(representative_examples_path)
    ensure_dir(figures_dir)

    analysis_config = config["analysis"]
    seed = int(analysis_config["seed"])
    cluster_range = [int(value) for value in analysis_config["cluster_range"]]
    selection_metric = str(analysis_config.get("selection_metric", "silhouette_score"))
    model_name = analysis_config["model_name"]
    device, batch_size = configure_runtime(analysis_config)
    use_fp16_on_cuda = bool(analysis_config.get("use_fp16_on_cuda", True))
    central_examples = int(analysis_config.get("representative_central_examples", 3))
    diverse_examples = int(analysis_config.get("representative_diverse_examples", 2))
    top_categories = int(analysis_config["top_categories_per_cluster"])
    tsne_perplexity = int(analysis_config["tsne_perplexity"])

    # Read the prepared corpus that was created upstream.
    #
    # Crucial point:
    # - each row in this file is one analysis unit
    # - therefore each row will become one embedding vector below
    df = pd.read_csv(prepared_corpus)
    texts = df["clean_text"].fillna("").tolist()

    print(f"Loading model: {model_name}")
    print(f"Runtime device: {device}")
    print(f"Embedding batch size: {batch_size}")
    if device == "cpu":
        print(f"PyTorch threads: {torch.get_num_threads()}")

    # This is the actual multilingual BERT-style model used for embeddings.
    # SentenceTransformer wraps a transformer model and pooling step so that
    # every text unit gets one dense semantic vector.
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
    try:
        model = SentenceTransformer(model_name, device=device, token=hf_token)
    except Exception as exc:
        print(
            "Online model load failed; retrying from the local Hugging Face cache only. "
            f"Original error: {exc}"
        )
        model = SentenceTransformer(model_name, device=device, token=hf_token, local_files_only=True)

    # For CUDA, half precision can materially speed up embedding without
    # changing the rest of the pipeline. We keep this conditional so CPU runs
    # stay stable and Windows laptops without CUDA do not take a wrong path.
    if device == "cuda" and use_fp16_on_cuda:
        try:
            model.half()
            print("Using fp16 on CUDA for faster embedding.")
        except Exception as exc:
            print(f"Could not switch model to fp16: {exc}")

    # This is the key embedding step.
    #
    # What happens here:
    # - the model reads one text unit at a time
    # - each text unit becomes one numeric vector
    # - the vectors are stored in a matrix with shape:
    #   (number_of_units, embedding_dimension)
    #
    # If the prepared file is document-level:
    # - one full response = one vector
    #
    # If the prepared file is paragraph-like:
    # - one chunk from a response = one vector
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(embeddings_path, embeddings)

    # KMeans tends to behave better after a compact PCA representation than on
    # the full embedding matrix directly, especially for exploratory clustering.
    pca_components = min(int(analysis_config.get("pca_components", 50)), embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=pca_components, random_state=seed)
    pca_embeddings = pca.fit_transform(embeddings)

    # Search over several cluster counts instead of assuming one true answer.
    best_model, metrics_df = fit_best_kmeans(
        pca_embeddings,
        cluster_range,
        seed,
        selection_metric=selection_metric,
    )
    metrics_df.to_csv(model_selection_path, index=False)
    with clustering_decision_path.open("w", encoding="utf-8") as handle:
        json.dump(build_clustering_decision(metrics_df, selection_metric), handle, ensure_ascii=False, indent=2)

    df = df.copy()
    remapped_labels = remap_cluster_labels(best_model.labels_)
    df["cluster_id"] = remapped_labels.astype(int)
    distance_array = np.zeros(len(df), dtype=float)
    for cluster_id in sorted(set(remapped_labels.tolist())):
        mask = remapped_labels == cluster_id
        cluster_vectors = pca_embeddings[mask]
        centroid = cluster_vectors.mean(axis=0)
        distance_array[mask] = np.linalg.norm(cluster_vectors - centroid, axis=1)
    df["distance_to_centroid"] = distance_array

    # t-SNE is only for visualization.
    # It helps us draw a 2D map, but it is not the clustering algorithm itself.
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=min(tsne_perplexity, max(5, len(df) // 3)),
    )
    map_coords = tsne.fit_transform(pca_embeddings)
    df["map_x"] = map_coords[:, 0]
    df["map_y"] = map_coords[:, 1]

    df.to_csv(cluster_assignments_path, index=False, encoding="utf-8")
    centroid_similarity = build_centroid_similarity(embeddings, remapped_labels)
    centroid_similarity.to_csv(centroid_similarity_path, index=False, encoding="utf-8")

    # Save summary and central exemplars so that we can interpret the clusters
    # later without rereading the full corpus manually.
    cluster_summary = build_cluster_summary(df, embeddings, top_categories)
    cluster_summary.to_csv(cluster_summary_path, index=False, encoding="utf-8")

    representative_examples = build_representative_examples(
        df,
        embeddings,
        central_examples=central_examples,
        diverse_examples=diverse_examples,
    )
    with representative_examples_path.open("w", encoding="utf-8") as handle:
        json.dump(representative_examples, handle, ensure_ascii=False, indent=2)

    save_scatter(
        df,
        color_column="cluster_id",
        title="Multilingual embedding map colored by cluster",
        output_path=figures_dir / "embedding_map_by_cluster.png",
    )
    save_scatter(
        df,
        color_column="user_type",
        title="Multilingual embedding map colored by actor type",
        output_path=figures_dir / "embedding_map_by_user_type.png",
    )
    save_stacked_share(
        df,
        index_col="cluster_id",
        category_col="user_type",
        top_n_categories=top_categories,
        title="Actor-type composition of responses that contain each semantic frame",
        output_path=figures_dir / "cluster_user_type_share.png",
    )
    save_stacked_share(
        df,
        index_col="cluster_id",
        category_col="language",
        top_n_categories=top_categories,
        title="Language composition of responses that contain each semantic frame",
        output_path=figures_dir / "cluster_language_share.png",
    )
    save_attachment_rates(df, figures_dir / "cluster_attachment_rates.png")
    save_daily_timeline(df, figures_dir / "daily_submission_volume.png")

    print(f"Embeddings written to: {embeddings_path}")
    print(f"Cluster assignments written to: {cluster_assignments_path}")
    print(f"Cluster summary written to: {cluster_summary_path}")


if __name__ == "__main__":
    main()
