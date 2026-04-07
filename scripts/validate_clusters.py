from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


# This script is the "are these clusters real?" layer.
#
# It does not prove that clustering is objectively true in some absolute sense.
# That would be too strong for exploratory text mining.
#
# What it can do, however, is answer three much better questions:
# 1. Does the embedding space show more cluster tendency than random data would?
# 2. Does the chosen clustering solution beat a sensible null baseline?
# 3. Is the solution stable when we perturb the data or switch algorithms?
#
# Those are the questions we actually want for a careful assignment report.


sns.set_theme(style="whitegrid")


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_config(config_path: Path) -> dict:
    """Read the YAML configuration file."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file output if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a project-relative path from the YAML configuration."""

    return (project_root / relative_path).resolve()


def build_pca_representation(embeddings: np.ndarray, seed: int) -> np.ndarray:
    """Recreate the compact PCA space used for clustering.

    Why validate in PCA space instead of the raw vectors?
    - the actual clustering pipeline already clusters in PCA space
    - we therefore validate the same representation that generated the labels
    - this keeps the validation aligned with the modeling choice
    """

    n_components = min(50, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(embeddings)


def compute_hopkins_statistic(data: np.ndarray, sample_size: int, seed: int) -> dict:
    """Estimate whether the space has cluster tendency before clustering.

    Hopkins is useful here because it asks:
    "Do the data points look more clumped than a random point cloud?"

    Interpretation:
    - around 0.5: data look roughly random in space
    - much above 0.5: data show clustering tendency
    - much below 0.5: data look unusually regular or over-dispersed

    We compute this on the first few PCA dimensions only. That is deliberate:
    in very high dimensions, nearest-neighbor distances become noisy and much
    harder to interpret.
    """

    rng = np.random.default_rng(seed)
    n_rows = data.shape[0]
    n_dims = min(10, data.shape[1])
    reduced = data[:, :n_dims]
    sample_size = int(max(20, min(sample_size, n_rows - 1)))

    mins = reduced.min(axis=0)
    maxs = reduced.max(axis=0)

    # Real points sampled from the actual corpus.
    sampled_idx = rng.choice(n_rows, size=sample_size, replace=False)
    real_points = reduced[sampled_idx]

    # Synthetic points sampled uniformly from the same PCA bounding box.
    synthetic_points = rng.uniform(low=mins, high=maxs, size=(sample_size, n_dims))

    # For real observations we need the nearest neighbor other than themselves.
    nn_real = NearestNeighbors(n_neighbors=2).fit(reduced)
    real_distances = nn_real.kneighbors(real_points, n_neighbors=2)[0][:, 1]

    # For synthetic points we just need the nearest real observation.
    nn_synth = NearestNeighbors(n_neighbors=1).fit(reduced)
    synthetic_distances = nn_synth.kneighbors(synthetic_points, n_neighbors=1)[0][:, 0]

    hopkins = float(synthetic_distances.sum() / (synthetic_distances.sum() + real_distances.sum()))

    return {
        "statistic": hopkins,
        "sample_size": sample_size,
        "dimensions_used": n_dims,
    }


def column_shuffle_null(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Destroy multivariate structure while preserving each dimension's marginal spread.

    This is our null model for "semantic structure by chance":
    - every PCA dimension keeps its observed values
    - but the columns are independently shuffled, so cross-dimensional structure
      and semantic neighborhoods are broken apart
    """

    shuffled = np.empty_like(data)
    for column_idx in range(data.shape[1]):
        shuffled[:, column_idx] = rng.permutation(data[:, column_idx])
    return shuffled


def compute_best_silhouette(data: np.ndarray, cluster_range: list[int], seed: int, n_init: int = 20) -> tuple[int, float]:
    """Find the best silhouette score across the configured k range."""

    best_k = None
    best_score = float("-inf")

    for cluster_count in cluster_range:
        model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=n_init)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels, metric="euclidean")
        if score > best_score:
            best_score = score
            best_k = cluster_count

    if best_k is None:
        raise RuntimeError("Could not compute a null silhouette score.")

    return int(best_k), float(best_score)


def compute_null_distribution(
    data: np.ndarray,
    cluster_range: list[int],
    observed_best_silhouette: float,
    observed_k: int,
    seed: int,
    n_draws: int,
) -> tuple[pd.DataFrame, dict]:
    """Build a permutation null distribution for the best silhouette score.

    This gives us the closest thing to a p-value that makes sense here.

    The question is not:
    "Is cluster 1 different from cluster 2 by t-test?"

    The better question is:
    "If the corpus had no genuine multivariate structure, how often would we
    still see a best silhouette score this large after searching over k?"
    """

    rng = np.random.default_rng(seed)
    records = []
    log(f"Permutation null: starting {n_draws} draws across k={cluster_range}")
    report_every = max(1, n_draws // 10)

    for draw_idx in range(1, n_draws + 1):
        shuffled = column_shuffle_null(data, rng)
        best_k, best_score = compute_best_silhouette(
            shuffled,
            cluster_range=cluster_range,
            seed=seed + draw_idx,
            n_init=10,
        )
        records.append(
            {
                "draw": draw_idx,
                "best_cluster_count": best_k,
                "best_silhouette_score": best_score,
                "beats_observed": bool(best_score >= observed_best_silhouette),
                "observed_cluster_count": observed_k,
            }
        )
        if draw_idx == 1 or draw_idx == n_draws or draw_idx % report_every == 0:
            log(
                f"Permutation null: completed draw {draw_idx}/{n_draws} "
                f"(best_k={best_k}, best_silhouette={best_score:.4f})"
            )

    null_df = pd.DataFrame(records)
    p_value = float((1 + null_df["beats_observed"].sum()) / (1 + len(null_df)))

    summary = {
        "observed_best_silhouette": float(observed_best_silhouette),
        "observed_cluster_count": int(observed_k),
        "null_draws": int(len(null_df)),
        "null_mean_best_silhouette": float(null_df["best_silhouette_score"].mean()),
        "null_std_best_silhouette": float(null_df["best_silhouette_score"].std(ddof=1)),
        "null_max_best_silhouette": float(null_df["best_silhouette_score"].max()),
        "permutation_p_value": p_value,
    }
    return null_df, summary


def compute_bootstrap_stability(
    data: np.ndarray,
    reference_labels: np.ndarray,
    cluster_count: int,
    seed: int,
    n_draws: int,
    sample_fraction: float,
) -> pd.DataFrame:
    """Estimate how stable the solution is under repeated subsampling.

    For each draw:
    - sample a large subset of documents
    - recluster only that subset
    - compare the new labels to the original labels on the same documents

    Adjusted Rand Index (ARI) is useful because:
    - it is invariant to label permutations
    - 0 means agreement no better than chance
    - 1 means identical partitioning
    """

    rng = np.random.default_rng(seed)
    n_rows = data.shape[0]
    subset_size = int(max(cluster_count * 25, round(sample_fraction * n_rows)))
    subset_size = min(subset_size, n_rows)

    records = []
    log(f"Bootstrap stability: starting {n_draws} draws with subset size {subset_size}")
    report_every = max(1, n_draws // 10)
    for draw_idx in range(1, n_draws + 1):
        subset_idx = np.sort(rng.choice(n_rows, size=subset_size, replace=False))
        subset_data = data[subset_idx]
        subset_reference = reference_labels[subset_idx]

        model = KMeans(n_clusters=cluster_count, random_state=seed + draw_idx, n_init=20)
        subset_labels = model.fit_predict(subset_data)

        records.append(
            {
                "draw": draw_idx,
                "sample_size": subset_size,
                "adjusted_rand_index": float(adjusted_rand_score(subset_reference, subset_labels)),
                "silhouette_score": float(silhouette_score(subset_data, subset_labels, metric="euclidean")),
            }
        )
        if draw_idx == 1 or draw_idx == n_draws or draw_idx % report_every == 0:
            log(
                f"Bootstrap stability: completed draw {draw_idx}/{n_draws} "
                f"(ARI={records[-1]['adjusted_rand_index']:.4f})"
            )

    return pd.DataFrame(records)


def compute_algorithm_agreement(data: np.ndarray, reference_labels: np.ndarray, cluster_count: int, seed: int) -> dict:
    """Check whether different clustering families recover similar partitions.

    This does not prove a single "true" clustering exists.
    It simply asks whether the discovered structure is robust across reasonable
    alternatives instead of being a quirk of one specific algorithm.
    """

    log("Algorithm agreement: comparing KMeans with Agglomerative and Gaussian Mixture")
    agg = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
    agg_labels = agg.fit_predict(data)

    gmm = GaussianMixture(n_components=cluster_count, random_state=seed, n_init=5)
    gmm_labels = gmm.fit_predict(data)

    result = {
        "kmeans_vs_agglomerative_ari": float(adjusted_rand_score(reference_labels, agg_labels)),
        "kmeans_vs_gmm_ari": float(adjusted_rand_score(reference_labels, gmm_labels)),
    }
    log(
        "Algorithm agreement: "
        f"KMeans vs Agglomerative ARI={result['kmeans_vs_agglomerative_ari']:.4f}, "
        f"KMeans vs GMM ARI={result['kmeans_vs_gmm_ari']:.4f}"
    )
    return result


def group_sparse_categories(series: pd.Series, top_n: int) -> pd.Series:
    """Keep the largest categories and collapse the rest into OTHER_GROUPED."""

    cleaned = series.fillna("UNKNOWN").astype(str)
    top_levels = cleaned.value_counts().head(top_n).index
    grouped = cleaned.where(cleaned.isin(top_levels), "OTHER_GROUPED")
    return grouped


def compute_metadata_association_tests(df: pd.DataFrame, top_n: int) -> list[dict]:
    """Run chi-square association tests between clusters and key metadata fields.

    These tests do not validate clustering by themselves.
    What they tell us is whether the semantic partition is meaningfully related
    to observable corpus structure such as actor type or language.
    """

    fields = ["user_type", "language", "country", "has_attachment"]
    results: list[dict] = []

    log("Metadata tests: starting chi-square association checks")
    for field in fields:
        if field not in df.columns:
            continue
        log(f"Metadata tests: field '{field}'")

        if field == "has_attachment":
            grouped = df[field].fillna(False).astype(str)
        else:
            grouped = group_sparse_categories(df[field], top_n=top_n)

        contingency = pd.crosstab(df["cluster_id"], grouped)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, p_value, dof, _ = chi2_contingency(contingency)
        n_total = contingency.to_numpy().sum()
        r_count, c_count = contingency.shape
        phi2 = chi2 / max(n_total, 1)
        cramers_v = np.sqrt(phi2 / max(1, min(r_count - 1, c_count - 1)))

        results.append(
            {
                "field": field,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": float(cramers_v),
                "rows": int(r_count),
                "columns": int(c_count),
            }
        )
        log(
            f"Metadata tests: field '{field}' done "
            f"(p={p_value:.4g}, Cramer's V={cramers_v:.3f})"
        )

    log(f"Metadata tests: completed {len(results)} fields")
    return results


def save_null_distribution_figure(null_df: pd.DataFrame, observed_score: float, output_path: Path) -> None:
    """Save a histogram of null best-silhouette scores with the observed value marked."""

    plt.figure(figsize=(10, 6))
    sns.histplot(null_df["best_silhouette_score"], bins=18, color="#5c7cfa", alpha=0.75)
    plt.axvline(observed_score, color="#d9480f", linestyle="--", linewidth=2, label="Observed best silhouette")
    plt.title("Permutation null distribution for the best silhouette score")
    plt.xlabel("Best silhouette score under shuffled-structure null")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_bootstrap_figure(bootstrap_df: pd.DataFrame, output_path: Path) -> None:
    """Save the distribution of bootstrap ARI values."""

    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_df["adjusted_rand_index"], bins=16, color="#2b8a3e", alpha=0.8)
    plt.axvline(
        bootstrap_df["adjusted_rand_index"].mean(),
        color="#1c7ed6",
        linestyle="--",
        linewidth=2,
        label="Mean bootstrap ARI",
    )
    plt.title("Bootstrap stability of the cluster solution")
    plt.xlabel("Adjusted Rand Index against original labels")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def hopkins_interpretation(statistic: float) -> str:
    """Translate the Hopkins statistic into a cautious plain-language note."""

    if statistic >= 0.75:
        return "strong cluster tendency"
    if statistic >= 0.60:
        return "moderate cluster tendency"
    if statistic >= 0.45:
        return "weak-to-moderate cluster tendency"
    return "little evidence of cluster tendency"


def build_validation_markdown(
    observed_metrics: dict,
    hopkins: dict,
    null_summary: dict,
    bootstrap_df: pd.DataFrame,
    algorithm_agreement: dict,
    metadata_tests: list[dict],
) -> str:
    """Create a compact narrative summary for the report."""

    lines = [
        "# Cluster Validation Notes",
        "",
        "These checks are meant to answer a narrower and better question than \"are the clusters absolutely true?\"",
        "",
        "The aim is to test whether the discovered partition is stronger than a random null baseline and stable enough to interpret as a meaningful pattern in the corpus.",
        "",
        "## Cluster tendency before interpretation",
        "",
        f"- Hopkins statistic: {hopkins['statistic']:.3f} based on {hopkins['sample_size']} sampled observations across {hopkins['dimensions_used']} PCA dimensions.",
        f"- Plain-language reading: {hopkins_interpretation(hopkins['statistic'])}.",
        "",
        "## Internal and null-model checks",
        "",
        f"- Observed cluster count: {observed_metrics['cluster_count']}.",
        f"- Observed silhouette score: {observed_metrics['silhouette_score']:.4f}.",
        f"- Observed Calinski-Harabasz score: {observed_metrics['calinski_harabasz_score']:.2f}.",
        f"- Observed Davies-Bouldin score: {observed_metrics['davies_bouldin_score']:.4f}. Lower is better for Davies-Bouldin.",
        f"- Null mean of the best silhouette score after shuffling structure: {null_summary['null_mean_best_silhouette']:.4f}.",
        f"- Null maximum of the best silhouette score after shuffling structure: {null_summary['null_max_best_silhouette']:.4f}.",
        f"- Permutation p-value for the observed best silhouette: {null_summary['permutation_p_value']:.4f}.",
        "",
        "## Stability under perturbation",
        "",
        f"- Mean bootstrap ARI: {bootstrap_df['adjusted_rand_index'].mean():.4f}.",
        f"- Median bootstrap ARI: {bootstrap_df['adjusted_rand_index'].median():.4f}.",
        f"- Standard deviation of bootstrap ARI: {bootstrap_df['adjusted_rand_index'].std(ddof=1):.4f}.",
        "",
        "## Agreement across clustering families",
        "",
        f"- KMeans vs Agglomerative clustering ARI: {algorithm_agreement['kmeans_vs_agglomerative_ari']:.4f}.",
        f"- KMeans vs Gaussian Mixture ARI: {algorithm_agreement['kmeans_vs_gmm_ari']:.4f}.",
    ]

    if metadata_tests:
        lines.extend(
            [
                "",
                "## Association with metadata",
                "",
                "These p-values do not validate clustering on their own, but they help show whether the semantic partition lines up with meaningful corpus structure.",
                "",
            ]
        )
        for item in metadata_tests:
            lines.append(
                f"- {item['field']}: chi-square={item['chi2']:.2f}, p={item['p_value']:.4g}, Cramer's V={item['cramers_v']:.3f}."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A low permutation p-value means the best observed silhouette is unlikely to arise from a shuffled embedding cloud with the same one-dimensional spread.",
            "",
            "That still does not prove the clusters are ontologically \"real\" in a philosophical sense, but it does support treating them as a meaningful empirical summary rather than a pure visualization artifact.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate whether the discovered cluster structure is stronger than chance.")
    parser.add_argument("--config", default="config/project_config.yml", help="Path to YAML config.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    start_time = time.time()
    log("Starting validation stage")
    config_path = (project_root / args.config).resolve()
    config = load_config(config_path)

    seed = int(config["analysis"]["seed"])
    cluster_range = [int(value) for value in config["analysis"]["cluster_range"]]
    n_null_draws = int(config["analysis"].get("validation_null_draws", 60))
    n_bootstrap_draws = int(config["analysis"].get("validation_bootstrap_draws", 40))
    bootstrap_fraction = float(config["analysis"].get("validation_bootstrap_fraction", 0.8))
    hopkins_samples = int(config["analysis"].get("validation_hopkins_samples", 250))
    top_categories_for_tests = int(config["analysis"].get("validation_top_categories_for_tests", 8))

    embeddings_path = project_path(project_root, config["paths"]["embeddings"])
    cluster_assignments_path = project_path(project_root, config["paths"]["cluster_assignments"])
    model_selection_path = project_path(project_root, config["paths"]["model_selection"])
    validation_summary_path = project_path(project_root, config["paths"]["cluster_validation_summary"])
    validation_resamples_path = project_path(project_root, config["paths"]["cluster_validation_resamples"])
    validation_markdown_path = project_path(project_root, config["paths"]["cluster_validation_markdown"])
    figures_dir = project_path(project_root, config["paths"]["figures_dir"])

    null_figure_path = figures_dir / "cluster_validation_null_distribution.png"
    bootstrap_figure_path = figures_dir / "cluster_validation_bootstrap_ari.png"

    ensure_parent(validation_summary_path)
    ensure_parent(validation_resamples_path)
    ensure_parent(validation_markdown_path)
    ensure_parent(null_figure_path)
    ensure_parent(bootstrap_figure_path)

    embeddings = np.load(embeddings_path)
    assignments = pd.read_csv(cluster_assignments_path)
    model_selection = pd.read_csv(model_selection_path)
    log(f"Loaded embeddings with shape {embeddings.shape}")
    log(f"Loaded cluster assignments with {len(assignments)} rows")

    log("Rebuilding PCA representation for validation")
    pca_embeddings = build_pca_representation(embeddings, seed=seed)
    labels = assignments["cluster_id"].to_numpy(dtype=int)
    cluster_count = int(len(np.unique(labels)))
    log(f"Validation will assess {cluster_count} clusters")

    observed_metrics = {
        "cluster_count": cluster_count,
        "silhouette_score": float(silhouette_score(pca_embeddings, labels, metric="euclidean")),
        "calinski_harabasz_score": float(calinski_harabasz_score(pca_embeddings, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(pca_embeddings, labels)),
        "selected_from_model_search": model_selection.sort_values("silhouette_score", ascending=False).iloc[0].to_dict(),
    }

    log("Computing Hopkins statistic")
    hopkins = compute_hopkins_statistic(pca_embeddings, sample_size=hopkins_samples, seed=seed)

    log("Computing permutation null distribution")
    null_df, null_summary = compute_null_distribution(
        pca_embeddings,
        cluster_range=cluster_range,
        observed_best_silhouette=observed_metrics["silhouette_score"],
        observed_k=cluster_count,
        seed=seed,
        n_draws=n_null_draws,
    )

    log("Computing bootstrap stability")
    bootstrap_df = compute_bootstrap_stability(
        pca_embeddings,
        reference_labels=labels,
        cluster_count=cluster_count,
        seed=seed,
        n_draws=n_bootstrap_draws,
        sample_fraction=bootstrap_fraction,
    )

    log("Computing algorithm agreement")
    algorithm_agreement = compute_algorithm_agreement(
        pca_embeddings,
        reference_labels=labels,
        cluster_count=cluster_count,
        seed=seed,
    )

    log("Computing metadata association tests")
    metadata_tests = compute_metadata_association_tests(assignments, top_n=top_categories_for_tests)

    log("Saving validation figures")
    save_null_distribution_figure(null_df, observed_metrics["silhouette_score"], null_figure_path)
    log(f"Saved null distribution figure to: {null_figure_path}")
    save_bootstrap_figure(bootstrap_df, bootstrap_figure_path)
    log(f"Saved bootstrap figure to: {bootstrap_figure_path}")

    resamples_df = null_df.copy()
    resamples_df["resample_type"] = "permutation_null"

    bootstrap_export = bootstrap_df.copy()
    bootstrap_export["resample_type"] = "bootstrap_stability"

    log(f"Writing resample table to: {validation_resamples_path}")
    pd.concat([resamples_df, bootstrap_export], ignore_index=True, sort=False).to_csv(
        validation_resamples_path,
        index=False,
        encoding="utf-8",
    )
    log("Writing validation summary and markdown")

    summary_payload = {
        "observed_metrics": observed_metrics,
        "hopkins": hopkins,
        "null_summary": null_summary,
        "bootstrap_summary": {
            "draws": int(len(bootstrap_df)),
            "mean_adjusted_rand_index": float(bootstrap_df["adjusted_rand_index"].mean()),
            "median_adjusted_rand_index": float(bootstrap_df["adjusted_rand_index"].median()),
            "std_adjusted_rand_index": float(bootstrap_df["adjusted_rand_index"].std(ddof=1)),
            "mean_silhouette_score": float(bootstrap_df["silhouette_score"].mean()),
        },
        "algorithm_agreement": algorithm_agreement,
        "metadata_association_tests": metadata_tests,
        "files": {
            "null_distribution_figure": str(null_figure_path),
            "bootstrap_figure": str(bootstrap_figure_path),
        },
    }

    with validation_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
    log(f"Validation summary written to: {validation_summary_path}")

    validation_markdown = build_validation_markdown(
        observed_metrics=observed_metrics,
        hopkins=hopkins,
        null_summary=null_summary,
        bootstrap_df=bootstrap_df,
        algorithm_agreement=algorithm_agreement,
        metadata_tests=metadata_tests,
    )
    validation_markdown_path.write_text(validation_markdown, encoding="utf-8")
    log(f"Validation markdown written to: {validation_markdown_path}")
    elapsed = time.time() - start_time
    log(f"Validation stage complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
