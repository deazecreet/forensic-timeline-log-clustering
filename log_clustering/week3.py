"""Week 3 visualization and forensic interpretability analysis."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .week1 import (
    DEFAULT_MAX_YEAR,
    DEFAULT_MIN_YEAR,
    build_stratified_quotas,
    profile_csv,
    sample_stratified_rows,
)
from .week2 import (
    DEFAULT_CLUSTERING_METHODS,
    DEFAULT_REPRESENTATIONS,
    _format_elapsed,
    _progress,
    _run_clustering_trial,
    load_or_build_embeddings,
    prepare_clustering_matrix,
)


DEFAULT_SCATTER_METHODS = ["pca", "umap", "tsne"]
REPORT_COMBO_ORDER = {
    (representation, method): index
    for index, (representation, method) in enumerate(
        (r, m) for r in DEFAULT_REPRESENTATIONS for m in DEFAULT_CLUSTERING_METHODS
    )
}


def _parse_csv_list(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    return values


def _safe_filename(*parts: Any) -> str:
    return "_".join(str(part).strip().lower().replace(" ", "_").replace("/", "_") for part in parts)


def _load_week2_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Week 2 {name}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Week 2 {name} is empty: {path}")
    return df


def _load_week2_config(week2_dir: Path) -> dict[str, Any]:
    path = week2_dir / "run_config_week2.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_metrics(metrics_df: pd.DataFrame, representations: list[str], methods: list[str]) -> pd.DataFrame:
    selected = metrics_df[
        metrics_df["representation"].isin(representations)
        & metrics_df["clustering_method"].isin(methods)
    ].copy()
    if selected.empty:
        raise ValueError("No matching Week 2 metrics for the requested representations/methods")
    selected["_combo_order"] = selected.apply(
        lambda row: REPORT_COMBO_ORDER.get((row["representation"], row["clustering_method"]), 10_000),
        axis=1,
    )
    return selected.sort_values(["_combo_order", "representation", "clustering_method"]).drop(
        columns=["_combo_order"]
    )


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if pd.isna(raw) or str(raw).strip() == "":
        return {}
    return json.loads(str(raw))


def _load_or_rebuild_sample(
    input_path: Path,
    *,
    sample_size: int,
    seed: int,
    min_year: int,
    max_year: int,
    output_dir: Path,
    verbose: bool,
) -> pd.DataFrame:
    sample_path = output_dir / "sample_metadata_week3.csv"
    if sample_path.exists():
        _progress(f"Load sample metadata Week 3: {sample_path}", verbose=verbose)
        return pd.read_csv(sample_path)

    _progress("Profiling CSV untuk merekonstruksi sample Week 2...", verbose=verbose)
    profile = profile_csv(input_path, min_year=min_year, max_year=max_year)
    quotas = build_stratified_quotas(profile["source_counts_valid"], sample_size)
    _progress("Sampling stratified rows untuk Week 3...", verbose=verbose)
    sample_df = sample_stratified_rows(input_path, quotas, seed=seed, min_year=min_year, max_year=max_year)
    sample_df = sample_df.reset_index(drop=True)
    sample_df.insert(0, "sample_index", np.arange(len(sample_df), dtype=int))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    return sample_df


def _select_visualization_indices(n_rows: int, max_points: int, seed: int) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def _reduce_2d(matrix: np.ndarray, method: str, *, seed: int, perplexity: int, n_neighbors: int) -> np.ndarray:
    if matrix.shape[0] < 3:
        return PCA(n_components=min(2, matrix.shape[1]), random_state=seed).fit_transform(matrix)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(matrix)
    if method == "tsne":
        effective_perplexity = max(2, min(perplexity, (matrix.shape[0] - 1) // 3))
        return TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(matrix)
    if method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError("UMAP visualization needs umap-learn. Install with: pip install umap-learn") from exc
        return UMAP(
            n_components=2,
            n_neighbors=max(2, min(n_neighbors, matrix.shape[0] - 1)),
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(matrix)
    raise ValueError(f"Unknown visualization method: {method}")


def _cluster_colors(labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    labels = np.asarray(labels)
    cluster_counts = Counter(int(label) for label in labels if int(label) != -1)
    major_clusters = [cluster for cluster, _ in cluster_counts.most_common(20)]
    color_ids = np.full(labels.shape, -2, dtype=int)
    color_ids[labels == -1] = -1
    for index, cluster_id in enumerate(major_clusters):
        color_ids[labels == cluster_id] = index
    return color_ids, major_clusters


def _plot_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    color_ids, major_clusters = _cluster_colors(labels)
    plt.figure(figsize=(8, 6))
    other_mask = color_ids == -2
    if np.any(other_mask):
        plt.scatter(coords[other_mask, 0], coords[other_mask, 1], s=5, c="#c7c7c7", alpha=0.25, label="other")
    noise_mask = color_ids == -1
    if np.any(noise_mask):
        plt.scatter(coords[noise_mask, 0], coords[noise_mask, 1], s=6, c="#4d4d4d", alpha=0.35, label="noise")
    cmap = plt.get_cmap("tab20")
    for index, cluster_id in enumerate(major_clusters):
        mask = color_ids == index
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=7,
            color=cmap(index % 20),
            alpha=0.75,
            label=f"c{cluster_id}",
        )
    plt.title(title)
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.grid(alpha=0.15)
    if len(major_clusters) <= 12:
        plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_timeline(
    sample_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    title: str,
    output_path: Path,
    max_points: int,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(sample_df["datetime"], errors="coerce", utc=True)
    valid_mask = dates.notna().to_numpy()
    candidate_indices = np.flatnonzero(valid_mask)
    if candidate_indices.size == 0:
        return
    if candidate_indices.size > max_points:
        rng = np.random.default_rng(seed)
        candidate_indices = np.sort(rng.choice(candidate_indices, size=max_points, replace=False))
    plot_labels = labels[candidate_indices]
    color_ids, _ = _cluster_colors(plot_labels)
    plt.figure(figsize=(10, 5))
    plt.scatter(dates.iloc[candidate_indices], plot_labels, c=color_ids, cmap="tab20", s=6, alpha=0.55)
    plt.title(title)
    plt.xlabel("datetime")
    plt.ylabel("cluster id")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_source_distribution(source_df: pd.DataFrame, *, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_df.empty:
        return
    top_clusters = (
        source_df[source_df["cluster_id"] != -1]
        .groupby("cluster_id")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    plot_df = source_df[source_df["cluster_id"].isin(top_clusters)].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot_table(index="cluster_id", columns="source", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5), width=0.85)
    ax.set_title(title)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("event count")
    ax.legend(title="source", fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _token_counter(cluster_rows: pd.DataFrame, top_clusters: int = 8) -> Counter[str]:
    counter: Counter[str] = Counter()
    rows = cluster_rows[cluster_rows["cluster_id"] != -1].sort_values("size", ascending=False).head(top_clusters)
    for _, row in rows.iterrows():
        for item in str(row.get("top_tokens", "")).split(";"):
            token = item.strip()
            if token:
                counter[token] += 1
    return counter


def _plot_top_terms(cluster_rows: pd.DataFrame, *, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counter = _token_counter(cluster_rows)
    if not counter:
        return
    tokens, counts = zip(*counter.most_common(20))
    plt.figure(figsize=(9, 5))
    y = np.arange(len(tokens))
    plt.barh(y, counts, color="#386cb0")
    plt.yticks(y, tokens)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("frequency across largest cluster top-token lists")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_dendrogram(matrix: np.ndarray, *, title: str, output_path: Path, max_points: int, seed: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.shape[0] < 3:
        return
    indices = _select_visualization_indices(matrix.shape[0], max_points, seed)
    subset = matrix[indices]
    linked = linkage(subset, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(linked, no_labels=True, color_threshold=None)
    plt.title(title)
    plt.xlabel(f"sampled events (n={len(indices)})")
    plt.ylabel("ward distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _source_purity(source_rows: pd.DataFrame) -> float:
    if source_rows.empty:
        return float("nan")
    cluster_totals = source_rows.groupby("cluster_id")["count"].sum()
    dominant = source_rows.groupby("cluster_id")["count"].max()
    weighted = (dominant / cluster_totals * cluster_totals).sum()
    total = cluster_totals.sum()
    return float(weighted / total) if total else float("nan")


def _token_clarity(cluster_rows: pd.DataFrame) -> float:
    rows = cluster_rows[cluster_rows["cluster_id"] != -1].copy()
    if rows.empty:
        return float("nan")
    clarity_values = []
    weights = []
    for _, row in rows.iterrows():
        tokens = [item.strip() for item in str(row.get("top_tokens", "")).split(";") if item.strip()]
        if not tokens:
            continue
        unique_ratio = len(set(tokens[:8])) / min(8, len(tokens))
        clarity_values.append(unique_ratio)
        weights.append(float(row.get("size", 1)))
    if not clarity_values:
        return float("nan")
    return float(np.average(clarity_values, weights=weights))


def _interpretability_score(
    metric_row: pd.Series,
    cluster_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> tuple[int, str, dict[str, float]]:
    silhouette = float(metric_row.get("silhouette", np.nan))
    noise_ratio = float(metric_row.get("noise_ratio", 0.0) or 0.0)
    n_clusters = float(metric_row.get("n_clusters", 0.0) or 0.0)
    purity = _source_purity(source_rows)
    clarity = _token_clarity(cluster_rows)

    score = 1.0
    if not math.isnan(silhouette):
        score += max(0.0, min(1.5, silhouette * 1.5))
    if not math.isnan(purity):
        score += max(0.0, min(1.0, (purity - 0.35) / 0.65))
    if not math.isnan(clarity):
        score += max(0.0, min(0.8, clarity * 0.8))
    if 5 <= n_clusters <= 250:
        score += 0.5
    elif n_clusters > 250:
        score += 0.2
    if noise_ratio <= 0.2:
        score += 0.4
    elif noise_ratio <= 0.35:
        score += 0.2
    final_score = int(max(1, min(5, round(score))))

    reasons = []
    if not math.isnan(silhouette):
        reasons.append(f"Silhouette {silhouette:.3f}")
    reasons.append(f"{int(n_clusters)} clusters")
    if noise_ratio:
        reasons.append(f"noise {noise_ratio:.1%}")
    if not math.isnan(purity):
        reasons.append(f"source purity {purity:.1%}")
    if not math.isnan(clarity):
        reasons.append(f"top-token clarity {clarity:.1%}")
    return final_score, "; ".join(reasons), {
        "source_purity": purity,
        "token_clarity": clarity,
    }


def _build_interpretability_tables(
    metrics_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combo_rows: list[dict[str, Any]] = []
    cluster_rows_out: list[dict[str, Any]] = []

    for _, metric_row in metrics_df.iterrows():
        representation = metric_row["representation"]
        method = metric_row["clustering_method"]
        combo_cluster_df = cluster_df[
            (cluster_df["representation"] == representation) & (cluster_df["clustering_method"] == method)
        ].copy()
        combo_source_df = source_df[
            (source_df["representation"] == representation) & (source_df["clustering_method"] == method)
        ].copy()
        score, rationale, extras = _interpretability_score(metric_row, combo_cluster_df, combo_source_df)
        largest = combo_cluster_df[combo_cluster_df["cluster_id"] != -1].sort_values("size", ascending=False).head(5)
        combo_rows.append(
            {
                "representation": representation,
                "clustering_method": method,
                "forensic_interpretability_score": score,
                "rationale": rationale,
                "n_clusters": metric_row.get("n_clusters"),
                "noise_ratio": metric_row.get("noise_ratio"),
                "silhouette": metric_row.get("silhouette"),
                "calinski_harabasz": metric_row.get("calinski_harabasz"),
                "davies_bouldin": metric_row.get("davies_bouldin"),
                "source_purity": extras["source_purity"],
                "token_clarity": extras["token_clarity"],
                "largest_cluster_ids": ", ".join(str(int(value)) for value in largest["cluster_id"].tolist()),
            }
        )

        for _, cluster_row in combo_cluster_df.sort_values("size", ascending=False).head(20).iterrows():
            top_sources = str(cluster_row.get("top_sources", ""))
            cluster_score = 1
            if cluster_row.get("is_noise") is True or int(cluster_row.get("cluster_id", 0)) == -1:
                cluster_score = 1
            else:
                source_parts = [item for item in top_sources.split(";") if item.strip()]
                token_parts = [item for item in str(cluster_row.get("top_tokens", "")).split(";") if item.strip()]
                cluster_score += 1 if len(source_parts) <= 3 else 0
                cluster_score += 1 if len(token_parts) >= 5 else 0
                cluster_score += 1 if int(cluster_row.get("size", 0)) >= 25 else 0
                cluster_score += 1 if any(
                    key in str(cluster_row.get("top_tokens", "")).lower()
                    for key in ["source_", "registry", "browser", "event", "path_token", "download", "run"]
                ) else 0
                cluster_score = min(5, cluster_score)
            cluster_rows_out.append(
                {
                    "representation": representation,
                    "clustering_method": method,
                    "cluster_id": cluster_row.get("cluster_id"),
                    "size": cluster_row.get("size"),
                    "is_noise": cluster_row.get("is_noise"),
                    "forensic_cluster_score": cluster_score,
                    "top_tokens": cluster_row.get("top_tokens"),
                    "top_sources": top_sources,
                    "example_messages": cluster_row.get("example_messages"),
                }
            )

    combo_df = pd.DataFrame(combo_rows).sort_values(
        ["forensic_interpretability_score", "silhouette"], ascending=[False, False]
    )
    cluster_out_df = pd.DataFrame(cluster_rows_out).sort_values(
        ["representation", "clustering_method", "forensic_cluster_score", "size"],
        ascending=[True, True, False, False],
    )
    return combo_df, cluster_out_df


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Tidak ada data._"
    safe_df = df.copy()
    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("\n", " ").replace("|", "\\|")
        )
    headers = list(safe_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in safe_df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    output_dir: Path,
    *,
    metrics_df: pd.DataFrame,
    interpretability_df: pd.DataFrame,
    run_summary: dict[str, Any],
) -> None:
    top_quality = metrics_df.sort_values("silhouette", ascending=False).head(8)
    top_interpretability = interpretability_df.head(8)
    lines = [
        "# WEEK 3 Progress - Visualization and Forensic Interpretability",
        "",
        "## Scope",
        "",
        "Pekan 3 mengintegrasikan hasil representasi dan clustering Pekan 2 ke tahap visualisasi serta interpretasi forensik. Fokusnya bukan menjalankan full 5-seed experiment, melainkan membaca struktur cluster yang sudah dihasilkan dari skenario inti Pekan 2.",
        "",
        "## Input",
        "",
        f"- Dataset: `{run_summary['input_file']}`",
        f"- Week 2 report folder: `{run_summary['week2_dir']}`",
        f"- Sample size: {run_summary['sample_size']:,} event",
        f"- Seed: {run_summary['seed']}",
        f"- Representasi: {', '.join(run_summary['representations'])}",
        f"- Clustering: {', '.join(run_summary['clustering_methods'])}",
        "",
        "## Artefak Yang Dihasilkan",
        "",
        "- `sample_metadata_week3.csv`: sample event yang dipakai untuk visualisasi dan rekonstruksi label.",
        "- `labels/*.csv`: label cluster per event untuk setiap kombinasi representasi dan clustering.",
        "- `scatter/{pca,umap,tsne}/*.png`: scatter plot 2D per kombinasi.",
        "- `dendrograms/*.png`: dendrogram untuk Agglomerative Clustering per representasi.",
        "- `timelines/*.png`: visualisasi datetime vs cluster.",
        "- `source_distribution/*.png`: distribusi source pada cluster terbesar.",
        "- `top_terms/*.png`: ringkasan token dominan pada cluster terbesar.",
        "- `interpretability_assessment_week3.csv`: skor interpretabilitas level kombinasi.",
        "- `cluster_interpretability_week3.csv`: skor interpretabilitas cluster penting.",
        "",
        "## Kombinasi Dengan Silhouette Tertinggi",
        "",
        _markdown_table(
            top_quality[
                ["representation", "clustering_method", "n_clusters", "noise_ratio", "silhouette", "davies_bouldin"]
            ]
        ),
        "",
        "## Kombinasi Dengan Interpretabilitas Tertinggi",
        "",
        _markdown_table(
            top_interpretability[
                [
                    "representation",
                    "clustering_method",
                    "forensic_interpretability_score",
                    "n_clusters",
                    "noise_ratio",
                    "silhouette",
                    "source_purity",
                    "rationale",
                ]
            ]
        ),
        "",
        "## Cara Membaca Hasil",
        "",
        "1. Mulai dari `interpretability_assessment_week3.csv` untuk memilih kombinasi yang bukan hanya tinggi metriknya, tetapi juga mudah dijelaskan secara forensik.",
        "2. Buka scatter plot PCA/UMAP/t-SNE untuk kombinasi tersebut. Cluster yang baik biasanya tampak punya separasi visual, tidak hanya angka metrik yang tinggi.",
        "3. Bandingkan dengan `source_distribution`. Cluster yang didominasi source tertentu lebih mudah dihubungkan dengan aktivitas forensik, misalnya file-system, registry, event log, atau browser history.",
        "4. Gunakan `top_terms` dan `cluster_interpretability_week3.csv` untuk menamai cluster secara manual.",
        "5. Gunakan timeline plot untuk melihat apakah cluster tertentu terkonsentrasi pada periode waktu tertentu.",
        "",
        "## Interpretasi Awal",
        "",
        "Metrik internal dari Pekan 2 menunjukkan bahwa kombinasi density-based seperti DBSCAN/HDBSCAN dapat menghasilkan Silhouette tinggi, terutama ketika noise dikeluarkan dari perhitungan cluster utama. Namun untuk kebutuhan forensic investigation, angka tersebut harus dibaca bersama noise ratio, jumlah cluster, dominasi source, dan contoh message. Cluster yang terlalu banyak dapat bagus secara pemisahan lokal, tetapi lebih sulit dipakai investigator jika tidak memiliki tema forensik yang jelas.",
        "",
        "TF-IDF cenderung sangat interpretabel karena top token langsung berasal dari isi log dan token source. SBERT dapat menangkap kemiripan semantik yang lebih halus, tetapi hasilnya perlu dibantu source distribution dan example message agar mudah dijelaskan.",
        "",
        "## Batasan",
        "",
        "- Visualisasi t-SNE/UMAP memakai subset titik agar proses tetap realistis pada 50.000 event.",
        "- Dendrogram Agglomerative juga memakai subset karena hierarchical linkage penuh pada 50.000 event terlalu berat untuk inspeksi visual.",
        "- Skor interpretabilitas bersifat heuristic awal. Nilai ini perlu dibaca sebagai panduan prioritas inspeksi, bukan label kebenaran forensik final.",
        "- Full 5-seed execution, k-sensitivity, dan cross-dataset validation tetap masuk ruang Pekan 4.",
        "",
    ]
    (output_dir / "WEEK3_PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")


def run_week3(
    *,
    input_path: Path,
    week2_dir: Path,
    output_dir: Path,
    sample_size: int | None = None,
    seed: int | None = None,
    representations: list[str] | None = None,
    clustering_methods: list[str] | None = None,
    max_viz_points: int = 4_000,
    max_timeline_points: int = 10_000,
    max_dendrogram_points: int = 1_000,
    scatter_methods: list[str] | None = None,
    perplexity: int = 30,
    umap_neighbors: int = 15,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build Week 3 visualizations and interpretability reports from Week 2 outputs."""
    input_path = Path(input_path)
    week2_dir = Path(week2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_week2_config(week2_dir)
    sample_config = config.get("sample", {})
    embedding_params = config.get("embedding_params", {})
    sample_size = int(sample_size or sample_config.get("actual_sample_size") or sample_config.get("requested_sample_size") or 50_000)
    seed = int(seed if seed is not None else sample_config.get("seed", 42))
    representations = representations or DEFAULT_REPRESENTATIONS
    clustering_methods = clustering_methods or DEFAULT_CLUSTERING_METHODS
    scatter_methods = scatter_methods or DEFAULT_SCATTER_METHODS

    embedding_profile_path = week2_dir / "embedding_profile_week2.csv"
    embedding_profile_df = pd.read_csv(embedding_profile_path) if embedding_profile_path.exists() else pd.DataFrame()
    device_by_representation = (
        embedding_profile_df.dropna(subset=["representation"])
        .drop_duplicates("representation", keep="last")
        .set_index("representation")["device"]
        .to_dict()
        if not embedding_profile_df.empty and "device" in embedding_profile_df.columns
        else {}
    )

    metrics_df = _best_metrics(
        _load_week2_table(week2_dir / "metrics_week2.csv", "metrics"),
        representations,
        clustering_methods,
    )
    cluster_summary_df = _load_week2_table(week2_dir / "cluster_summary_week2.csv", "cluster summary")
    source_by_cluster_df = _load_week2_table(week2_dir / "source_by_cluster_week2.csv", "source-by-cluster")

    _progress("Week 3 visualization + interpretability dimulai", verbose=verbose)
    _progress(f"Input: {input_path}", verbose=verbose)
    _progress(f"Week 2 dir: {week2_dir}", verbose=verbose)
    _progress(f"Output: {output_dir}", verbose=verbose)
    _progress(f"Sample size: {sample_size:,} | seed={seed}", verbose=verbose)

    sample_df = _load_or_rebuild_sample(
        input_path,
        sample_size=sample_size,
        seed=seed,
        min_year=min_year,
        max_year=max_year,
        output_dir=output_dir,
        verbose=verbose,
    )
    viz_indices = _select_visualization_indices(len(sample_df), max_viz_points, seed)

    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []
    labels_by_combo: dict[tuple[str, str], np.ndarray] = {}
    matrix_by_representation: dict[str, np.ndarray] = {}
    coords_by_representation: dict[tuple[str, str], np.ndarray] = {}

    for representation_index, representation in enumerate(representations, start=1):
        start = time.perf_counter()
        _progress(f"[{representation_index}/{len(representations)}] Load embedding {representation}...", verbose=verbose)
        sbert_device = embedding_params.get("sbert_device_resolved") or embedding_params.get("sbert_device_requested") or "auto"
        if representation == "sbert":
            sbert_device = device_by_representation.get("sbert") or sbert_device
        embeddings, _ = load_or_build_embeddings(
            sample_df,
            representation,
            input_path=input_path,
            output_dir=week2_dir,
            sample_size=sample_size,
            seed=seed,
            min_year=min_year,
            max_year=max_year,
            vector_size=int(embedding_params.get("word2vec_doc2vec_vector_size", 100)),
            window=int(embedding_params.get("window", 5)),
            min_count=int(embedding_params.get("min_count", 2)),
            epochs=int(embedding_params.get("epochs", 20)),
            workers=int(embedding_params.get("workers", 1)),
            tfidf_max_features=int(embedding_params.get("tfidf_max_features", 5000)),
            tfidf_min_df=int(embedding_params.get("tfidf_min_df", 5)),
            tfidf_max_df=float(embedding_params.get("tfidf_max_df", 0.95)),
            sbert_model=str(embedding_params.get("sbert_model", "all-MiniLM-L6-v2")),
            sbert_batch_size=int(embedding_params.get("sbert_batch_size", 64)),
            sbert_device=str(sbert_device),
            use_cache=True,
            verbose=verbose,
        )
        matrix, _ = prepare_clustering_matrix(
            embeddings,
            pca_components=int(config.get("pca_components", 50)),
            seed=seed,
        )
        matrix_by_representation[representation] = matrix
        _progress(
            f"  Matrix {representation} siap dalam {_format_elapsed(time.perf_counter() - start)}",
            verbose=verbose,
        )

        viz_matrix = matrix[viz_indices]
        for scatter_method in scatter_methods:
            stage_start = time.perf_counter()
            _progress(f"  Reduksi 2D {scatter_method.upper()} untuk {representation}...", verbose=verbose)
            coords_by_representation[(representation, scatter_method)] = _reduce_2d(
                viz_matrix,
                scatter_method,
                seed=seed,
                perplexity=perplexity,
                n_neighbors=umap_neighbors,
            )
            _progress(
                f"  {scatter_method.upper()} selesai dalam {_format_elapsed(time.perf_counter() - stage_start)}",
                verbose=verbose,
            )

        if "agglomerative" in clustering_methods:
            dendro_path = output_dir / "dendrograms" / f"{_safe_filename(representation, 'agglomerative')}_dendrogram.png"
            _plot_dendrogram(
                matrix,
                title=f"Dendrogram - {representation} + Agglomerative",
                output_path=dendro_path,
                max_points=max_dendrogram_points,
                seed=seed,
            )
            generated_files.append(str(dendro_path))

        representation_metrics = metrics_df[metrics_df["representation"] == representation]
        for _, metric_row in representation_metrics.iterrows():
            method = str(metric_row["clustering_method"])
            params = _parse_params(metric_row["params"])
            _progress(f"  Rekonstruksi label {representation}+{method}...", verbose=verbose)
            labels, _ = _run_clustering_trial(matrix, method, params, seed=seed)
            labels = np.asarray(labels, dtype=int)
            labels_by_combo[(representation, method)] = labels

            label_df = pd.DataFrame(
                {
                    "sample_index": sample_df["sample_index"],
                    "datetime": sample_df["datetime"],
                    "source": sample_df["source"],
                    "cluster_id": labels,
                    "is_noise": labels == -1,
                }
            )
            label_path = label_dir / f"{_safe_filename(representation, method)}_labels.csv"
            label_df.to_csv(label_path, index=False)
            generated_files.append(str(label_path))

            for scatter_method in scatter_methods:
                coords = coords_by_representation[(representation, scatter_method)]
                scatter_path = (
                    output_dir
                    / "scatter"
                    / scatter_method
                    / f"{_safe_filename(representation, method, scatter_method)}.png"
                )
                _plot_scatter(
                    coords,
                    labels[viz_indices],
                    title=f"{scatter_method.upper()} - {representation} + {method}",
                    output_path=scatter_path,
                )
                generated_files.append(str(scatter_path))

            timeline_path = output_dir / "timelines" / f"{_safe_filename(representation, method)}_timeline.png"
            _plot_timeline(
                sample_df,
                labels,
                title=f"Timeline - {representation} + {method}",
                output_path=timeline_path,
                max_points=max_timeline_points,
                seed=seed,
            )
            generated_files.append(str(timeline_path))

            combo_source_df = source_by_cluster_df[
                (source_by_cluster_df["representation"] == representation)
                & (source_by_cluster_df["clustering_method"] == method)
            ]
            source_path = output_dir / "source_distribution" / f"{_safe_filename(representation, method)}_source.png"
            _plot_source_distribution(
                combo_source_df,
                title=f"Source Distribution - {representation} + {method}",
                output_path=source_path,
            )
            generated_files.append(str(source_path))

            combo_cluster_df = cluster_summary_df[
                (cluster_summary_df["representation"] == representation)
                & (cluster_summary_df["clustering_method"] == method)
            ]
            terms_path = output_dir / "top_terms" / f"{_safe_filename(representation, method)}_top_terms.png"
            _plot_top_terms(
                combo_cluster_df,
                title=f"Top Terms - {representation} + {method}",
                output_path=terms_path,
            )
            generated_files.append(str(terms_path))

    interpretability_df, cluster_interpretability_df = _build_interpretability_tables(
        metrics_df,
        cluster_summary_df[
            cluster_summary_df["representation"].isin(representations)
            & cluster_summary_df["clustering_method"].isin(clustering_methods)
        ],
        source_by_cluster_df[
            source_by_cluster_df["representation"].isin(representations)
            & source_by_cluster_df["clustering_method"].isin(clustering_methods)
        ],
    )
    interpretability_path = output_dir / "interpretability_assessment_week3.csv"
    cluster_interpretability_path = output_dir / "cluster_interpretability_week3.csv"
    interpretability_df.to_csv(interpretability_path, index=False)
    cluster_interpretability_df.to_csv(cluster_interpretability_path, index=False)
    generated_files.extend([str(interpretability_path), str(cluster_interpretability_path)])

    run_summary = {
        "input_file": str(input_path),
        "week2_dir": str(week2_dir),
        "output_dir": str(output_dir),
        "sample_size": int(len(sample_df)),
        "seed": seed,
        "representations": representations,
        "clustering_methods": clustering_methods,
        "scatter_methods": scatter_methods,
        "max_viz_points": max_viz_points,
        "max_timeline_points": max_timeline_points,
        "max_dendrogram_points": max_dendrogram_points,
        "generated_file_count": 0,
    }
    _write_markdown_report(
        output_dir,
        metrics_df=metrics_df,
        interpretability_df=interpretability_df,
        run_summary=run_summary,
    )
    generated_files.extend([str(output_dir / "run_config_week3.json"), str(output_dir / "WEEK3_PROGRESS.md")])
    run_summary["generated_file_count"] = len(generated_files)
    (output_dir / "run_config_week3.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    _progress(f"Week 3 selesai. Generated files: {len(generated_files)}", verbose=verbose)

    return {
        "sample": sample_df,
        "metrics": metrics_df,
        "interpretability": interpretability_df,
        "cluster_interpretability": cluster_interpretability_df,
        "labels_by_combo": labels_by_combo,
        "generated_files": generated_files,
        "run_summary": run_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 3 visualization and forensic interpretability analysis.")
    parser.add_argument("--input", type=Path, default=Path("timeline.csv"), help="Path to Plaso timeline CSV.")
    parser.add_argument("--week2-dir", type=Path, default=Path("reports/week2"), help="Week 2 report directory.")
    parser.add_argument("--output", type=Path, default=Path("reports/week3"), help="Output report directory.")
    parser.add_argument("--sample-size", type=int, default=None, help="Override Week 2 sample size.")
    parser.add_argument("--seed", type=int, default=None, help="Override Week 2 seed.")
    parser.add_argument("--representations", type=_parse_csv_list, default=None, help="Comma-separated representations.")
    parser.add_argument("--clustering-methods", type=_parse_csv_list, default=None, help="Comma-separated clustering methods.")
    parser.add_argument(
        "--scatter-methods",
        type=_parse_csv_list,
        default=DEFAULT_SCATTER_METHODS,
        help="Comma-separated visualization methods: pca,umap,tsne.",
    )
    parser.add_argument("--max-viz-points", type=int, default=4_000, help="Max points for PCA/UMAP/t-SNE plots.")
    parser.add_argument("--max-timeline-points", type=int, default=10_000, help="Max points for timeline plots.")
    parser.add_argument("--max-dendrogram-points", type=int, default=1_000, help="Max points for dendrogram plots.")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR, help="Minimum valid datetime year.")
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR, help="Maximum valid datetime year.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_week3(
        input_path=args.input,
        week2_dir=args.week2_dir,
        output_dir=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        representations=args.representations,
        clustering_methods=args.clustering_methods,
        max_viz_points=args.max_viz_points,
        max_timeline_points=args.max_timeline_points,
        max_dendrogram_points=args.max_dendrogram_points,
        scatter_methods=args.scatter_methods,
        perplexity=args.perplexity,
        umap_neighbors=args.umap_neighbors,
        min_year=args.min_year,
        max_year=args.max_year,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
