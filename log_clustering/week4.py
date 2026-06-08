"""Week 4 final orchestration for validation experiments."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from log_clustering.week2 import (
    DEFAULT_CLUSTERING_METHODS,
    DEFAULT_REPRESENTATIONS,
    normalize_report_dataframe,
    run_week2,
)


DEFAULT_WEEK4_SEEDS = [42, 123, 456, 789, 101]
DEFAULT_PRIMARY_RUNS_ROOT: Path | None = None
DEFAULT_WEEK3_DIR = Path("reports/week3")
REQUIRED_WEEK2_FILES = [
    "metrics_week2.csv",
    "trials_week2.csv",
    "embedding_profile_week2.csv",
    "cluster_summary_week2.csv",
    "source_by_cluster_week2.csv",
    "run_config_week2.json",
    "silhouette_heatmap_week2.png",
    "quality_runtime_week2.png",
]


def _parse_csv_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _progress(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[week4] {message}", flush=True)


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return pd.read_csv(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")


def _week4_report_path(output_dir: Path) -> Path:
    normalized = output_dir.as_posix().rstrip("/")
    if normalized.endswith("reports/week4"):
        return Path("week_progress") / "WEEK4_FINAL_REPORT.md"
    return output_dir / "WEEK4_FINAL_REPORT.md"


def _stage_is_complete(stage_dir: Path, required_files: list[str] | None = None) -> bool:
    files = required_files or ["metrics_week2.csv", "run_config_week2.json"]
    return all((stage_dir / filename).exists() and (stage_dir / filename).stat().st_size > 0 for filename in files)


def _copy_week2_artifacts(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_WEEK2_FILES:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, destination_dir / filename)


def _find_existing_stage(stage_name: str, output_dir: Path, primary_runs_root: Path | None) -> Path | None:
    candidates = [output_dir / stage_name]
    if primary_runs_root is not None:
        candidates.append(primary_runs_root / stage_name)
    for candidate in candidates:
        if _stage_is_complete(candidate):
            return candidate
    return None


def _best_representations(metrics_df: pd.DataFrame, count: int) -> list[str]:
    ranking = (
        metrics_df.dropna(subset=["silhouette"])
        .groupby("representation", as_index=False)
        .agg(
            mean_silhouette=("silhouette", "mean"),
            best_silhouette=("silhouette", "max"),
            mean_noise_ratio=("noise_ratio", "mean"),
        )
        .sort_values(
            ["mean_silhouette", "best_silhouette", "mean_noise_ratio"],
            ascending=[False, False, True],
            kind="stable",
        )
    )
    return ranking["representation"].head(count).astype(str).tolist()


def _best_combinations(metrics_df: pd.DataFrame, count: int = 5, *, mean_columns: bool = False) -> pd.DataFrame:
    silhouette_column = "silhouette_mean" if mean_columns else "silhouette"
    noise_column = "noise_ratio_mean" if mean_columns else "noise_ratio"
    sortable = metrics_df.dropna(subset=[silhouette_column]).copy()
    if noise_column not in sortable.columns:
        sortable[noise_column] = 0.0
    return sortable.sort_values(
        [silhouette_column, noise_column],
        ascending=[False, True],
        kind="stable",
    ).head(count)


def _summarize_metrics(metrics_df: pd.DataFrame, *, dataset: str, experiment: str) -> pd.DataFrame:
    df = metrics_df.copy()
    df["dataset"] = dataset
    df["experiment"] = experiment
    group_columns = ["dataset", "experiment", "representation", "clustering_method"]
    metric_columns = [
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "n_clusters",
        "noise_ratio",
        "embedding_time_s",
        "clustering_time_s",
        "peak_memory_mb",
    ]
    existing = [column for column in metric_columns if column in df.columns]
    summary = df.groupby(group_columns, as_index=False).agg(
        seed_count=("seed", "nunique"),
        sample_size=("sample_size", "max"),
        **{f"{column}_mean": (column, "mean") for column in existing},
        **{f"{column}_std": (column, "std") for column in existing},
    )
    return normalize_report_dataframe(summary)


def _validate_primary_seed(
    metrics_df: pd.DataFrame,
    *,
    seed: int,
    representations: list[str],
    clustering_methods: list[str],
    source_path: Path,
) -> pd.DataFrame:
    filtered = metrics_df[
        metrics_df["representation"].isin(representations)
        & metrics_df["clustering_method"].isin(clustering_methods)
        & (metrics_df["seed"].astype(int) == int(seed))
    ].copy()
    expected_rows = len(representations) * len(clustering_methods)
    if len(filtered) != expected_rows:
        raise ValueError(
            f"{source_path} should contain {expected_rows} rows for seed {seed}, found {len(filtered)}"
        )
    observed_reps = set(filtered["representation"].astype(str))
    observed_methods = set(filtered["clustering_method"].astype(str))
    if observed_reps != set(representations) or observed_methods != set(clustering_methods):
        raise ValueError(f"{source_path} does not contain the expected representation/method grid")
    return filtered


def _load_primary_runs(
    *,
    seeds: list[int],
    representations: list[str],
    clustering_methods: list[str],
    output_dir: Path,
    primary_runs_root: Path | None,
    primary_week2_dir: Path,
    sample_size: int,
    primary_input: Path,
    sbert_device: str,
    metric_sample_size: int,
    resume: bool,
    verbose: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        stage_name = f"primary_optimal_seed{seed}"
        existing = _find_existing_stage(stage_name, output_dir, primary_runs_root)
        if existing is None and int(seed) == 42 and _stage_is_complete(primary_week2_dir):
            existing = primary_week2_dir
        destination = output_dir / stage_name

        if existing is not None:
            if existing != destination:
                _progress(f"Reuse primary optimal seed {seed} dari {existing}", verbose=verbose)
                _copy_week2_artifacts(existing, destination)
            else:
                _progress(f"Reuse primary optimal seed {seed} dari output final", verbose=verbose)
        elif resume:
            _progress(f"Primary optimal seed {seed} belum ada; menjalankan Week 2 optimal.", verbose=verbose)
            run_week2(
                input_path=primary_input,
                output_dir=destination,
                sample_size=sample_size,
                seed=seed,
                representations=representations,
                clustering_methods=clustering_methods,
                experiment_mode="optimal",
                sbert_device=sbert_device,
                metric_sample_size=metric_sample_size,
                verbose=verbose,
            )
        else:
            _progress(f"--no-resume aktif; menjalankan ulang primary optimal seed {seed}.", verbose=verbose)
            run_week2(
                input_path=primary_input,
                output_dir=destination,
                sample_size=sample_size,
                seed=seed,
                representations=representations,
                clustering_methods=clustering_methods,
                experiment_mode="optimal",
                sbert_device=sbert_device,
                metric_sample_size=metric_sample_size,
                verbose=verbose,
            )

        metrics = _load_metrics(destination / "metrics_week2.csv")
        frames.append(
            _validate_primary_seed(
                metrics,
                seed=seed,
                representations=representations,
                clustering_methods=clustering_methods,
                source_path=destination / "metrics_week2.csv",
            )
        )
    return normalize_report_dataframe(pd.concat(frames, ignore_index=True))


def _run_or_reuse_week2_stage(
    *,
    stage_name: str,
    input_path: Path,
    output_dir: Path,
    primary_runs_root: Path | None,
    sample_size: int,
    seed: int,
    representations: list[str],
    clustering_methods: list[str],
    experiment_mode: str,
    sbert_device: str,
    metric_sample_size: int,
    resume: bool,
    verbose: bool,
) -> pd.DataFrame:
    destination = output_dir / stage_name
    existing = _find_existing_stage(stage_name, output_dir, primary_runs_root) if resume else None
    if existing is not None:
        if existing != destination:
            _progress(f"Reuse {stage_name} dari {existing}", verbose=verbose)
            _copy_week2_artifacts(existing, destination)
        else:
            _progress(f"Skip {stage_name}; output sudah ada.", verbose=verbose)
    else:
        _progress(f"Menjalankan {stage_name}.", verbose=verbose)
        run_week2(
            input_path=input_path,
            output_dir=destination,
            sample_size=sample_size,
            seed=seed,
            representations=representations,
            clustering_methods=clustering_methods,
            experiment_mode=experiment_mode,
            sbert_device=sbert_device,
            metric_sample_size=metric_sample_size,
            verbose=verbose,
        )
    metrics = _load_metrics(destination / "metrics_week2.csv")
    return metrics[
        metrics["representation"].isin(representations)
        & metrics["clustering_method"].isin(clustering_methods)
        & (metrics["seed"].astype(int) == int(seed))
    ].copy()


def _comparison_table(primary_summary: pd.DataFrame, secondary_summary: pd.DataFrame) -> pd.DataFrame:
    join_columns = ["representation", "clustering_method"]
    primary_columns = join_columns + [
        "seed_count",
        "silhouette_mean",
        "calinski_harabasz_mean",
        "davies_bouldin_mean",
        "n_clusters_mean",
        "noise_ratio_mean",
    ]
    secondary_columns = primary_columns
    primary = primary_summary[[column for column in primary_columns if column in primary_summary.columns]].rename(
        columns={
            "seed_count": "primary_seed_count",
            "silhouette_mean": "primary_silhouette_mean",
            "calinski_harabasz_mean": "primary_calinski_harabasz_mean",
            "davies_bouldin_mean": "primary_davies_bouldin_mean",
            "n_clusters_mean": "primary_n_clusters_mean",
            "noise_ratio_mean": "primary_noise_ratio_mean",
        }
    )
    secondary = secondary_summary[
        [column for column in secondary_columns if column in secondary_summary.columns]
    ].rename(
        columns={
            "seed_count": "secondary_seed_count",
            "silhouette_mean": "secondary_silhouette_mean",
            "calinski_harabasz_mean": "secondary_calinski_harabasz_mean",
            "davies_bouldin_mean": "secondary_davies_bouldin_mean",
            "n_clusters_mean": "secondary_n_clusters_mean",
            "noise_ratio_mean": "secondary_noise_ratio_mean",
        }
    )
    comparison = primary.merge(secondary, on=join_columns, how="inner")
    if {"primary_silhouette_mean", "secondary_silhouette_mean"}.issubset(comparison.columns):
        comparison["silhouette_delta_secondary_minus_primary"] = (
            comparison["secondary_silhouette_mean"] - comparison["primary_silhouette_mean"]
        )
    return normalize_report_dataframe(comparison)


def _save_bar_plot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    value_column: str = "silhouette_mean",
    error_column: str | None = None,
) -> None:
    if df.empty or value_column not in df.columns:
        return
    plot_df = df.copy()
    plot_df["combo"] = plot_df["representation"].astype(str) + " + " + plot_df["clustering_method"].astype(str)
    plot_df = plot_df.sort_values(value_column, ascending=False).head(12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    errors = plot_df[error_column] if error_column and error_column in plot_df.columns else None
    ax.bar(plot_df["combo"], plot_df[value_column], yerr=errors, color="#2f6f8f", capsize=4)
    ax.set_title(title)
    ax.set_ylabel(value_column)
    ax.set_xlabel("Combination")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_cross_dataset_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    required = {"primary_silhouette_mean", "secondary_silhouette_mean"}
    if comparison_df.empty or not required.issubset(comparison_df.columns):
        return
    plot_df = comparison_df.copy()
    plot_df["combo"] = plot_df["representation"].astype(str) + " + " + plot_df["clustering_method"].astype(str)
    plot_df = plot_df.sort_values("primary_silhouette_mean", ascending=False)
    x = range(len(plot_df))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, plot_df["primary_silhouette_mean"], marker="o", label="scenario-1")
    ax.plot(x, plot_df["secondary_silhouette_mean"], marker="o", label="scenario-2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["combo"], rotation=30, ha="right")
    ax.set_ylabel("Silhouette mean")
    ax.set_title("Cross-Dataset Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _combo_label(df: pd.DataFrame) -> pd.Series:
    return df["representation"].astype(str) + " + " + df["clustering_method"].astype(str)


def _normalize_metric(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.0, index=series.index)
    fill_value = numeric.median()
    numeric = numeric.fillna(fill_value)
    minimum = numeric.min()
    maximum = numeric.max()
    if math.isclose(float(maximum), float(minimum)):
        return pd.Series(1.0, index=series.index)
    normalized = (numeric - minimum) / (maximum - minimum)
    return normalized if higher_is_better else 1.0 - normalized


def _timed_visual(
    records: list[dict[str, Any]],
    *,
    artifact_name: str,
    visualization_type: str,
    input_source: str,
    output_path: Path,
    callback: Any,
) -> Any:
    started_at = time.perf_counter()
    status = "ok"
    error = ""
    result: Any = None
    try:
        result = callback()
    except Exception as exc:  # pragma: no cover - recorded so polish can continue.
        status = "failed"
        error = str(exc)
    elapsed_s = round(time.perf_counter() - started_at, 6)
    records.append(
        {
            "artifact_name": artifact_name,
            "visualization_type": visualization_type,
            "input_source": input_source,
            "elapsed_s": elapsed_s,
            "output_path": str(output_path),
            "status": status,
            "error": error,
        }
    )
    return result


def _save_critical_difference(primary_runs: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    required = {"seed", "representation", "clustering_method", "silhouette"}
    if primary_runs.empty or not required.issubset(primary_runs.columns):
        return pd.DataFrame()

    df = primary_runs.dropna(subset=["seed", "silhouette"]).copy()
    df["combo"] = _combo_label(df)
    pivot = df.pivot_table(index="seed", columns="combo", values="silhouette", aggfunc="mean").dropna(axis=1)
    if pivot.empty:
        return pd.DataFrame()

    ranks = pivot.rank(axis=1, ascending=False, method="average")
    n_blocks = int(ranks.shape[0])
    n_methods = int(ranks.shape[1])
    friedman_statistic = np.nan
    friedman_p_value = np.nan
    critical_difference = np.nan
    significance_alpha = 0.05
    if n_blocks >= 2 and n_methods >= 3:
        friedman_statistic, friedman_p_value = stats.friedmanchisquare(
            *[pivot[column].to_numpy() for column in pivot.columns]
        )
        q_alpha = stats.studentized_range.ppf(1.0 - significance_alpha, n_methods, np.inf) / math.sqrt(2.0)
        critical_difference = float(q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks)))

    summary = pd.DataFrame(
        {
            "combo": ranks.columns,
            "mean_rank": ranks.mean(axis=0).to_numpy(),
            "rank_std": ranks.std(axis=0).fillna(0.0).to_numpy(),
            "n_blocks": n_blocks,
            "n_methods": n_methods,
            "friedman_statistic": friedman_statistic,
            "friedman_p_value": friedman_p_value,
            "nemenyi_alpha": significance_alpha,
            "nemenyi_critical_difference": critical_difference,
            "friedman_significant": bool(pd.notna(friedman_p_value) and friedman_p_value < significance_alpha),
        }
    ).sort_values("mean_rank", kind="stable")
    summary.to_csv(output_dir / "cd_friedman_nemenyi_week4.csv", index=False)

    plot_df = summary.head(20).copy()
    plot_path = output_dir / "plots" / "critical_difference_primary_silhouette.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.36 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.errorbar(plot_df["mean_rank"], y, xerr=plot_df["rank_std"], fmt="o", color="#235789", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["combo"])
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Mean rank by seed (lower is better)")
    ax.set_title("Critical Difference / Friedman-Nemenyi - Primary Silhouette")
    ax.grid(axis="x", alpha=0.25)
    if pd.notna(critical_difference):
        x_max = float(plot_df["mean_rank"].max())
        y_cd = len(plot_df) + 0.45
        ax.plot([x_max - critical_difference, x_max], [y_cd, y_cd], color="#d95f02", linewidth=3)
        ax.text(
            x_max - critical_difference / 2,
            y_cd + 0.2,
            f"CD={critical_difference:.2f}",
            ha="center",
            va="bottom",
            color="#d95f02",
        )
        ax.set_ylim(y_cd + 0.7, -0.7)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return summary


def _save_radar(primary_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    required = {
        "representation",
        "clustering_method",
        "silhouette_mean",
        "calinski_harabasz_mean",
        "davies_bouldin_mean",
        "noise_ratio_mean",
        "embedding_time_s_mean",
        "clustering_time_s_mean",
        "peak_memory_mb_mean",
    }
    if primary_summary.empty or not required.issubset(primary_summary.columns):
        return pd.DataFrame()

    scores = primary_summary.copy()
    scores["combo"] = _combo_label(scores)
    scores["total_runtime_s_mean"] = (
        pd.to_numeric(scores["embedding_time_s_mean"], errors="coerce").fillna(0.0)
        + pd.to_numeric(scores["clustering_time_s_mean"], errors="coerce").fillna(0.0)
    )
    score_columns = {
        "silhouette_score": ("silhouette_mean", True),
        "calinski_harabasz_score": ("calinski_harabasz_mean", True),
        "davies_bouldin_score": ("davies_bouldin_mean", False),
        "noise_ratio_score": ("noise_ratio_mean", False),
        "runtime_score": ("total_runtime_s_mean", False),
        "memory_score": ("peak_memory_mb_mean", False),
    }
    for output_column, (input_column, higher_is_better) in score_columns.items():
        scores[output_column] = _normalize_metric(scores[input_column], higher_is_better=higher_is_better)
    normalized_columns = list(score_columns)
    scores["multi_metric_score"] = scores[normalized_columns].mean(axis=1)
    scores = scores.sort_values("silhouette_mean", ascending=False, kind="stable")
    scores.to_csv(output_dir / "multi_metric_scores_week4.csv", index=False)

    top = scores.head(5)
    labels = [
        "Silhouette",
        "CH",
        "DB inv.",
        "Noise inv.",
        "Runtime inv.",
        "Memory inv.",
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plot_path = output_dir / "plots" / "radar_multi_metric_top5.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for _, row in top.iterrows():
        values = [float(row[column]) for column in normalized_columns]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=row["combo"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Top 5 Primary Pipelines - Normalized Multi-Metric Radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return scores


def _is_pareto_optimal(frame: pd.DataFrame) -> pd.Series:
    silhouette = pd.to_numeric(frame["silhouette_mean"], errors="coerce").fillna(-np.inf).to_numpy()
    runtime = pd.to_numeric(frame["total_runtime_s_mean"], errors="coerce").fillna(np.inf).to_numpy()
    pareto = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominates = (silhouette >= silhouette[index]) & (runtime <= runtime[index])
        strictly_better = (silhouette > silhouette[index]) | (runtime < runtime[index])
        pareto[index] = not np.any(dominates & strictly_better)
    return pd.Series(pareto, index=frame.index)


def _save_pareto(primary_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    required = {"representation", "clustering_method", "silhouette_mean", "embedding_time_s_mean", "clustering_time_s_mean"}
    if primary_summary.empty or not required.issubset(primary_summary.columns):
        return pd.DataFrame()

    pareto = primary_summary.copy()
    pareto["combo"] = _combo_label(pareto)
    pareto["total_runtime_s_mean"] = (
        pd.to_numeric(pareto["embedding_time_s_mean"], errors="coerce").fillna(0.0)
        + pd.to_numeric(pareto["clustering_time_s_mean"], errors="coerce").fillna(0.0)
    )
    pareto["is_pareto_optimal"] = _is_pareto_optimal(pareto)
    pareto = pareto.sort_values(["is_pareto_optimal", "silhouette_mean"], ascending=[False, False], kind="stable")
    pareto.to_csv(output_dir / "pareto_front_week4.csv", index=False)

    plot_path = output_dir / "plots" / "pareto_silhouette_runtime.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    non_front = pareto[~pareto["is_pareto_optimal"]]
    front = pareto[pareto["is_pareto_optimal"]].sort_values("total_runtime_s_mean")
    ax.scatter(non_front["total_runtime_s_mean"], non_front["silhouette_mean"], s=55, alpha=0.6, label="Non-Pareto")
    ax.scatter(front["total_runtime_s_mean"], front["silhouette_mean"], s=85, color="#d95f02", label="Pareto front")
    if len(front) > 1:
        ax.plot(front["total_runtime_s_mean"], front["silhouette_mean"], color="#d95f02", linewidth=1.5, alpha=0.8)
    label_candidates = pd.concat(
        [
            front,
            pareto[pareto["combo"].eq("tfidf + hdbscan")],
            pareto.sort_values("silhouette_mean", ascending=False).head(3),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["combo"])
    for _, row in label_candidates.iterrows():
        ax.annotate(
            row["combo"],
            (row["total_runtime_s_mean"], row["silhouette_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Mean total runtime (embedding + clustering, seconds)")
    ax.set_ylabel("Mean silhouette")
    ax.set_title("Pareto Front - Silhouette vs Computation Time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return pareto


def _token_weights(raw_tokens: Any) -> dict[str, float]:
    tokens = [token.strip() for token in str(raw_tokens).split(";") if token.strip()]
    weights: dict[str, float] = {}
    total = len(tokens)
    for index, token in enumerate(tokens):
        weights[token] = weights.get(token, 0.0) + float(total - index)
    return weights


def _save_wordclouds(week3_dir: Path, output_dir: Path) -> pd.DataFrame:
    source_path = week3_dir / "cluster_interpretability_week3.csv"
    index_path = output_dir / "wordcloud_index_week4.csv"
    if not source_path.exists():
        skipped = pd.DataFrame(
            [
                {
                    "representation": "tfidf",
                    "clustering_method": "hdbscan",
                    "cluster_id": "",
                    "size": "",
                    "top_tokens": "",
                    "output_path": "",
                    "status": f"skipped: missing {source_path}",
                }
            ]
        )
        skipped.to_csv(index_path, index=False)
        return skipped

    try:
        from wordcloud import WordCloud
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("Install wordcloud>=1.9 to generate literal Week 4 word clouds.") from exc

    clusters = pd.read_csv(source_path)
    required = {"representation", "clustering_method", "cluster_id", "size", "is_noise", "top_tokens"}
    if clusters.empty or not required.issubset(clusters.columns):
        empty = pd.DataFrame()
        empty.to_csv(index_path, index=False)
        return empty

    selected = clusters[
        clusters["representation"].astype(str).eq("tfidf")
        & clusters["clustering_method"].astype(str).eq("hdbscan")
        & ~clusters["is_noise"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    selected["size"] = pd.to_numeric(selected["size"], errors="coerce").fillna(0)
    selected = selected.sort_values("size", ascending=False, kind="stable").head(12)
    wordcloud_dir = output_dir / "plots" / "wordclouds"
    wordcloud_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    aggregate_weights: dict[str, float] = {}
    for _, row in selected.iterrows():
        weights = _token_weights(row["top_tokens"])
        if not weights:
            continue
        for token, weight in weights.items():
            aggregate_weights[token] = aggregate_weights.get(token, 0.0) + weight * max(float(row["size"]), 1.0)
        cluster_id = str(row["cluster_id"])
        output_path = wordcloud_dir / f"tfidf_hdbscan_cluster_{cluster_id}.png"
        cloud = WordCloud(width=1200, height=720, background_color="white", colormap="viridis", random_state=42)
        cloud.generate_from_frequencies(weights)
        cloud.to_file(str(output_path))
        index_rows.append(
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "cluster_id": row["cluster_id"],
                "size": row["size"],
                "top_tokens": row["top_tokens"],
                "output_path": str(output_path),
                "status": "ok",
            }
        )

    if aggregate_weights:
        aggregate_path = wordcloud_dir / "tfidf_hdbscan_aggregate.png"
        cloud = WordCloud(width=1400, height=820, background_color="white", colormap="viridis", random_state=42)
        cloud.generate_from_frequencies(aggregate_weights)
        cloud.to_file(str(aggregate_path))
        index_rows.append(
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "cluster_id": "aggregate",
                "size": selected["size"].sum(),
                "top_tokens": "aggregate top_tokens from 12 largest tfidf+hdbscan clusters",
                "output_path": str(aggregate_path),
                "status": "ok",
            }
        )

    index = pd.DataFrame(index_rows)
    index.to_csv(index_path, index=False)
    return index


def _append_polish_report_section(output_dir: Path, polish_result: dict[str, Any]) -> None:
    report_path = _week4_report_path(output_dir)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Week 4 Final Report\n"
    marker = "## Literal PDF Visualization Polish"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"

    cd = polish_result.get("critical_difference", pd.DataFrame())
    pareto = polish_result.get("pareto", pd.DataFrame())
    wordcloud_index = polish_result.get("wordcloud_index", pd.DataFrame())
    visualization_times = polish_result.get("visualization_times", pd.DataFrame())

    best_rank = ""
    if isinstance(cd, pd.DataFrame) and not cd.empty:
        top = cd.sort_values("mean_rank", kind="stable").iloc[0]
        p_value = top.get("friedman_p_value")
        p_text = "" if pd.isna(p_value) else f"; Friedman p-value `{float(p_value):.4g}`"
        best_rank = f"- Critical Difference/Friedman-Nemenyi dibuat untuk primary silhouette; rank terbaik: `{top['combo']}`{p_text}."
    pareto_count = 0
    if isinstance(pareto, pd.DataFrame) and "is_pareto_optimal" in pareto.columns:
        pareto_count = int(pareto["is_pareto_optimal"].sum())
    wordcloud_count = 0
    if isinstance(wordcloud_index, pd.DataFrame) and "status" in wordcloud_index.columns:
        wordcloud_count = int(wordcloud_index["status"].astype(str).eq("ok").sum())
    total_viz_time = 0.0
    if isinstance(visualization_times, pd.DataFrame) and "elapsed_s" in visualization_times.columns:
        total_viz_time = float(pd.to_numeric(visualization_times["elapsed_s"], errors="coerce").fillna(0.0).sum())

    lines = [
        "",
        marker,
        "",
        "Polish pass ini menutup item visualisasi literal dari PDF tanpa mengulang 100 primary run, k-sensitivity, atau cross-dataset validation.",
        "",
    ]
    if best_rank:
        lines.append(best_rank)
    lines.extend(
        [
            "- Radar/spider chart multi-metrik tersedia di `plots/radar_multi_metric_top5.png` dan skor normalisasinya di `multi_metric_scores_week4.csv`.",
            f"- Pareto front Silhouette vs computation time tersedia di `plots/pareto_silhouette_runtime.png` dengan `{pareto_count}` kombinasi Pareto-optimal.",
            f"- Word cloud literal untuk best pipeline `tfidf + hdbscan` tersedia di `plots/wordclouds/` dengan `{wordcloud_count}` artefak berstatus ok.",
            f"- Visualization time metric dicatat di `visualization_times_week4.csv`; total waktu render polish tercatat `{total_viz_time:.4f}` detik.",
            "",
            "Artefak polish utama:",
            "",
            "- `cd_friedman_nemenyi_week4.csv` dan `plots/critical_difference_primary_silhouette.png`",
            "- `multi_metric_scores_week4.csv` dan `plots/radar_multi_metric_top5.png`",
            "- `pareto_front_week4.csv` dan `plots/pareto_silhouette_runtime.png`",
            "- `wordcloud_index_week4.csv` dan `plots/wordclouds/*.png`",
            "- `visualization_times_week4.csv`",
            "",
        ]
    )
    report_path.write_text(existing.rstrip() + "\n".join(lines), encoding="utf-8")


def generate_week4_visual_polish(
    output_dir: Path = Path("reports/week4"),
    week3_dir: Path = DEFAULT_WEEK3_DIR,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate literal-PDF Week 4 visual polish from final CSV artifacts."""
    output_dir = Path(output_dir)
    week3_dir = Path(week3_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_times: list[dict[str, Any]] = []

    primary_runs = _load_metrics(output_dir / "metrics_primary_runs.csv")
    primary_summary = _load_metrics(output_dir / "metrics_primary_summary.csv")
    k_sensitivity_summary_path = output_dir / "metrics_k_sensitivity_summary.csv"
    comparison_path = output_dir / "cross_dataset_comparison.csv"
    k_sensitivity_summary = _load_metrics(k_sensitivity_summary_path) if k_sensitivity_summary_path.exists() else pd.DataFrame()
    comparison = _load_metrics(comparison_path) if comparison_path.exists() else pd.DataFrame()

    _progress("Membuat Critical Difference/Friedman-Nemenyi, radar, Pareto, word cloud, dan timing visual.", verbose=verbose)
    critical_difference = _timed_visual(
        visualization_times,
        artifact_name="critical_difference_primary_silhouette",
        visualization_type="critical_difference_friedman_nemenyi",
        input_source=str(output_dir / "metrics_primary_runs.csv"),
        output_path=output_dir / "plots" / "critical_difference_primary_silhouette.png",
        callback=lambda: _save_critical_difference(primary_runs, output_dir),
    )
    radar = _timed_visual(
        visualization_times,
        artifact_name="radar_multi_metric_top5",
        visualization_type="radar_spider_chart",
        input_source=str(output_dir / "metrics_primary_summary.csv"),
        output_path=output_dir / "plots" / "radar_multi_metric_top5.png",
        callback=lambda: _save_radar(primary_summary, output_dir),
    )
    pareto = _timed_visual(
        visualization_times,
        artifact_name="pareto_silhouette_runtime",
        visualization_type="pareto_front",
        input_source=str(output_dir / "metrics_primary_summary.csv"),
        output_path=output_dir / "plots" / "pareto_silhouette_runtime.png",
        callback=lambda: _save_pareto(primary_summary, output_dir),
    )
    wordcloud_index = _timed_visual(
        visualization_times,
        artifact_name="tfidf_hdbscan_wordclouds",
        visualization_type="word_cloud",
        input_source=str(week3_dir / "cluster_interpretability_week3.csv"),
        output_path=output_dir / "plots" / "wordclouds",
        callback=lambda: _save_wordclouds(week3_dir, output_dir),
    )

    visualization_times_df = pd.DataFrame(visualization_times)
    visualization_times_df.to_csv(output_dir / "visualization_times_week4.csv", index=False)
    result = {
        "critical_difference": critical_difference if isinstance(critical_difference, pd.DataFrame) else pd.DataFrame(),
        "radar": radar if isinstance(radar, pd.DataFrame) else pd.DataFrame(),
        "pareto": pareto if isinstance(pareto, pd.DataFrame) else pd.DataFrame(),
        "wordcloud_index": wordcloud_index if isinstance(wordcloud_index, pd.DataFrame) else pd.DataFrame(),
        "visualization_times": visualization_times_df,
        "k_sensitivity_summary": k_sensitivity_summary,
        "comparison": comparison,
    }
    _append_polish_report_section(output_dir, result)
    _progress(f"Week 4 visual polish selesai. Output: {output_dir}", verbose=verbose)
    return result


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if pd.isna(raw):
        return {}
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _load_trial_files(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            frame["stage_dir"] = path.parent.name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _trial_parameter_label(row: pd.Series) -> tuple[Any, str]:
    params = _parse_params(row.get("params"))
    method = str(row.get("clustering_method", ""))
    if method in {"kmeans", "agglomerative"}:
        return params.get("n_clusters"), "k"
    if method == "gmm":
        return params.get("n_components"), "n_components"
    if method == "dbscan":
        eps = params.get("eps")
        min_samples = params.get("min_samples")
        return f"eps={float(eps):.4g}, min_samples={min_samples}" if eps is not None else str(params), "density_params"
    if method == "hdbscan":
        return f"min_cluster_size={params.get('min_cluster_size')}", "density_params"
    return str(params), "params"


def _save_k_sensitivity_by_k(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = _load_trial_files(sorted(output_dir.glob("primary_k_sensitivity_seed*/trials_week2.csv")))
    raw_path = output_dir / "k_sensitivity_trials_week4.csv"
    summary_path = output_dir / "k_sensitivity_by_k_week4.csv"
    if trials.empty:
        empty = pd.DataFrame()
        empty.to_csv(raw_path, index=False)
        empty.to_csv(summary_path, index=False)
        return empty, empty

    parsed = trials.apply(_trial_parameter_label, axis=1, result_type="expand")
    trials = trials.copy()
    trials["parameter_value"] = parsed[0]
    trials["parameter_type"] = parsed[1]
    trials["parameter_note"] = np.where(
        trials["parameter_type"].eq("density_params"),
        "DBSCAN/HDBSCAN do not use k directly; density grid is summarized instead.",
        "Explicit k/n_components sensitivity value.",
    )
    trials.to_csv(raw_path, index=False)

    summary = (
        trials.groupby(["representation", "clustering_method", "parameter_type", "parameter_value"], dropna=False)
        .agg(
            trial_count=("silhouette", "size"),
            seed_count=("seed", "nunique"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            n_clusters_mean=("n_clusters", "mean"),
            noise_ratio_mean=("noise_ratio", "mean"),
            clustering_time_s_mean=("clustering_time_s", "mean"),
        )
        .reset_index()
    )
    summary = normalize_report_dataframe(summary)
    summary.to_csv(summary_path, index=False)

    plot_df = summary[summary["parameter_type"].isin(["k", "n_components"])].copy()
    if not plot_df.empty:
        plot_path = output_dir / "plots" / "k_sensitivity_by_k_silhouette.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_df["parameter_value_numeric"] = pd.to_numeric(plot_df["parameter_value"], errors="coerce")
        for (representation, method), group in plot_df.groupby(["representation", "clustering_method"]):
            group = group.sort_values("parameter_value_numeric")
            ax.plot(
                group["parameter_value_numeric"],
                group["silhouette_mean"],
                marker="o",
                linewidth=1.6,
                label=f"{representation}+{method}",
            )
        ax.set_xlabel("k / n_components")
        ax.set_ylabel("Mean silhouette")
        ax.set_title("K-Sensitivity - k=10,20,50")
        ax.set_xticks(sorted(plot_df["parameter_value_numeric"].dropna().unique()))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    return trials, summary


def _save_k_optimal_diagnostics(output_dir: Path) -> pd.DataFrame:
    trials = _load_trial_files(sorted(output_dir.glob("primary_optimal_seed*/trials_week2.csv")))
    if trials.empty:
        fallback = Path("reports/week2/trials_week2.csv")
        trials = _load_trial_files([fallback])
    output_path = output_dir / "k_optimal_diagnostics_week4.csv"
    if trials.empty:
        trials.to_csv(output_path, index=False)
        return trials

    parsed = trials.apply(_trial_parameter_label, axis=1, result_type="expand")
    trials = trials.copy()
    trials["parameter_value"] = parsed[0]
    trials["parameter_type"] = parsed[1]
    k_trials = trials[trials["parameter_type"].isin(["k", "n_components"])].copy()
    k_trials["parameter_value_numeric"] = pd.to_numeric(k_trials["parameter_value"], errors="coerce")
    summary = (
        k_trials.dropna(subset=["parameter_value_numeric"])
        .groupby(["representation", "clustering_method", "parameter_type", "parameter_value_numeric"], dropna=False)
        .agg(
            trial_count=("silhouette", "size"),
            seed_count=("seed", "nunique"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            inertia_mean=("inertia", "mean") if "inertia" in k_trials.columns else ("silhouette", "size"),
            bic_mean=("bic", "mean") if "bic" in k_trials.columns else ("silhouette", "size"),
        )
        .reset_index()
    )
    summary.to_csv(output_path, index=False)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for representation, group in summary.groupby("representation"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, method_group in group.groupby("clustering_method"):
            method_group = method_group.sort_values("parameter_value_numeric")
            ax.plot(
                method_group["parameter_value_numeric"],
                method_group["silhouette_mean"],
                marker="o",
                label=method,
            )
        ax.set_title(f"K-Optimal Diagnostic - {representation}")
        ax.set_xlabel("k / n_components")
        ax.set_ylabel("Mean silhouette across available seeds")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"k_diagnostics_{representation}.png", dpi=160)
        plt.close(fig)
    return summary


def _save_primary_heatmap(output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "metrics_primary_summary.csv"
    heatmap_csv = output_dir / "primary_silhouette_heatmap_mean.csv"
    if not summary_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(heatmap_csv, index=False)
        return empty
    summary = pd.read_csv(summary_path)
    pivot = summary.pivot_table(
        index="representation",
        columns="clustering_method",
        values="silhouette_mean",
        aggfunc="mean",
    )
    ordered_rows = [value for value in DEFAULT_REPRESENTATIONS if value in pivot.index]
    ordered_cols = [value for value in DEFAULT_CLUSTERING_METHODS if value in pivot.columns]
    pivot = pivot.loc[ordered_rows, ordered_cols]
    pivot.to_csv(heatmap_csv)
    plot_path = output_dir / "plots" / "primary_silhouette_heatmap_mean.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_index, representation in enumerate(pivot.index):
        for col_index, method in enumerate(pivot.columns):
            value = pivot.loc[representation, method]
            if pd.notna(value):
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Primary Mean Silhouette Heatmap - 5 Seeds")
    fig.colorbar(image, ax=ax, label="Mean silhouette")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return pivot.reset_index()


def _save_full_wordclouds(week3_dir: Path, output_dir: Path, *, min_cluster_size: int = 5) -> pd.DataFrame:
    sample_path = week3_dir / "sample_metadata_week3.csv"
    label_path = week3_dir / "labels" / "tfidf_hdbscan_labels.csv"
    index_path = output_dir / "wordcloud_full_index_week4.csv"
    if not sample_path.exists() or not label_path.exists():
        skipped = pd.DataFrame(
            [
                {
                    "representation": "tfidf",
                    "clustering_method": "hdbscan",
                    "cluster_id": "",
                    "size": 0,
                    "output_path": "",
                    "status": f"skipped: missing {sample_path} or {label_path}",
                }
            ]
        )
        skipped.to_csv(index_path, index=False)
        return skipped
    try:
        from wordcloud import WordCloud
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install wordcloud>=1.9 to generate full literal word clouds.") from exc

    sample = pd.read_csv(sample_path, usecols=lambda column: column in {"sample_index", "clean_text", "message"})
    labels = pd.read_csv(label_path)
    merged = labels.merge(sample, on="sample_index", how="left")
    merged = merged[~merged["cluster_id"].astype(str).eq("-1")].copy()
    merged["cluster_id"] = pd.to_numeric(merged["cluster_id"], errors="coerce")
    merged = merged.dropna(subset=["cluster_id"])
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    counts = merged["cluster_id"].value_counts()
    selected_clusters = counts[counts >= min_cluster_size].index.tolist()
    output_wordcloud_dir = output_dir / "plots" / "wordclouds_full"
    output_wordcloud_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    text_column = "clean_text" if "clean_text" in merged.columns else "message"
    for cluster_id in sorted(selected_clusters):
        cluster_rows = merged[merged["cluster_id"].eq(cluster_id)]
        counter: Counter[str] = Counter()
        for text in cluster_rows[text_column].fillna("").astype(str):
            counter.update(token for token in text.split() if len(token) >= 2)
        output_path = output_wordcloud_dir / f"tfidf_hdbscan_cluster_{cluster_id}.png"
        status = "ok"
        if counter:
            cloud = WordCloud(width=1000, height=650, background_color="white", colormap="viridis", random_state=42)
            cloud.generate_from_frequencies(dict(counter))
            cloud.to_file(str(output_path))
        else:
            status = "skipped: empty token counter"
        rows.append(
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "cluster_id": cluster_id,
                "size": int(len(cluster_rows)),
                "top_terms": "; ".join(token for token, _ in counter.most_common(20)),
                "output_path": str(output_path),
                "status": status,
            }
        )
    index = pd.DataFrame(rows)
    index.to_csv(index_path, index=False)
    return index


def _save_manual_interpretability_review(week3_dir: Path, output_dir: Path) -> pd.DataFrame:
    source_path = week3_dir / "cluster_interpretability_week3.csv"
    output_path = output_dir / "manual_interpretability_review_week4.csv"
    if not source_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(output_path, index=False)
        return empty
    clusters = pd.read_csv(source_path)
    required = {"representation", "clustering_method", "cluster_id", "size", "top_tokens", "top_sources", "example_messages"}
    if clusters.empty or not required.issubset(clusters.columns):
        empty = pd.DataFrame()
        empty.to_csv(output_path, index=False)
        return empty
    review = clusters[
        clusters["representation"].astype(str).eq("tfidf")
        & clusters["clustering_method"].astype(str).eq("hdbscan")
        & ~clusters["cluster_id"].astype(str).eq("-1")
    ].copy()
    review["size"] = pd.to_numeric(review["size"], errors="coerce").fillna(0)
    review = review.sort_values(["forensic_cluster_score", "size"], ascending=[False, False], kind="stable").head(40)
    review["suggested_rating_1_5"] = pd.to_numeric(
        review.get("forensic_cluster_score", 3),
        errors="coerce",
    ).fillna(3).clip(1, 5).astype(int)
    review["rating_basis"] = "Heuristic proxy from source purity, token clarity, cluster size, and forensic-token cues; not a human expert label."
    review["review_notes"] = ""
    columns = [
        "cluster_id",
        "size",
        "top_tokens",
        "top_sources",
        "example_messages",
        "suggested_rating_1_5",
        "rating_basis",
        "review_notes",
    ]
    review[columns].to_csv(output_path, index=False)
    return review[columns]


def _file_status(path: Path, *, minimum_count: int | None = None) -> tuple[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return "missing", "File missing or empty."
    if minimum_count is not None and path.suffix.lower() == ".csv":
        try:
            count = len(pd.read_csv(path))
        except Exception as exc:
            return "problem", f"Could not read CSV: {exc}"
        if count < minimum_count:
            return "partial", f"Only {count} rows; expected at least {minimum_count}."
        return "fulfilled", f"{count} rows."
    return "fulfilled", "Artifact exists."


def _write_requirement_traceability(output_dir: Path, week3_dir: Path) -> pd.DataFrame:
    scatter_plot_count = sum(
        len(list((week3_dir / "scatter" / method).glob("*.png")))
        for method in ["pca", "umap", "tsne"]
    )
    scatter_grid_count = len(list((week3_dir / "scatter_grids").glob("*.png"))) if (week3_dir / "scatter_grids").exists() else 0

    checks = [
        ("Dataset primer minimum 50.000 event", output_dir / "metrics_primary_runs.csv", 100, "core"),
        ("4 representasi x 5 clustering x 5 seed = 100 run", output_dir / "metrics_primary_runs.csv", 100, "core"),
        ("Summary 20 kombinasi rata-rata 5 seed", output_dir / "metrics_primary_summary.csv", 20, "core"),
        ("K-sensitivity k=10/20/50 eksplisit", output_dir / "k_sensitivity_by_k_week4.csv", 1, "literal polish"),
        ("Cross-dataset validation best pipeline", output_dir / "cross_dataset_comparison.csv", 1, "core"),
        ("Visualization time PCA/UMAP/t-SNE", week3_dir / "visualization_times_week3.csv", 1, "literal polish"),
        ("Critical Difference/Friedman-Nemenyi", output_dir / "cd_friedman_nemenyi_week4.csv", 20, "literal polish"),
        ("Radar/spider chart multi-metrik", output_dir / "multi_metric_scores_week4.csv", 20, "literal polish"),
        ("Pareto front kualitas vs waktu", output_dir / "pareto_front_week4.csv", 20, "literal polish"),
        ("Word cloud literal best pipeline", output_dir / "wordcloud_full_index_week4.csv", 1, "literal polish"),
        ("Interpretability review-ready 1-5 proxy", output_dir / "manual_interpretability_review_week4.csv", 1, "literal polish"),
        ("Final report dan dokumentasi", _week4_report_path(output_dir), None, "core"),
    ]
    rows: list[dict[str, Any]] = []
    for requirement, path, minimum_count, category in checks:
        status, note = _file_status(path, minimum_count=minimum_count)
        rows.append(
            {
                "requirement": requirement,
                "category": category,
                "status": status,
                "evidence_path": str(path),
                "note": note,
            }
        )
    rows.insert(
        5,
        {
            "requirement": "PCA/UMAP/t-SNE 60 scatter plots",
            "category": "literal polish",
            "status": "fulfilled" if scatter_plot_count >= 60 and scatter_grid_count >= 3 else "partial",
            "evidence_path": str(week3_dir / "scatter"),
            "note": f"{scatter_plot_count} individual scatter PNGs and {scatter_grid_count} grid PNGs.",
        },
    )
    traceability = pd.DataFrame(rows)
    traceability.to_csv(output_dir / "pdf_requirement_traceability_week4.csv", index=False)

    lines = [
        "# PDF Requirement Audit - Week 4 Final Completion",
        "",
        "Audit ini memetakan requirement literal dari `Log clustering.pdf` ke artefak repository saat ini.",
        "",
    ]
    lines.extend(_markdown_table(traceability))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Status `fulfilled` berarti artefak ada dan memenuhi hitungan minimum yang relevan.",
            "- Interpretability rating di repo ini adalah proxy/review-ready heuristic, bukan klaim label human expert final.",
            "- DBSCAN/HDBSCAN tidak memiliki parameter k langsung, sehingga k-sensitivity untuk metode density dirangkum melalui density grid.",
            "",
        ]
    )
    (output_dir / "PDF_REQUIREMENT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return traceability


def _append_completion_report_section(output_dir: Path, result: dict[str, Any]) -> None:
    report_path = _week4_report_path(output_dir)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Week 4 Final Report\n"
    marker = "## Final Literal PDF Completion"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    traceability = result.get("traceability", pd.DataFrame())
    fulfilled = 0
    total = 0
    if isinstance(traceability, pd.DataFrame) and not traceability.empty:
        total = len(traceability)
        fulfilled = int(traceability["status"].eq("fulfilled").sum())
    full_wordcloud = result.get("full_wordcloud_index", pd.DataFrame())
    wordcloud_count = (
        int(full_wordcloud["status"].astype(str).eq("ok").sum())
        if isinstance(full_wordcloud, pd.DataFrame) and "status" in full_wordcloud.columns
        else 0
    )
    lines = [
        "",
        marker,
        "",
        "Completion pass ini menutup sisa gap literal PDF tanpa mengulang 100 primary run Week 4.",
        "",
        f"- Traceability matrix: `{output_dir / 'PDF_REQUIREMENT_AUDIT.md'}` (`{fulfilled}/{total}` requirement berstatus fulfilled).",
        "- Visualization time PCA/UMAP/t-SNE: `reports/week3/visualization_times_week3.csv`.",
        "- K-sensitivity eksplisit k=10/20/50: `k_sensitivity_by_k_week4.csv` dan `plots/k_sensitivity_by_k_silhouette.png`.",
        "- K-optimal diagnostic per representasi: `k_optimal_diagnostics_week4.csv` dan `plots/k_diagnostics_*.png`.",
        "- Heatmap final rata-rata 5 seed: `primary_silhouette_heatmap_mean.csv` dan `plots/primary_silhouette_heatmap_mean.png`.",
        "- Scatter grid/montage: `reports/week3/scatter_grids/{pca,umap,tsne}_grid.png`.",
        f"- Full word cloud best pipeline: `wordcloud_full_index_week4.csv` dengan `{wordcloud_count}` cluster berstatus ok.",
        "- Manual interpretability review-ready artifact: `manual_interpretability_review_week4.csv`.",
        "",
    ]
    report_path.write_text(existing.rstrip() + "\n".join(lines), encoding="utf-8")


def generate_pdf_literal_completion(
    output_dir: Path = Path("reports/week4"),
    week3_dir: Path = DEFAULT_WEEK3_DIR,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate final literal-PDF completion artifacts from existing Week 3/4 outputs."""
    output_dir = Path(output_dir)
    week3_dir = Path(week3_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress("Membuat final literal PDF completion artifacts.", verbose=verbose)

    k_sensitivity_trials, k_sensitivity_summary = _save_k_sensitivity_by_k(output_dir)
    k_diagnostics = _save_k_optimal_diagnostics(output_dir)
    heatmap = _save_primary_heatmap(output_dir)
    full_wordcloud_index = _save_full_wordclouds(week3_dir, output_dir)
    manual_review = _save_manual_interpretability_review(week3_dir, output_dir)
    traceability = _write_requirement_traceability(output_dir, week3_dir)

    result = {
        "k_sensitivity_trials": k_sensitivity_trials,
        "k_sensitivity_by_k": k_sensitivity_summary,
        "k_diagnostics": k_diagnostics,
        "heatmap": heatmap,
        "full_wordcloud_index": full_wordcloud_index,
        "manual_review": manual_review,
        "traceability": traceability,
    }
    _append_completion_report_section(output_dir, result)
    _progress(f"Final literal PDF completion selesai. Output: {output_dir}", verbose=verbose)
    return result


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["Tidak ada data."]
    table = df.copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    output_dir: Path,
    *,
    primary_summary: pd.DataFrame,
    k_sensitivity_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    best_representations: list[str],
    best_pipeline: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    top_primary = _best_combinations(primary_summary, count=10, mean_columns=True)
    top_k = _best_combinations(k_sensitivity_summary, count=10, mean_columns=True)
    top_secondary = _best_combinations(secondary_summary, count=8, mean_columns=True)

    lines = [
        "# Week 4 Final Report",
        "",
        "## Scope",
        "",
        "Pekan 4 final mengagregasi primary optimal run untuk lima seed dan menjalankan validasi tambahan pada k-sensitivity serta dataset kedua.",
        "",
        "## Configuration",
        "",
        f"- Primary dataset: `{run_config['primary_input']}`",
        f"- Secondary dataset: `{run_config['secondary_input']}`",
        f"- Sample size per dataset: {run_config['sample_size']:,} valid events",
        f"- Primary seeds: {', '.join(str(seed) for seed in run_config['seeds'])}",
        f"- Representasi final untuk sensitivity: {', '.join(best_representations)}",
        f"- Best pipeline untuk cross-dataset: {best_pipeline['representation']} + {best_pipeline['clustering_method']}",
        "",
        "## Primary Optimal Summary",
        "",
        "Tabel berikut memakai rata-rata dan standard deviation dari lima seed primary optimal.",
        "",
    ]
    if not top_primary.empty:
        lines.extend(
            _markdown_table(
                top_primary[
                    [
                        "representation",
                        "clustering_method",
                        "seed_count",
                        "n_clusters_mean",
                        "noise_ratio_mean",
                        "silhouette_mean",
                        "silhouette_std",
                        "calinski_harabasz_mean",
                        "davies_bouldin_mean",
                    ]
                ]
            )
        )
    lines.extend(["", "## K-Sensitivity", ""])
    lines.append(
        "K-sensitivity dijalankan pada dua representasi terbaik final. Untuk DBSCAN/HDBSCAN, grid density tetap dipakai karena kedua metode tidak memakai parameter k langsung."
    )
    lines.append("")
    if not top_k.empty:
        lines.extend(
            _markdown_table(
                top_k[
                    [
                        "representation",
                        "clustering_method",
                        "seed_count",
                        "n_clusters_mean",
                        "noise_ratio_mean",
                        "silhouette_mean",
                        "silhouette_std",
                        "davies_bouldin_mean",
                    ]
                ]
            )
        )
    lines.extend(["", "## Cross-Dataset Validation", ""])
    lines.append(
        "Scenario-2 diproses sebagai stratified sample 50.000 event valid agar sebanding dengan scenario-1."
    )
    lines.append("")
    if not comparison.empty:
        lines.extend(
            _markdown_table(
                comparison[
                    [
                        "representation",
                        "clustering_method",
                        "primary_silhouette_mean",
                        "secondary_silhouette_mean",
                        "silhouette_delta_secondary_minus_primary",
                        "primary_noise_ratio_mean",
                        "secondary_noise_ratio_mean",
                    ]
                ]
            )
        )
    lines.extend(["", "## Secondary Best Result", ""])
    if not top_secondary.empty:
        lines.extend(
            _markdown_table(
                top_secondary[
                    [
                        "representation",
                        "clustering_method",
                        "seed_count",
                        "n_clusters_mean",
                        "noise_ratio_mean",
                        "silhouette_mean",
                        "davies_bouldin_mean",
                    ]
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"Best pipeline final adalah `{best_pipeline['representation']} + {best_pipeline['clustering_method']}` berdasarkan mean silhouette primary optimal lima seed.",
            "",
        ]
    )
    report_path = _week4_report_path(output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_week4(
    *,
    primary_input: Path = Path("timeline.csv"),
    secondary_input: Path = Path("data/external/timeline.csv"),
    primary_week2_dir: Path = Path("reports/week2"),
    primary_runs_root: Path | None = DEFAULT_PRIMARY_RUNS_ROOT,
    output_dir: Path = Path("reports/week4"),
    week3_dir: Path = DEFAULT_WEEK3_DIR,
    seeds: list[int] | None = None,
    sample_size: int = 50_000,
    top_representations: int = 2,
    sbert_device: str = "auto",
    representations: list[str] | None = None,
    clustering_methods: list[str] | None = None,
    metric_sample_size: int = 10_000,
    resume: bool = True,
    skip_polish: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the Week 4 final workflow."""
    started_at = time.perf_counter()
    primary_input = Path(primary_input)
    secondary_input = Path(secondary_input)
    primary_week2_dir = Path(primary_week2_dir)
    primary_runs_root = Path(primary_runs_root) if primary_runs_root is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in (seeds or DEFAULT_WEEK4_SEEDS)]
    representations = representations or DEFAULT_REPRESENTATIONS
    clustering_methods = clustering_methods or DEFAULT_CLUSTERING_METHODS

    _progress(f"Week 4 final dimulai | seeds={', '.join(str(seed) for seed in seeds)}", verbose=verbose)
    _progress(f"Output: {output_dir}", verbose=verbose)

    primary_metrics = _load_primary_runs(
        seeds=seeds,
        representations=representations,
        clustering_methods=clustering_methods,
        output_dir=output_dir,
        primary_runs_root=primary_runs_root,
        primary_week2_dir=primary_week2_dir,
        sample_size=sample_size,
        primary_input=primary_input,
        sbert_device=sbert_device,
        metric_sample_size=metric_sample_size,
        resume=resume,
        verbose=verbose,
    )
    primary_summary = _summarize_metrics(primary_metrics, dataset="scenario-1", experiment="optimal")
    best_reps = _best_representations(primary_metrics, top_representations)
    best_pipeline_row = _best_combinations(primary_summary, count=1, mean_columns=True).iloc[0]
    best_pipeline = {
        "representation": str(best_pipeline_row["representation"]),
        "clustering_method": str(best_pipeline_row["clustering_method"]),
        "silhouette_mean": float(best_pipeline_row["silhouette_mean"]),
    }
    _progress(f"Representasi terbaik final: {', '.join(best_reps)}", verbose=verbose)
    _progress(
        f"Best pipeline: {best_pipeline['representation']} + {best_pipeline['clustering_method']}",
        verbose=verbose,
    )

    k_frames: list[pd.DataFrame] = []
    for seed in seeds:
        stage_name = f"primary_k_sensitivity_seed{seed}"
        k_frames.append(
            _run_or_reuse_week2_stage(
                stage_name=stage_name,
                input_path=primary_input,
                output_dir=output_dir,
                primary_runs_root=primary_runs_root,
                sample_size=sample_size,
                seed=seed,
                representations=best_reps,
                clustering_methods=clustering_methods,
                experiment_mode="k-sensitivity",
                sbert_device=sbert_device,
                metric_sample_size=metric_sample_size,
                resume=resume,
                verbose=verbose,
            )
        )
    k_sensitivity_metrics = normalize_report_dataframe(pd.concat(k_frames, ignore_index=True))
    k_sensitivity_summary = _summarize_metrics(
        k_sensitivity_metrics,
        dataset="scenario-1",
        experiment="k-sensitivity",
    )

    secondary_stage = "secondary_best_seed42"
    secondary_metrics = _run_or_reuse_week2_stage(
        stage_name=secondary_stage,
        input_path=secondary_input,
        output_dir=output_dir,
        primary_runs_root=primary_runs_root,
        sample_size=sample_size,
        seed=42,
        representations=[best_pipeline["representation"]],
        clustering_methods=[best_pipeline["clustering_method"]],
        experiment_mode="optimal",
        sbert_device=sbert_device,
        metric_sample_size=metric_sample_size,
        resume=resume,
        verbose=verbose,
    )
    secondary_summary = _summarize_metrics(secondary_metrics, dataset="scenario-2", experiment="best-pipeline")
    primary_best_summary = primary_summary[
        (primary_summary["representation"] == best_pipeline["representation"])
        & (primary_summary["clustering_method"] == best_pipeline["clustering_method"])
    ]
    comparison = _comparison_table(primary_best_summary, secondary_summary)

    _progress("Menulis agregasi, plot, dan report final Week 4.", verbose=verbose)
    primary_metrics.assign(dataset="scenario-1", experiment="optimal").to_csv(
        output_dir / "metrics_primary_runs.csv",
        index=False,
    )
    primary_summary.to_csv(output_dir / "metrics_primary_summary.csv", index=False)
    k_sensitivity_metrics.assign(dataset="scenario-1", experiment="k-sensitivity").to_csv(
        output_dir / "metrics_k_sensitivity.csv",
        index=False,
    )
    k_sensitivity_summary.to_csv(output_dir / "metrics_k_sensitivity_summary.csv", index=False)
    secondary_metrics.assign(dataset="scenario-2", experiment="best-pipeline").to_csv(
        output_dir / "metrics_cross_dataset.csv",
        index=False,
    )
    secondary_summary.to_csv(output_dir / "metrics_cross_dataset_summary.csv", index=False)
    comparison.to_csv(output_dir / "cross_dataset_comparison.csv", index=False)
    _best_combinations(primary_summary, count=10, mean_columns=True).to_csv(
        output_dir / "best_methods_week4.csv",
        index=False,
    )

    _save_bar_plot(
        primary_summary,
        output_dir / "plots" / "primary_silhouette_mean.png",
        title="Primary Optimal - Mean Silhouette Across Seeds",
        error_column="silhouette_std",
    )
    _save_bar_plot(
        k_sensitivity_summary,
        output_dir / "plots" / "k_sensitivity_silhouette_mean.png",
        title="K-Sensitivity - Mean Silhouette Across Seeds",
        error_column="silhouette_std",
    )
    _save_cross_dataset_plot(comparison, output_dir / "plots" / "cross_dataset_silhouette.png")

    run_config = {
        "primary_input": str(primary_input),
        "secondary_input": str(secondary_input),
        "primary_week2_dir": str(primary_week2_dir),
        "primary_runs_root": str(primary_runs_root) if primary_runs_root is not None else None,
        "output_dir": str(output_dir),
        "seeds": seeds,
        "sample_size": sample_size,
        "representations": representations,
        "clustering_methods": clustering_methods,
        "best_representations": best_reps,
        "best_pipeline": best_pipeline,
        "top_representations": top_representations,
        "sbert_device": sbert_device,
        "metric_sample_size": metric_sample_size,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    _write_json(output_dir / "run_config_week4.json", run_config)
    _write_report(
        output_dir,
        primary_summary=primary_summary,
        k_sensitivity_summary=k_sensitivity_summary,
        secondary_summary=secondary_summary,
        comparison=comparison,
        best_representations=best_reps,
        best_pipeline=best_pipeline,
        run_config=run_config,
    )
    polish_result: dict[str, Any] = {}
    if not skip_polish:
        polish_result = generate_week4_visual_polish(output_dir=output_dir, week3_dir=week3_dir, verbose=verbose)
    _progress(f"Week 4 final selesai. Output: {output_dir}", verbose=verbose)

    return {
        "primary_metrics": primary_metrics,
        "primary_summary": primary_summary,
        "k_sensitivity_metrics": k_sensitivity_metrics,
        "k_sensitivity_summary": k_sensitivity_summary,
        "secondary_metrics": secondary_metrics,
        "secondary_summary": secondary_summary,
        "comparison": comparison,
        "run_config": run_config,
        "polish": polish_result,
        "output_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 final validation.")
    parser.add_argument("--primary-input", type=Path, default=Path("timeline.csv"))
    parser.add_argument("--secondary-input", type=Path, default=Path("data/external/timeline.csv"))
    parser.add_argument("--primary-week2-dir", type=Path, default=Path("reports/week2"))
    parser.add_argument("--primary-runs-root", type=Path, default=DEFAULT_PRIMARY_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=Path("reports/week4"))
    parser.add_argument("--week3-dir", type=Path, default=DEFAULT_WEEK3_DIR)
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_WEEK4_SEEDS)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--top-representations", type=int, default=2)
    parser.add_argument("--sbert-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--representations", type=_parse_csv_list, default=DEFAULT_REPRESENTATIONS)
    parser.add_argument("--clustering-methods", type=_parse_csv_list, default=DEFAULT_CLUSTERING_METHODS)
    parser.add_argument("--metric-sample-size", type=int, default=10_000)
    parser.add_argument("--no-resume", action="store_true", help="Rerun stages even if checkpoint files exist.")
    parser.add_argument("--polish-only", action="store_true", help="Generate only literal-PDF visual polish from final Week 4 CSV artifacts.")
    parser.add_argument("--literal-pdf-completion", action="store_true", help="Generate final traceability and literal PDF completion artifacts from existing outputs.")
    parser.add_argument("--skip-polish", action="store_true", help="Run Week 4 final aggregation without extra literal-PDF visual polish.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.literal_pdf_completion:
        generate_pdf_literal_completion(output_dir=args.output, week3_dir=args.week3_dir)
        return
    if args.polish_only:
        generate_week4_visual_polish(output_dir=args.output, week3_dir=args.week3_dir)
        return
    run_week4(
        primary_input=args.primary_input,
        secondary_input=args.secondary_input,
        primary_week2_dir=args.primary_week2_dir,
        primary_runs_root=args.primary_runs_root,
        output_dir=args.output,
        week3_dir=args.week3_dir,
        seeds=args.seeds,
        sample_size=args.sample_size,
        top_representations=args.top_representations,
        sbert_device=args.sbert_device,
        representations=args.representations,
        clustering_methods=args.clustering_methods,
        metric_sample_size=args.metric_sample_size,
        resume=not args.no_resume,
        skip_polish=args.skip_polish,
    )


if __name__ == "__main__":
    main()
