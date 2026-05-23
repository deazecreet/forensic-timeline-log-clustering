"""Week 1 TF-IDF + K-Means baseline for Plaso timeline clustering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .preprocessing import STOP_WORDS, build_clustering_text


REQUIRED_COLUMNS = [
    "datetime",
    "timestamp_desc",
    "source",
    "source_long",
    "message",
    "parser",
    "display_name",
    "tag",
]

DEFAULT_MIN_YEAR = 2000
DEFAULT_MAX_YEAR = 2026


def raise_csv_field_limit() -> None:
    """Raise csv module field limit for unusually long Plaso messages."""
    limit = 2**31 - 1
    while limit > 1024:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_k_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("k-values must contain integers >= 2")
    return values


def _json_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _top_counter(counter: Counter[str], n: int = 20) -> list[dict[str, Any]]:
    return [{"value": key, "count": int(value)} for key, value in counter.most_common(n)]


def _valid_year(value: str, min_year: int, max_year: int) -> bool:
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return False
    return min_year <= year <= max_year


def _validate_columns(fieldnames: list[str] | None) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in (fieldnames or [])]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def iter_rows(input_path: Path):
    raise_csv_field_limit()
    with input_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames)
        for row in reader:
            yield row


def profile_csv(
    input_path: Path,
    *,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
) -> dict[str, Any]:
    """Scan the CSV once and collect dataset/profile statistics."""
    total_rows = 0
    valid_rows = 0
    blank_message_rows = 0
    message_min = None
    message_max = 0
    message_total = 0
    valid_min_datetime = None
    valid_max_datetime = None
    examples_valid: list[dict[str, str]] = []
    examples_invalid: list[dict[str, str]] = []

    source_counts = Counter()
    valid_source_counts = Counter()
    invalid_source_counts = Counter()
    timestamp_desc_counts = Counter()
    parser_counts = Counter()
    tag_counts = Counter()
    valid_year_counts = Counter()

    for row in iter_rows(input_path):
        total_rows += 1
        source = row.get("source", "") or "UNKNOWN"
        timestamp_desc = row.get("timestamp_desc", "") or "UNKNOWN"
        parser = row.get("parser", "") or "UNKNOWN"
        tag = row.get("tag", "") or "UNKNOWN"
        message = row.get("message", "") or ""
        datetime_value = row.get("datetime", "") or ""

        source_counts[source] += 1
        timestamp_desc_counts[timestamp_desc] += 1
        parser_counts[parser] += 1
        tag_counts[tag] += 1

        message_len = len(message)
        if message_len == 0:
            blank_message_rows += 1
        message_total += message_len
        message_min = message_len if message_min is None else min(message_min, message_len)
        message_max = max(message_max, message_len)

        if message and _valid_year(datetime_value, min_year, max_year):
            valid_rows += 1
            valid_source_counts[source] += 1
            valid_year_counts[datetime_value[:4]] += 1
            if valid_min_datetime is None or datetime_value < valid_min_datetime:
                valid_min_datetime = datetime_value
            if valid_max_datetime is None or datetime_value > valid_max_datetime:
                valid_max_datetime = datetime_value
            if len(examples_valid) < 5:
                examples_valid.append(
                    {
                        "datetime": datetime_value,
                        "source": source,
                        "timestamp_desc": timestamp_desc,
                        "message": message[:180],
                    }
                )
        else:
            invalid_source_counts[source] += 1
            if len(examples_invalid) < 5:
                examples_invalid.append(
                    {
                        "datetime": datetime_value,
                        "source": source,
                        "timestamp_desc": timestamp_desc,
                        "message": message[:180],
                    }
                )

    return {
        "input_file": input_path.name,
        "file_size_bytes": int(input_path.stat().st_size),
        "required_columns": REQUIRED_COLUMNS,
        "year_filter": {"min_year": min_year, "max_year": max_year},
        "total_rows": int(total_rows),
        "valid_rows": int(valid_rows),
        "invalid_or_empty_rows": int(total_rows - valid_rows),
        "blank_message_rows": int(blank_message_rows),
        "valid_datetime_min": valid_min_datetime,
        "valid_datetime_max": valid_max_datetime,
        "message_length": {
            "min": int(message_min or 0),
            "average": round(message_total / total_rows, 2) if total_rows else 0,
            "max": int(message_max),
        },
        "source_counts": _json_counter(source_counts),
        "source_counts_valid": _json_counter(valid_source_counts),
        "source_counts_invalid_or_empty": _json_counter(invalid_source_counts),
        "valid_year_counts": _json_counter(valid_year_counts),
        "top_sources": _top_counter(source_counts),
        "top_valid_sources": _top_counter(valid_source_counts),
        "top_timestamp_desc": _top_counter(timestamp_desc_counts),
        "top_parsers": _top_counter(parser_counts),
        "top_tags": _top_counter(tag_counts),
        "examples_valid": examples_valid,
        "examples_invalid_or_empty": examples_invalid,
    }


def build_stratified_quotas(source_counts: dict[str, int], sample_size: int) -> dict[str, int]:
    """Allocate deterministic per-source sample quotas."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    counts = {source: int(count) for source, count in source_counts.items() if int(count) > 0}
    if not counts:
        raise ValueError("No valid rows available for sampling")

    total = sum(counts.values())
    if sample_size >= total:
        return dict(sorted(counts.items()))

    sources = sorted(counts)
    if sample_size < len(sources):
        top_sources = sorted(sources, key=lambda source: (-counts[source], source))[:sample_size]
        return {source: 1 for source in sorted(top_sources)}

    ideal = {source: (counts[source] / total) * sample_size for source in sources}
    quotas = {source: min(counts[source], max(1, math.floor(ideal[source]))) for source in sources}

    while sum(quotas.values()) > sample_size:
        candidates = [source for source in sources if quotas[source] > 1]
        if not candidates:
            break
        source = max(candidates, key=lambda item: (quotas[item] - ideal[item], item))
        quotas[source] -= 1

    while sum(quotas.values()) < sample_size:
        candidates = [source for source in sources if quotas[source] < counts[source]]
        if not candidates:
            break
        source = max(candidates, key=lambda item: (ideal[item] - quotas[item], counts[item], item))
        quotas[source] += 1

    return {source: int(quota) for source, quota in sorted(quotas.items()) if quota > 0}


def sample_stratified_rows(
    input_path: Path,
    quotas: dict[str, int],
    *,
    seed: int,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
) -> pd.DataFrame:
    """Reservoir-sample valid rows independently per source."""
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = Counter()

    for row in iter_rows(input_path):
        source = row.get("source", "") or "UNKNOWN"
        if source not in quotas:
            continue

        datetime_value = row.get("datetime", "") or ""
        message = row.get("message", "") or ""
        if not message or not _valid_year(datetime_value, min_year, max_year):
            continue

        seen[source] += 1
        item = {column: row.get(column, "") for column in REQUIRED_COLUMNS}
        item["clean_text"] = build_clustering_text(source, message)

        quota = quotas[source]
        bucket = reservoirs[source]
        if len(bucket) < quota:
            bucket.append(item)
            continue

        replacement_index = rng.randrange(seen[source])
        if replacement_index < quota:
            bucket[replacement_index] = item

    rows = [row for source in sorted(reservoirs) for row in reservoirs[source]]
    rng.shuffle(rows)
    if not rows:
        raise ValueError("Sampling produced no rows")
    return pd.DataFrame(rows)


def _reduce_dimensions(
    matrix,
    *,
    requested_components: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    n_samples, n_features = matrix.shape
    if n_features == 0:
        raise ValueError("TF-IDF produced zero features")
    if n_features == 1 or n_samples <= 2:
        return matrix.toarray(), min(n_features, requested_components)

    n_components = min(requested_components, n_features - 1, n_samples - 1)
    reducer = make_pipeline(
        TruncatedSVD(n_components=n_components, random_state=seed),
        Normalizer(copy=False),
    )
    return reducer.fit_transform(matrix), int(n_components)


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        value = metric_fn(*args, **kwargs)
    except ValueError:
        return float("nan")
    return float(value)


def _format_counter(counter: Counter[str], limit: int = 5) -> str:
    return "; ".join(f"{key}:{int(value)}" for key, value in counter.most_common(limit))


def _top_terms_for_cluster(matrix, feature_names: np.ndarray, row_indexes: np.ndarray, limit: int = 10) -> str:
    if row_indexes.size == 0:
        return ""
    means = np.asarray(matrix[row_indexes].mean(axis=0)).ravel()
    if means.size == 0:
        return ""
    top_indexes = means.argsort()[-limit:][::-1]
    return "; ".join(feature_names[index] for index in top_indexes if means[index] > 0)


def build_cluster_outputs(
    sample_df: pd.DataFrame,
    matrix,
    vectorizer: TfidfVectorizer,
    labels_by_k: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for k, labels in labels_by_k.items():
        label_series = pd.Series(labels, name="cluster_id")
        for cluster_id in sorted(label_series.unique()):
            row_indexes = np.flatnonzero(labels == cluster_id)
            cluster_df = sample_df.iloc[row_indexes]
            source_counts = Counter(cluster_df["source"])
            examples = [
                " ".join(str(message).split())[:160]
                for message in cluster_df["message"].head(3)
            ]
            summary_rows.append(
                {
                    "k": int(k),
                    "cluster_id": int(cluster_id),
                    "size": int(row_indexes.size),
                    "top_terms": _top_terms_for_cluster(matrix, feature_names, row_indexes),
                    "top_sources": _format_counter(source_counts),
                    "example_messages": " || ".join(examples),
                }
            )
            for source, count in sorted(source_counts.items()):
                source_rows.append(
                    {
                        "k": int(k),
                        "cluster_id": int(cluster_id),
                        "source": source,
                        "count": int(count),
                        "share_cluster": round(count / row_indexes.size, 6) if row_indexes.size else 0,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(source_rows)


def save_plots(metrics_df: pd.DataFrame, sample_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(metrics_df["k"], metrics_df["inertia"], marker="o")
    plt.title("Elbow Plot TF-IDF + K-Means")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_inertia.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="#2471a3")
    plt.title("Silhouette Score TF-IDF + K-Means")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_scores.png", dpi=150)
    plt.close()

    source_counts = sample_df["source"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(8, 4.5))
    source_counts.plot(kind="barh", color="#2e7d32")
    plt.title("Distribusi Source pada Sample")
    plt.xlabel("Jumlah event")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.savefig(output_dir / "source_distribution_sample.png", dpi=150)
    plt.close()


def run_week1(
    *,
    input_path: Path,
    output_dir: Path,
    sample_size: int = 50_000,
    seed: int = 42,
    k_values: list[int] | None = None,
    max_features: int = 5000,
    svd_components: int = 50,
    min_df: int = 5,
    max_df: float = 0.95,
    metric_sample_size: int = 10_000,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
) -> dict[str, Any]:
    """Run the complete Week 1 baseline and write report artifacts."""
    k_values = k_values or [10, 20, 50]
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_start = time.perf_counter()
    profile = profile_csv(input_path, min_year=min_year, max_year=max_year)
    profile["profiling_time_s"] = round(time.perf_counter() - profile_start, 3)

    quotas = build_stratified_quotas(profile["source_counts_valid"], sample_size)
    sample_start = time.perf_counter()
    sample_df = sample_stratified_rows(
        input_path,
        quotas,
        seed=seed,
        min_year=min_year,
        max_year=max_year,
    )
    sampling_time_s = time.perf_counter() - sample_start

    embedding_start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z_][a-z_]+\b",
        stop_words=list(STOP_WORDS),
    )
    matrix = vectorizer.fit_transform(sample_df["clean_text"])
    reduced_matrix, effective_components = _reduce_dimensions(
        matrix,
        requested_components=svd_components,
        seed=seed,
    )
    embedding_time_s = time.perf_counter() - embedding_start

    metrics_rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}

    for k in k_values:
        clustering_start = time.perf_counter()
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(reduced_matrix)
        clustering_time_s = time.perf_counter() - clustering_start
        labels_by_k[k] = labels

        distinct_labels = len(set(labels))
        can_score = 1 < distinct_labels < len(labels)
        silhouette = (
            _safe_metric(
                silhouette_score,
                reduced_matrix,
                labels,
                sample_size=min(metric_sample_size, len(labels) - 1),
                random_state=seed,
            )
            if can_score
            else float("nan")
        )
        calinski = _safe_metric(calinski_harabasz_score, reduced_matrix, labels) if can_score else float("nan")
        davies = _safe_metric(davies_bouldin_score, reduced_matrix, labels) if can_score else float("nan")

        metrics_rows.append(
            {
                "sample_size": int(len(sample_df)),
                "seed": int(seed),
                "k": int(k),
                "tfidf_features": int(matrix.shape[1]),
                "svd_components": int(effective_components),
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "inertia": float(model.inertia_),
                "embedding_time_s": round(embedding_time_s, 3),
                "clustering_time_s": round(clustering_time_s, 3),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    cluster_summary_df, source_by_cluster_df = build_cluster_outputs(
        sample_df,
        matrix,
        vectorizer,
        labels_by_k,
    )

    profile["sample"] = {
        "requested_sample_size": int(sample_size),
        "actual_sample_size": int(len(sample_df)),
        "seed": int(seed),
        "sampling_time_s": round(sampling_time_s, 3),
        "quotas_by_source": quotas,
        "actual_counts_by_source": {
            source: int(count) for source, count in sorted(sample_df["source"].value_counts().items())
        },
    }
    profile["tfidf"] = {
        "max_features": int(max_features),
        "actual_features": int(matrix.shape[1]),
        "min_df": int(min_df),
        "max_df": float(max_df),
        "ngram_range": [1, 2],
    }
    profile["dimensionality_reduction"] = {
        "method": "TruncatedSVD + Normalizer",
        "requested_components": int(svd_components),
        "actual_components": int(effective_components),
    }
    profile["clustering"] = {
        "method": "KMeans",
        "k_values": [int(k) for k in k_values],
        "seed": int(seed),
    }

    metrics_df.to_csv(output_dir / "metrics_week1.csv", index=False)
    cluster_summary_df.to_csv(output_dir / "cluster_summary_week1.csv", index=False)
    source_by_cluster_df.to_csv(output_dir / "source_by_cluster.csv", index=False)
    with (output_dir / "dataset_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    save_plots(metrics_df, sample_df, output_dir)

    return {
        "profile": profile,
        "metrics": metrics_df,
        "cluster_summary": cluster_summary_df,
        "source_by_cluster": source_by_cluster_df,
        "output_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 1 log clustering baseline.")
    parser.add_argument("--input", type=Path, default=Path("timeline.csv"), help="Path to Plaso timeline CSV.")
    parser.add_argument("--output", type=Path, default=Path("reports/week1"), help="Output report directory.")
    parser.add_argument("--sample-size", type=int, default=50_000, help="Valid events to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k-values", type=parse_k_values, default=[10, 20, 50], help="Comma-separated k values.")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max_features.")
    parser.add_argument("--svd-components", type=int, default=50, help="SVD dimensions before clustering.")
    parser.add_argument("--min-df", type=int, default=5, help="TF-IDF min_df.")
    parser.add_argument("--max-df", type=float, default=0.95, help="TF-IDF max_df.")
    parser.add_argument("--metric-sample-size", type=int, default=10_000, help="Silhouette sample size.")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR, help="Minimum accepted event year.")
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR, help="Maximum accepted event year.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_week1(
        input_path=args.input,
        output_dir=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        k_values=args.k_values,
        max_features=args.max_features,
        svd_components=args.svd_components,
        min_df=args.min_df,
        max_df=args.max_df,
        metric_sample_size=args.metric_sample_size,
        min_year=args.min_year,
        max_year=args.max_year,
    )
    metrics = result["metrics"]
    print(f"Week 1 baseline complete. Output: {result['output_dir']}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
