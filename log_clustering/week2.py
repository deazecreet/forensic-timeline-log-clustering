"""Week 2 embedding + clustering experiments for Plaso timeline events."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from .week1 import (
    DEFAULT_MAX_YEAR,
    DEFAULT_MIN_YEAR,
    build_stratified_quotas,
    profile_csv,
    sample_stratified_rows,
)
from .preprocessing import STOP_WORDS


DEFAULT_REPRESENTATIONS = ["tfidf", "word2vec", "doc2vec", "sbert"]
DEFAULT_CLUSTERING_METHODS = ["kmeans", "dbscan", "hdbscan", "agglomerative", "gmm"]
REPORT_REPRESENTATION_ORDER = ["tfidf", "word2vec", "doc2vec", "sbert"]
REPORT_CLUSTERING_ORDER = ["agglomerative", "kmeans", "dbscan", "hdbscan", "gmm"]
DEFAULT_OPTIMAL_CLUSTER_COUNTS = [2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100]
DEFAULT_K_SENSITIVITY_CLUSTER_COUNTS = [10, 20, 50]


def _progress(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def _parse_csv_list(raw: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("integer list must contain values >= 2")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("float list cannot be empty")
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _hash_config(config: dict[str, Any]) -> str:
    payload = json.dumps(_json_safe(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _tokens_from_sample(sample_df: pd.DataFrame) -> list[list[str]]:
    return [
        [token for token in str(text).split() if token]
        for text in sample_df["clean_text"].fillna("")
    ]


def _require_gensim():
    try:
        from gensim.models import Doc2Vec, Word2Vec
        from gensim.models.callbacks import CallbackAny2Vec
        from gensim.models.doc2vec import TaggedDocument
    except ImportError as exc:
        raise ImportError(
            "Word2Vec/Doc2Vec membutuhkan dependency gensim. Jalankan: pip install gensim"
        ) from exc
    return Word2Vec, Doc2Vec, TaggedDocument, CallbackAny2Vec


def build_word2vec_embeddings(
    tokenized_texts: list[list[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
    workers: int,
    verbose: bool,
) -> np.ndarray:
    Word2Vec, _, _, CallbackAny2Vec = _require_gensim()

    class EpochProgress(CallbackAny2Vec):
        def __init__(self) -> None:
            self.epoch = 0

        def on_epoch_end(self, model) -> None:
            self.epoch += 1
            _progress(f"    Word2Vec epoch {self.epoch}/{epochs} selesai", verbose=verbose)

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        seed=seed,
        epochs=epochs,
        callbacks=[EpochProgress()] if verbose else (),
    )
    zero_vector = np.zeros(vector_size, dtype=np.float32)
    rows = []
    for tokens in tokenized_texts:
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        rows.append(np.mean(vectors, axis=0) if vectors else zero_vector)
    return np.vstack(rows).astype(np.float32)


def build_doc2vec_embeddings(
    tokenized_texts: list[list[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
    workers: int,
    verbose: bool,
) -> np.ndarray:
    _, Doc2Vec, TaggedDocument, CallbackAny2Vec = _require_gensim()
    documents = [TaggedDocument(words=tokens, tags=[index]) for index, tokens in enumerate(tokenized_texts)]

    class EpochProgress(CallbackAny2Vec):
        def __init__(self) -> None:
            self.epoch = 0

        def on_epoch_end(self, model) -> None:
            self.epoch += 1
            _progress(f"    Doc2Vec epoch {self.epoch}/{epochs} selesai", verbose=verbose)

    model = Doc2Vec(
        documents=documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        dm=1,
        seed=seed,
        epochs=epochs,
        callbacks=[EpochProgress()] if verbose else (),
    )
    return np.vstack([model.dv[index] for index in range(len(tokenized_texts))]).astype(np.float32)


def build_sbert_embeddings(
    texts: pd.Series,
    *,
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "SBERT membutuhkan dependency sentence-transformers. Jalankan: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_tfidf_embeddings(
    texts: pd.Series,
    *,
    max_features: int,
    min_df: int,
    max_df: float,
    svd_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z_][a-z_]+\b",
        stop_words=list(STOP_WORDS),
    )
    matrix = vectorizer.fit_transform(texts.fillna("").astype(str))
    n_samples, n_features = matrix.shape
    if n_features == 0:
        raise ValueError("TF-IDF produced zero features")

    if n_samples > 2 and n_features > svd_components:
        effective_components = min(svd_components, n_features - 1, n_samples - 1)
        reducer = make_pipeline(
            TruncatedSVD(n_components=effective_components, random_state=seed),
            Normalizer(copy=False),
        )
        embeddings = reducer.fit_transform(matrix).astype(np.float32)
        reduced = True
    else:
        effective_components = n_features
        embeddings = matrix.toarray().astype(np.float32)
        reduced = False

    return embeddings, {
        "tfidf_features": int(n_features),
        "tfidf_max_features": int(max_features),
        "tfidf_min_df": int(min_df),
        "tfidf_max_df": float(max_df),
        "tfidf_svd_applied": bool(reduced),
        "tfidf_svd_components": int(effective_components),
    }


def build_embeddings(
    sample_df: pd.DataFrame,
    representation: str,
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
    workers: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
    sbert_model: str,
    sbert_batch_size: int,
    sbert_device: str,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenized_texts = _tokens_from_sample(sample_df)
    if representation == "tfidf":
        return build_tfidf_embeddings(
            sample_df["clean_text"],
            max_features=tfidf_max_features,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df,
            svd_components=50,
            seed=seed,
        )
    if representation == "word2vec":
        return build_word2vec_embeddings(
            tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            seed=seed,
            workers=workers,
            verbose=verbose,
        ), {}
    if representation == "doc2vec":
        return build_doc2vec_embeddings(
            tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            seed=seed,
            workers=workers,
            verbose=verbose,
        ), {}
    if representation == "sbert":
        return build_sbert_embeddings(
            sample_df["clean_text"],
            model_name=sbert_model,
            batch_size=sbert_batch_size,
            device=sbert_device,
        ), {}
    raise ValueError(f"Unknown representation: {representation}")


def build_embedding_cache_key(
    *,
    representation: str,
    input_path: Path,
    sample_size: int,
    seed: int,
    min_year: int,
    max_year: int,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
    sbert_model: str,
    sbert_batch_size: int,
    sbert_device: str,
) -> str:
    config: dict[str, Any] = {
        "representation": representation,
        "input_name": input_path.name,
        "input_size_bytes": input_path.stat().st_size if input_path.exists() else None,
        "input_mtime_ns": input_path.stat().st_mtime_ns if input_path.exists() else None,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "min_year": int(min_year),
        "max_year": int(max_year),
    }
    if representation in {"word2vec", "doc2vec"}:
        config.update(
            {
                "vector_size": int(vector_size),
                "window": int(window),
                "min_count": int(min_count),
                "epochs": int(epochs),
            }
        )
    if representation == "tfidf":
        config.update(
            {
                "tfidf_max_features": int(tfidf_max_features),
                "tfidf_min_df": int(tfidf_min_df),
                "tfidf_max_df": float(tfidf_max_df),
                "tfidf_svd_components": 50,
            }
        )
    if representation == "sbert":
        config.update(
            {
                "sbert_model": sbert_model,
                "sbert_batch_size": int(sbert_batch_size),
                "sbert_device": sbert_device,
            }
        )
    return f"{representation}_{_hash_config(config)}"


def load_or_build_embeddings(
    sample_df: pd.DataFrame,
    representation: str,
    *,
    input_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    min_year: int,
    max_year: int,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    workers: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
    sbert_model: str,
    sbert_batch_size: int,
    sbert_device: str,
    use_cache: bool,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_key = build_embedding_cache_key(
        representation=representation,
        input_path=input_path,
        sample_size=sample_size,
        seed=seed,
        min_year=min_year,
        max_year=max_year,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        tfidf_max_df=tfidf_max_df,
        sbert_model=sbert_model,
        sbert_batch_size=sbert_batch_size,
        sbert_device=sbert_device,
    )
    cache_dir = output_dir / "cache"
    cache_path = cache_dir / f"{cache_key}.npz"

    cache_info = {
        "embedding_cache_key": cache_key,
        "embedding_cache_path": str(cache_path),
        "embedding_cache_hit": False,
    }
    if use_cache and cache_path.exists():
        stage_start = time.perf_counter()
        _progress(f"  Load embedding cache: {cache_path}", verbose=verbose)
        with np.load(cache_path, allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            representation_info = (
                json.loads(str(data["representation_info"]))
                if "representation_info" in data.files
                else {}
            )
        cache_info["embedding_cache_hit"] = True
        cache_info["embedding_load_time_s"] = round(time.perf_counter() - stage_start, 3)
        return embeddings, {**cache_info, **representation_info}

    embeddings, representation_info = build_embeddings(
        sample_df,
        representation,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seed=seed,
        workers=workers,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        tfidf_max_df=tfidf_max_df,
        sbert_model=sbert_model,
        sbert_batch_size=sbert_batch_size,
        sbert_device=sbert_device,
        verbose=verbose,
    )

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _progress(f"  Simpan embedding cache: {cache_path}", verbose=verbose)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings.astype(np.float32),
            representation_info=json.dumps(_json_safe(representation_info), sort_keys=True),
        )

    return embeddings, {**cache_info, **representation_info}


def resolve_sbert_device(requested_device: str) -> str:
    """Resolve SBERT device and fail loudly when CUDA was requested but unavailable."""
    requested_device = requested_device.lower().strip()
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("sbert_device must be one of: auto, cuda, cpu")

    try:
        import torch
    except ImportError as exc:
        if requested_device == "cuda":
            raise RuntimeError("PyTorch belum terpasang, jadi SBERT tidak bisa memakai CUDA.") from exc
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA diminta untuk SBERT, tetapi PyTorch di environment ini tidak mendeteksi CUDA. "
            "Install PyTorch CUDA build terlebih dahulu."
        )
    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested_device


def prepare_clustering_matrix(
    embeddings: np.ndarray,
    *,
    pca_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D matrix")

    matrix = np.nan_to_num(embeddings, copy=True)
    n_samples, n_features = matrix.shape
    info: dict[str, Any] = {
        "embedding_dimensions": int(n_features),
        "requested_pca_components": int(pca_components),
        "pca_applied": False,
        "effective_dimensions": int(n_features),
    }

    if n_samples > 2 and n_features > pca_components:
        effective_components = min(pca_components, n_features, n_samples - 1)
        pca = PCA(n_components=effective_components, random_state=seed)
        matrix = pca.fit_transform(matrix)
        info.update(
            {
                "pca_applied": True,
                "effective_dimensions": int(effective_components),
                "pca_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
            }
        )

    matrix = StandardScaler().fit_transform(matrix)
    return matrix.astype(np.float32), info


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        value = metric_fn(*args, **kwargs)
    except ValueError:
        return float("nan")
    return float(value)


def evaluate_labels(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    metric_sample_size: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    clustered_mask = labels != -1
    scoring_matrix = matrix[clustered_mask]
    scoring_labels = labels[clustered_mask]
    cluster_ids = sorted(label for label in set(labels.tolist()) if label != -1)
    noise_count = int(np.sum(labels == -1))
    can_score = 1 < len(set(scoring_labels.tolist())) < len(scoring_labels)
    can_score_all = 1 < len(set(labels.tolist())) < len(labels)

    return {
        "n_clusters": int(len(cluster_ids)),
        "noise_count": noise_count,
        "noise_ratio": round(noise_count / len(labels), 6) if len(labels) else 0,
        "silhouette": (
            _safe_metric(
                silhouette_score,
                scoring_matrix,
                scoring_labels,
                sample_size=min(metric_sample_size, len(scoring_labels) - 1),
                random_state=seed,
            )
            if can_score
            else float("nan")
        ),
        "calinski_harabasz": (
            _safe_metric(calinski_harabasz_score, scoring_matrix, scoring_labels)
            if can_score
            else float("nan")
        ),
        "davies_bouldin": (
            _safe_metric(davies_bouldin_score, scoring_matrix, scoring_labels)
            if can_score
            else float("nan")
        ),
        "silhouette_all": (
            _safe_metric(
                silhouette_score,
                matrix,
                labels,
                sample_size=min(metric_sample_size, len(labels) - 1),
                random_state=seed,
            )
            if can_score_all
            else float("nan")
        ),
        "calinski_harabasz_all": (
            _safe_metric(calinski_harabasz_score, matrix, labels)
            if can_score_all
            else float("nan")
        ),
        "davies_bouldin_all": (
            _safe_metric(davies_bouldin_score, matrix, labels)
            if can_score_all
            else float("nan")
        ),
    }


def _score_for_selection(metrics: dict[str, Any]) -> float:
    silhouette = metrics.get("silhouette", float("nan"))
    if np.isnan(silhouette):
        return -1_000_000.0 + float(metrics.get("n_clusters", 0))
    return float(silhouette)


def _valid_cluster_counts(cluster_counts: list[int], n_samples: int) -> list[int]:
    return sorted({count for count in cluster_counts if 1 < count < n_samples})


def _dbscan_eps_candidates(matrix: np.ndarray, min_samples: int, quantiles: list[float]) -> list[float]:
    if len(matrix) <= min_samples:
        return []
    n_neighbors = min(min_samples, len(matrix) - 1)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)
    kth_distances = np.sort(distances[:, -1])
    positive_distances = kth_distances[kth_distances > 0]
    if positive_distances.size == 0:
        return [0.5]
    values = {
        round(float(np.quantile(positive_distances, min(max(quantile, 0.0), 1.0))), 6)
        for quantile in quantiles
    }
    return sorted(value for value in values if value > 0)


def _run_clustering_trial(
    matrix: np.ndarray,
    method: str,
    params: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "dbscan":
        model = DBSCAN(**params)
        return model.fit_predict(matrix), {}
    if method == "hdbscan":
        model = HDBSCAN(**params)
        return model.fit_predict(matrix), {}
    if method == "kmeans":
        model = KMeans(random_state=seed, n_init=10, **params)
        labels = model.fit_predict(matrix)
        return labels, {"inertia": float(model.inertia_)}
    if method == "agglomerative":
        model = AgglomerativeClustering(**params)
        return model.fit_predict(matrix), {}
    if method == "gmm":
        gmm_matrix = matrix.astype(np.float64, copy=False)
        model = GaussianMixture(random_state=seed, **params)
        labels = model.fit_predict(gmm_matrix)
        return labels, {"bic": float(model.bic(gmm_matrix)), "aic": float(model.aic(gmm_matrix))}
    raise ValueError(f"Unknown clustering method: {method}")


def _parameter_grid(
    matrix: np.ndarray,
    method: str,
    *,
    cluster_counts: list[int],
    dbscan_min_samples: list[int],
    dbscan_eps_quantiles: list[float],
    hdbscan_min_cluster_sizes: list[int],
) -> list[dict[str, Any]]:
    n_samples = len(matrix)
    if method == "dbscan":
        params = []
        for min_samples in dbscan_min_samples:
            for eps in _dbscan_eps_candidates(matrix, min_samples, dbscan_eps_quantiles):
                params.append({"eps": eps, "min_samples": int(min_samples), "metric": "euclidean"})
        return params
    if method == "hdbscan":
        valid_sizes = [size for size in hdbscan_min_cluster_sizes if 1 < size < n_samples]
        return [
            {"min_cluster_size": int(size), "min_samples": None, "metric": "euclidean", "allow_single_cluster": False}
            for size in sorted(set(valid_sizes))
        ]
    if method == "kmeans":
        return [
            {"n_clusters": int(count)}
            for count in _valid_cluster_counts(cluster_counts, n_samples)
        ]
    if method == "agglomerative":
        return [
            {"n_clusters": int(count), "linkage": "ward"}
            for count in _valid_cluster_counts(cluster_counts, n_samples)
        ]
    if method == "gmm":
        return [
            {"n_components": int(count), "covariance_type": "diag", "reg_covar": float(reg_covar)}
            for count in _valid_cluster_counts(cluster_counts, n_samples)
            for reg_covar in [1e-6, 1e-5, 1e-4, 1e-3]
        ]
    raise ValueError(f"Unknown clustering method: {method}")


def tune_clustering_method(
    matrix: np.ndarray,
    method: str,
    *,
    representation: str,
    seed: int,
    metric_sample_size: int,
    cluster_counts: list[int],
    dbscan_min_samples: list[int],
    dbscan_eps_quantiles: list[float],
    hdbscan_min_cluster_sizes: list[int],
    verbose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_labels: np.ndarray | None = None
    grid = _parameter_grid(
        matrix,
        method,
        cluster_counts=cluster_counts,
        dbscan_min_samples=dbscan_min_samples,
        dbscan_eps_quantiles=dbscan_eps_quantiles,
        hdbscan_min_cluster_sizes=hdbscan_min_cluster_sizes,
    )

    _progress(f"  [{representation}] tuning {method}: {len(grid)} trial", verbose=verbose)
    for trial_index, params in enumerate(grid, start=1):
        _progress(
            f"    [{method} trial {trial_index}/{len(grid)}] params={json.dumps(_json_safe(params), sort_keys=True)}",
            verbose=verbose,
        )
        start = time.perf_counter()
        try:
            labels, model_info = _run_clustering_trial(matrix, method, params, seed=seed)
        except ValueError as exc:
            clustering_time_s = time.perf_counter() - start
            trial = {
                "representation": representation,
                "clustering_method": method,
                "seed": int(seed),
                "params": json.dumps(_json_safe(params), sort_keys=True),
                "clustering_time_s": round(clustering_time_s, 3),
                "n_clusters": 0,
                "noise_count": 0,
                "noise_ratio": 0,
                "silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan"),
                "silhouette_all": float("nan"),
                "calinski_harabasz_all": float("nan"),
                "davies_bouldin_all": float("nan"),
                "error": str(exc)[:500],
            }
            trials.append(trial)
            _progress(
                f"      gagal {_format_elapsed(clustering_time_s)} | {str(exc)[:160]}",
                verbose=verbose,
            )
            continue
        clustering_time_s = time.perf_counter() - start
        metrics = evaluate_labels(
            matrix,
            labels,
            seed=seed,
            metric_sample_size=metric_sample_size,
        )
        trial = {
            "representation": representation,
            "clustering_method": method,
            "seed": int(seed),
            "params": json.dumps(_json_safe(params), sort_keys=True),
            "clustering_time_s": round(clustering_time_s, 3),
            "error": "",
            **metrics,
            **model_info,
        }
        trials.append(trial)

        if method == "gmm" and not np.isnan(trial.get("bic", float("nan"))):
            selection_score = -float(trial["bic"])
        else:
            selection_score = _score_for_selection(metrics)

        if best_trial is None or selection_score > float(best_trial["_selection_score"]):
            best_trial = {**trial, "_selection_score": selection_score}
            best_labels = labels
        _progress(
            "      selesai "
            f"{_format_elapsed(clustering_time_s)} | "
            f"clusters={metrics['n_clusters']} | "
            f"noise={metrics['noise_ratio']:.2%} | "
            f"silhouette={metrics['silhouette']:.4f}",
            verbose=verbose,
        )

    if best_trial is None or best_labels is None:
        raise ValueError(f"No valid hyperparameter trials for {method}")

    best_trial.pop("_selection_score", None)
    return best_trial, trials, best_labels


def _format_counter(counter: Counter[str], limit: int = 5) -> str:
    return "; ".join(f"{key}:{int(value)}" for key, value in counter.most_common(limit))


def _top_tokens(texts: pd.Series, limit: int = 12) -> str:
    counter: Counter[str] = Counter()
    for text in texts.fillna(""):
        counter.update(str(text).split())
    return "; ".join(token for token, _ in counter.most_common(limit))


def build_cluster_summary(
    sample_df: pd.DataFrame,
    labels_by_combo: dict[tuple[str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for (representation, method), labels in labels_by_combo.items():
        for cluster_id in sorted(set(labels.tolist())):
            row_indexes = np.flatnonzero(labels == cluster_id)
            cluster_df = sample_df.iloc[row_indexes]
            source_counts = Counter(cluster_df["source"])
            examples = [" ".join(str(message).split())[:160] for message in cluster_df["message"].head(3)]
            summary_rows.append(
                {
                    "representation": representation,
                    "clustering_method": method,
                    "cluster_id": int(cluster_id),
                    "is_noise": bool(cluster_id == -1),
                    "size": int(row_indexes.size),
                    "top_tokens": _top_tokens(cluster_df["clean_text"]),
                    "top_sources": _format_counter(source_counts),
                    "example_messages": " || ".join(examples),
                }
            )
            for source, count in sorted(source_counts.items()):
                source_rows.append(
                    {
                        "representation": representation,
                        "clustering_method": method,
                        "cluster_id": int(cluster_id),
                        "source": source,
                        "count": int(count),
                        "share_cluster": round(count / row_indexes.size, 6) if row_indexes.size else 0,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(source_rows)


def save_week2_plots(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_df.empty:
        return

    pivot = metrics_df.pivot_table(
        index="representation",
        columns="clustering_method",
        values="silhouette",
        aggfunc="mean",
    )
    plt.figure(figsize=(8, 4.5))
    image = plt.imshow(pivot.fillna(np.nan).to_numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(image, label="Silhouette")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Week 2 Silhouette Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_heatmap_week2.png", dpi=150)
    plt.close()

    time_df = metrics_df.copy()
    time_df["total_time_s"] = time_df["embedding_time_s"] + time_df["clustering_time_s"]
    plt.figure(figsize=(7, 4.5))
    for representation, group in time_df.groupby("representation"):
        plt.scatter(group["total_time_s"], group["silhouette"], label=representation, s=60)
    plt.xlabel("Embedding + clustering time (s)")
    plt.ylabel("Silhouette")
    plt.title("Quality vs Runtime")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "quality_runtime_week2.png", dpi=150)
    plt.close()


def merge_with_existing_output(
    path: Path,
    new_df: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    """Merge a partial Week 2 run into an existing output CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return new_df

    existing_df = pd.read_csv(path)
    if existing_df.empty:
        return new_df

    combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    available_keys = [column for column in key_columns if column in combined_df.columns]
    if available_keys:
        combined_df = combined_df.drop_duplicates(subset=available_keys, keep="last")
    return combined_df.reset_index(drop=True)


def normalize_report_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stable report ordering and keep noise-aware metrics only where relevant."""
    if df.empty:
        return df

    df = df.copy()
    noise_metric_columns = [
        "silhouette_all",
        "calinski_harabasz_all",
        "davies_bouldin_all",
    ]
    if "clustering_method" in df.columns:
        non_density_mask = ~df["clustering_method"].isin(["dbscan", "hdbscan"])
        for column in noise_metric_columns:
            if column in df.columns:
                df.loc[non_density_mask, column] = pd.NA

    sort_columns: list[str] = []
    if "representation" in df.columns:
        df["_representation_order"] = pd.Categorical(
            df["representation"],
            categories=REPORT_REPRESENTATION_ORDER,
            ordered=True,
        )
        sort_columns.append("_representation_order")
    if "clustering_method" in df.columns:
        df["_clustering_order"] = pd.Categorical(
            df["clustering_method"],
            categories=REPORT_CLUSTERING_ORDER,
            ordered=True,
        )
        sort_columns.append("_clustering_order")

    for column in ["seed", "sample_size", "cluster_id", "source", "params"]:
        if column in df.columns:
            sort_columns.append(column)

    if sort_columns:
        df = df.sort_values(sort_columns, kind="stable")

    return df.drop(columns=[column for column in ["_representation_order", "_clustering_order"] if column in df])


def run_week2(
    *,
    input_path: Path,
    output_dir: Path,
    sample_size: int = 50_000,
    seed: int = 42,
    representations: list[str] | None = None,
    clustering_methods: list[str] | None = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 20,
    workers: int = 1,
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 5,
    tfidf_max_df: float = 0.95,
    sbert_model: str = "all-MiniLM-L6-v2",
    sbert_batch_size: int = 64,
    sbert_device: str = "auto",
    pca_components: int = 50,
    experiment_mode: str = "optimal",
    cluster_counts: list[int] | None = None,
    dbscan_min_samples: list[int] | None = None,
    dbscan_eps_quantiles: list[float] | None = None,
    hdbscan_min_cluster_sizes: list[int] | None = None,
    metric_sample_size: int = 10_000,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    use_embedding_cache: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Week 2 representations, clustering grids, and internal evaluation."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    representations = representations or DEFAULT_REPRESENTATIONS
    clustering_methods = clustering_methods or DEFAULT_CLUSTERING_METHODS
    if experiment_mode not in {"optimal", "k-sensitivity"}:
        raise ValueError("experiment_mode must be one of: optimal, k-sensitivity")
    if cluster_counts is None:
        cluster_counts = (
            DEFAULT_K_SENSITIVITY_CLUSTER_COUNTS
            if experiment_mode == "k-sensitivity"
            else DEFAULT_OPTIMAL_CLUSTER_COUNTS
        )
    dbscan_min_samples = dbscan_min_samples or [5, 10]
    dbscan_eps_quantiles = dbscan_eps_quantiles or [0.8, 0.9, 0.95]
    hdbscan_min_cluster_sizes = hdbscan_min_cluster_sizes or [15, 30, 50]

    unknown_representations = sorted(set(representations) - set(DEFAULT_REPRESENTATIONS))
    unknown_methods = sorted(set(clustering_methods) - set(DEFAULT_CLUSTERING_METHODS))
    if unknown_representations:
        raise ValueError(f"Unknown representations: {', '.join(unknown_representations)}")
    if unknown_methods:
        raise ValueError(f"Unknown clustering methods: {', '.join(unknown_methods)}")

    total_start = time.perf_counter()
    _progress("Week 2 experiment dimulai", verbose=verbose)
    _progress(f"Input: {input_path}", verbose=verbose)
    _progress(f"Output: {output_dir}", verbose=verbose)
    _progress(f"Representasi: {', '.join(representations)}", verbose=verbose)
    _progress(f"Clustering: {', '.join(clustering_methods)}", verbose=verbose)
    _progress(f"Experiment mode: {experiment_mode}", verbose=verbose)
    _progress(f"Cluster count candidates: {cluster_counts}", verbose=verbose)
    _progress(f"Sample size target: {sample_size:,}", verbose=verbose)

    stage_start = time.perf_counter()
    _progress("Profiling CSV dan menghitung distribusi source...", verbose=verbose)
    profile = profile_csv(input_path, min_year=min_year, max_year=max_year)
    _progress(
        f"Profiling selesai dalam {_format_elapsed(time.perf_counter() - stage_start)} "
        f"| valid rows={profile['valid_rows']:,}",
        verbose=verbose,
    )
    quotas = build_stratified_quotas(profile["source_counts_valid"], sample_size)
    stage_start = time.perf_counter()
    _progress("Sampling stratified rows...", verbose=verbose)
    sample_df = sample_stratified_rows(
        input_path,
        quotas,
        seed=seed,
        min_year=min_year,
        max_year=max_year,
    )
    _progress(
        f"Sampling selesai dalam {_format_elapsed(time.perf_counter() - stage_start)} "
        f"| actual sample={len(sample_df):,}",
        verbose=verbose,
    )

    metrics_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    labels_by_combo: dict[tuple[str, str], np.ndarray] = {}
    resolved_sbert_device = resolve_sbert_device(sbert_device) if "sbert" in representations else None
    if resolved_sbert_device:
        _progress(f"SBERT device: {resolved_sbert_device}", verbose=verbose)

    for representation_index, representation in enumerate(representations, start=1):
        _progress(
            f"[{representation_index}/{len(representations)}] Membuat embedding {representation}...",
            verbose=verbose,
        )
        tracemalloc.start()
        embedding_start = time.perf_counter()
        embeddings, cache_info = load_or_build_embeddings(
            sample_df,
            representation,
            input_path=input_path,
            output_dir=output_dir,
            sample_size=sample_size,
            seed=seed,
            min_year=min_year,
            max_year=max_year,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            tfidf_max_features=tfidf_max_features,
            tfidf_min_df=tfidf_min_df,
            tfidf_max_df=tfidf_max_df,
            sbert_model=sbert_model,
            sbert_batch_size=sbert_batch_size,
            sbert_device=resolved_sbert_device or "cpu",
            use_cache=use_embedding_cache,
            verbose=verbose,
        )
        embedding_time_s = time.perf_counter() - embedding_start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        _progress(
            f"Embedding {representation} selesai dalam {_format_elapsed(embedding_time_s)} "
            f"| shape={embeddings.shape[0]:,}x{embeddings.shape[1]:,} "
            f"| peak={peak_memory / (1024 * 1024):.1f} MB",
            verbose=verbose,
        )

        stage_start = time.perf_counter()
        _progress(f"Menyiapkan matrix clustering untuk {representation}...", verbose=verbose)
        matrix, matrix_info = prepare_clustering_matrix(
            embeddings,
            pca_components=pca_components,
            seed=seed,
        )
        _progress(
            f"Matrix siap dalam {_format_elapsed(time.perf_counter() - stage_start)} "
            f"| dims={matrix.shape[1]} | pca={matrix_info['pca_applied']}",
            verbose=verbose,
        )
        embedding_info = {
            "representation": representation,
            "sample_size": int(len(sample_df)),
            "seed": int(seed),
            "embedding_time_s": round(embedding_time_s, 3),
            "peak_memory_mb": round(peak_memory / (1024 * 1024), 3),
            "device": resolved_sbert_device if representation == "sbert" else "cpu",
            **cache_info,
            **matrix_info,
        }
        embedding_rows.append(embedding_info)

        for method_index, method in enumerate(clustering_methods, start=1):
            _progress(
                f"[{representation} {method_index}/{len(clustering_methods)}] Mulai clustering {method}...",
                verbose=verbose,
            )
            best_trial, trials, labels = tune_clustering_method(
                matrix,
                method,
                representation=representation,
                seed=seed,
                metric_sample_size=metric_sample_size,
                cluster_counts=cluster_counts,
                dbscan_min_samples=dbscan_min_samples,
                dbscan_eps_quantiles=dbscan_eps_quantiles,
                hdbscan_min_cluster_sizes=hdbscan_min_cluster_sizes,
                verbose=verbose,
            )
            labels_by_combo[(representation, method)] = labels
            trial_rows.extend(trials)
            metrics_rows.append(
                {
                    **embedding_info,
                    **best_trial,
                }
            )
            _progress(
                f"[{representation}/{method}] best silhouette={best_trial['silhouette']:.4f} "
                f"| clusters={best_trial['n_clusters']} | params={best_trial['params']}",
                verbose=verbose,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    trials_df = pd.DataFrame(trial_rows)
    embeddings_df = pd.DataFrame(embedding_rows)
    cluster_summary_df, source_by_cluster_df = build_cluster_summary(sample_df, labels_by_combo)

    run_config = {
        "input_file": input_path.name,
        "sample": {
            "requested_sample_size": int(sample_size),
            "actual_sample_size": int(len(sample_df)),
            "seed": int(seed),
            "quotas_by_source": quotas,
            "actual_counts_by_source": {
                source: int(count) for source, count in sorted(sample_df["source"].value_counts().items())
            },
        },
        "representations": representations,
        "clustering_methods": clustering_methods,
        "experiment_mode": experiment_mode,
        "embedding_params": {
            "word2vec_doc2vec_vector_size": int(vector_size),
            "window": int(window),
            "min_count": int(min_count),
            "epochs": int(epochs),
            "workers": int(workers),
            "tfidf_max_features": int(tfidf_max_features),
            "tfidf_min_df": int(tfidf_min_df),
            "tfidf_max_df": float(tfidf_max_df),
            "sbert_model": sbert_model,
            "sbert_batch_size": int(sbert_batch_size),
            "sbert_device_requested": sbert_device,
            "sbert_device_resolved": resolved_sbert_device,
            "use_embedding_cache": bool(use_embedding_cache),
        },
        "clustering_grid": {
            "cluster_counts": cluster_counts,
            "dbscan_min_samples": dbscan_min_samples,
            "dbscan_eps_quantiles": dbscan_eps_quantiles,
            "hdbscan_min_cluster_sizes": hdbscan_min_cluster_sizes,
        },
        "pca_components": int(pca_components),
        "metric_sample_size": int(metric_sample_size),
    }

    _progress("Menggabungkan output lama dan menulis file hasil...", verbose=verbose)
    metrics_df = merge_with_existing_output(
        output_dir / "metrics_week2.csv",
        metrics_df,
        key_columns=["representation", "clustering_method", "seed", "sample_size"],
    )
    trials_df = merge_with_existing_output(
        output_dir / "trials_week2.csv",
        trials_df,
        key_columns=["representation", "clustering_method", "seed", "params"],
    )
    embeddings_df = merge_with_existing_output(
        output_dir / "embedding_profile_week2.csv",
        embeddings_df,
        key_columns=["representation", "seed", "sample_size"],
    )
    cluster_summary_df = merge_with_existing_output(
        output_dir / "cluster_summary_week2.csv",
        cluster_summary_df,
        key_columns=["representation", "clustering_method", "cluster_id"],
    )
    source_by_cluster_df = merge_with_existing_output(
        output_dir / "source_by_cluster_week2.csv",
        source_by_cluster_df,
        key_columns=["representation", "clustering_method", "cluster_id", "source"],
    )

    metrics_df = normalize_report_dataframe(metrics_df)
    trials_df = normalize_report_dataframe(trials_df)
    embeddings_df = normalize_report_dataframe(embeddings_df)
    cluster_summary_df = normalize_report_dataframe(cluster_summary_df)
    source_by_cluster_df = normalize_report_dataframe(source_by_cluster_df)

    metrics_df.to_csv(output_dir / "metrics_week2.csv", index=False)
    trials_df.to_csv(output_dir / "trials_week2.csv", index=False)
    embeddings_df.to_csv(output_dir / "embedding_profile_week2.csv", index=False)
    cluster_summary_df.to_csv(output_dir / "cluster_summary_week2.csv", index=False)
    source_by_cluster_df.to_csv(output_dir / "source_by_cluster_week2.csv", index=False)
    with (output_dir / "run_config_week2.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(run_config), handle, indent=2)
        handle.write("\n")
    _progress("Membuat plot Week 2...", verbose=verbose)
    save_week2_plots(metrics_df, output_dir)
    _progress(
        f"Week 2 selesai dalam {_format_elapsed(time.perf_counter() - total_start)}. Output: {output_dir}",
        verbose=verbose,
    )

    return {
        "metrics": metrics_df,
        "trials": trials_df,
        "embedding_profile": embeddings_df,
        "cluster_summary": cluster_summary_df,
        "source_by_cluster": source_by_cluster_df,
        "run_config": run_config,
        "output_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 2 embedding and clustering experiments.")
    parser.add_argument("--input", type=Path, default=Path("timeline.csv"), help="Path to Plaso timeline CSV.")
    parser.add_argument("--output", type=Path, default=Path("reports/week2"), help="Output report directory.")
    parser.add_argument("--sample-size", type=int, default=50_000, help="Valid events to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--representations",
        type=_parse_csv_list,
        default=DEFAULT_REPRESENTATIONS,
        help="Comma-separated representations: tfidf,word2vec,doc2vec,sbert.",
    )
    parser.add_argument(
        "--clustering-methods",
        type=_parse_csv_list,
        default=DEFAULT_CLUSTERING_METHODS,
        help="Comma-separated clustering methods: kmeans,dbscan,hdbscan,agglomerative,gmm.",
    )
    parser.add_argument("--vector-size", type=int, default=100, help="Word2Vec/Doc2Vec vector size.")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec/Doc2Vec context window.")
    parser.add_argument("--min-count", type=int, default=2, help="Word2Vec/Doc2Vec min_count.")
    parser.add_argument("--epochs", type=int, default=20, help="Word2Vec/Doc2Vec training epochs.")
    parser.add_argument("--workers", type=int, default=1, help="Word2Vec/Doc2Vec workers.")
    parser.add_argument("--tfidf-max-features", type=int, default=5000, help="TF-IDF max_features.")
    parser.add_argument("--tfidf-min-df", type=int, default=5, help="TF-IDF min_df.")
    parser.add_argument("--tfidf-max-df", type=float, default=0.95, help="TF-IDF max_df.")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--sbert-batch-size", type=int, default=64, help="SBERT encode batch size.")
    parser.add_argument(
        "--sbert-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device untuk SBERT. auto memakai CUDA jika PyTorch mendeteksinya.",
    )
    parser.add_argument("--pca-components", type=int, default=50, help="PCA dimensions before clustering.")
    parser.add_argument(
        "--experiment-mode",
        choices=["optimal", "k-sensitivity"],
        default="optimal",
        help="optimal mencari k/component terbaik; k-sensitivity memakai 10,20,50.",
    )
    parser.add_argument(
        "--cluster-counts",
        type=_parse_int_list,
        default=None,
        help="Override Agglomerative/GMM k grid. Jika kosong, mengikuti experiment mode.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=_parse_int_list,
        default=[5, 10],
        help="Comma-separated DBSCAN min_samples grid.",
    )
    parser.add_argument(
        "--dbscan-eps-quantiles",
        type=_parse_float_list,
        default=[0.8, 0.9, 0.95],
        help="Quantiles from k-distance curve used as DBSCAN eps candidates.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-sizes",
        type=_parse_int_list,
        default=[15, 30, 50],
        help="Comma-separated HDBSCAN min_cluster_size grid.",
    )
    parser.add_argument("--metric-sample-size", type=int, default=10_000, help="Silhouette sample size.")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR, help="Minimum accepted event year.")
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR, help="Maximum accepted event year.")
    parser.add_argument("--no-embedding-cache", action="store_true", help="Recompute embeddings and skip cache files.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_week2(
        input_path=args.input,
        output_dir=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        representations=args.representations,
        clustering_methods=args.clustering_methods,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_max_df=args.tfidf_max_df,
        sbert_model=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
        sbert_device=args.sbert_device,
        pca_components=args.pca_components,
        experiment_mode=args.experiment_mode,
        cluster_counts=args.cluster_counts,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_eps_quantiles=args.dbscan_eps_quantiles,
        hdbscan_min_cluster_sizes=args.hdbscan_min_cluster_sizes,
        metric_sample_size=args.metric_sample_size,
        min_year=args.min_year,
        max_year=args.max_year,
        use_embedding_cache=not args.no_embedding_cache,
        verbose=not args.quiet,
    )
    print(f"Week 2 experiments complete. Output: {result['output_dir']}")
    print(result["metrics"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
