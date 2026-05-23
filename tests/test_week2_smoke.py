import csv
from pathlib import Path

import pandas as pd

from log_clustering.week1 import REQUIRED_COLUMNS
from log_clustering.week2 import resolve_sbert_device, run_week2


def _write_tiny_timeline(path: Path, rows: int = 120) -> None:
    sources = ["FILE", "REG", "EVT", "WEBHIST"]
    messages = {
        "FILE": r"C:\Users\User\Downloads\invoice.pdf Type: file created",
        "REG": r"HKEY_CURRENT_USER\Software\Microsoft\Windows Run key updated",
        "EVT": "Windows event log service started event id 4624",
        "WEBHIST": "https://example.com/download payload visited in browser",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for index in range(rows):
            source = sources[index % len(sources)]
            writer.writerow(
                {
                    "datetime": f"2023-02-{(index % 28) + 1:02d}T11:00:00.000000+00:00",
                    "timestamp_desc": "Content Modification Time",
                    "source": source,
                    "source_long": source,
                    "message": f"{messages[source]} sample {index}",
                    "parser": "test/parser",
                    "display_name": r"NTFS:\sample",
                    "tag": "-",
                }
            )


def test_week2_pipeline_smoke(tmp_path):
    input_csv = tmp_path / "timeline.csv"
    output_dir = tmp_path / "reports"
    _write_tiny_timeline(input_csv)

    result = run_week2(
        input_path=input_csv,
        output_dir=output_dir,
        sample_size=80,
        seed=42,
        representations=["tfidf", "word2vec"],
        clustering_methods=["kmeans", "dbscan", "hdbscan", "agglomerative", "gmm"],
        vector_size=12,
        min_count=1,
        epochs=5,
        pca_components=5,
        cluster_counts=[2, 3],
        dbscan_min_samples=[3],
        dbscan_eps_quantiles=[0.5, 0.8],
        hdbscan_min_cluster_sizes=[5, 10],
        metric_sample_size=40,
        verbose=False,
    )

    expected_files = [
        "metrics_week2.csv",
        "trials_week2.csv",
        "embedding_profile_week2.csv",
        "cluster_summary_week2.csv",
        "source_by_cluster_week2.csv",
        "run_config_week2.json",
        "silhouette_heatmap_week2.png",
        "quality_runtime_week2.png",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0

    metrics = pd.read_csv(output_dir / "metrics_week2.csv")
    assert set(metrics["representation"]) == {"tfidf", "word2vec"}
    assert set(metrics["clustering_method"]) == {"kmeans", "dbscan", "hdbscan", "agglomerative", "gmm"}
    assert result["run_config"]["sample"]["actual_sample_size"] == 80


def test_resolve_sbert_device_accepts_cpu():
    assert resolve_sbert_device("cpu") == "cpu"
