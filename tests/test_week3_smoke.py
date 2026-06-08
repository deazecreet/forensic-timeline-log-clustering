import csv
from pathlib import Path

import pandas as pd

from log_clustering.week1 import REQUIRED_COLUMNS
from log_clustering.week2 import run_week2
from log_clustering.week3 import run_week3


def _write_tiny_timeline(path: Path, rows: int = 90) -> None:
    sources = ["FILE", "REG", "EVT"]
    messages = {
        "FILE": r"C:\Users\User\Downloads\invoice.pdf Type: file created",
        "REG": r"HKEY_CURRENT_USER\Software\Microsoft\Windows Run key updated",
        "EVT": "Windows event log service started event id 4624",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for index in range(rows):
            source = sources[index % len(sources)]
            writer.writerow(
                {
                    "datetime": f"2023-03-{(index % 28) + 1:02d}T10:00:00.000000+00:00",
                    "timestamp_desc": "Content Modification Time",
                    "source": source,
                    "source_long": source,
                    "message": f"{messages[source]} sample {index}",
                    "parser": "test/parser",
                    "display_name": r"NTFS:\sample",
                    "tag": "-",
                }
            )


def test_week3_pipeline_smoke(tmp_path):
    input_csv = tmp_path / "timeline.csv"
    week2_dir = tmp_path / "week2"
    week3_dir = tmp_path / "week3"
    _write_tiny_timeline(input_csv)

    run_week2(
        input_path=input_csv,
        output_dir=week2_dir,
        sample_size=60,
        seed=42,
        representations=["tfidf"],
        clustering_methods=["kmeans"],
        pca_components=5,
        cluster_counts=[2, 3],
        metric_sample_size=40,
        verbose=False,
    )
    result = run_week3(
        input_path=input_csv,
        week2_dir=week2_dir,
        output_dir=week3_dir,
        representations=["tfidf"],
        clustering_methods=["kmeans"],
        scatter_methods=["pca"],
        max_viz_points=40,
        max_timeline_points=40,
        max_dendrogram_points=20,
        verbose=False,
    )

    expected_files = [
        "WEEK3_PROGRESS.md",
        "interpretability_assessment_week3.csv",
        "cluster_interpretability_week3.csv",
        "sample_metadata_week3.csv",
        "labels/tfidf_kmeans_labels.csv",
        "scatter/pca/tfidf_kmeans_pca.png",
        "scatter_grids/pca_grid.png",
        "scatter_grid_index_week3.csv",
        "visualization_times_week3.csv",
        "timelines/tfidf_kmeans_timeline.png",
        "source_distribution/tfidf_kmeans_source.png",
        "top_terms/tfidf_kmeans_top_terms.png",
    ]
    for filename in expected_files:
        path = week3_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0

    interpretability = pd.read_csv(week3_dir / "interpretability_assessment_week3.csv")
    assert set(interpretability["representation"]) == {"tfidf"}
    assert set(interpretability["clustering_method"]) == {"kmeans"}
    visualization_times = pd.read_csv(week3_dir / "visualization_times_week3.csv")
    assert not visualization_times.empty
    assert set(visualization_times["visualization_method"]) == {"pca"}
    assert (visualization_times["status"] == "ok").any()
    scatter_grid_index = pd.read_csv(week3_dir / "scatter_grid_index_week3.csv")
    assert set(scatter_grid_index["status"]) == {"ok"}
    assert result["run_summary"]["generated_file_count"] >= len(expected_files)
