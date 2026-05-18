import csv
from pathlib import Path

import pandas as pd

from log_clustering.week1 import REQUIRED_COLUMNS, run_week1


def _write_tiny_timeline(path: Path, rows: int = 180) -> None:
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
                    "datetime": f"2023-01-{(index % 28) + 1:02d}T10:00:00.000000+00:00",
                    "timestamp_desc": "Content Modification Time",
                    "source": source,
                    "source_long": source,
                    "message": f"{messages[source]} sample {index}",
                    "parser": "test/parser",
                    "display_name": r"NTFS:\sample",
                    "tag": "-",
                }
            )


def test_week1_pipeline_smoke(tmp_path):
    input_csv = tmp_path / "timeline.csv"
    output_dir = tmp_path / "reports"
    _write_tiny_timeline(input_csv)

    result = run_week1(
        input_path=input_csv,
        output_dir=output_dir,
        sample_size=80,
        seed=42,
        k_values=[2, 3],
        max_features=200,
        svd_components=5,
        min_df=1,
        metric_sample_size=40,
    )

    expected_files = [
        "metrics_week1.csv",
        "cluster_summary_week1.csv",
        "source_by_cluster.csv",
        "dataset_profile.json",
        "elbow_inertia.png",
        "silhouette_scores.png",
        "source_distribution_sample.png",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()
        assert (output_dir / filename).stat().st_size > 0

    metrics = pd.read_csv(output_dir / "metrics_week1.csv")
    assert set(metrics["k"]) == {2, 3}
    assert metrics["sample_size"].eq(80).all()
    assert result["profile"]["sample"]["actual_sample_size"] == 80
