import csv
from pathlib import Path

import pandas as pd

from log_clustering.week1 import REQUIRED_COLUMNS
from log_clustering.week2 import run_week2
from log_clustering.week4 import generate_pdf_literal_completion, generate_week4_visual_polish, run_week4


def _write_tiny_timeline(path: Path, rows: int = 120, scenario: str = "primary") -> None:
    sources = ["FILE", "REG", "EVT", "WEBHIST"]
    messages = {
        "FILE": rf"C:\Users\User\Downloads\{scenario}_invoice.pdf Type: file created",
        "REG": rf"HKEY_CURRENT_USER\Software\{scenario}\Run key updated",
        "EVT": f"Windows event log service started {scenario} event id 4624",
        "WEBHIST": f"https://example.com/{scenario}/payload visited in browser",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for index in range(rows):
            source = sources[index % len(sources)]
            writer.writerow(
                {
                    "datetime": f"2023-04-{(index % 28) + 1:02d}T09:00:00.000000+00:00",
                    "timestamp_desc": "Content Modification Time",
                    "source": source,
                    "source_long": source,
                    "message": f"{messages[source]} sample {index}",
                    "parser": "test/parser",
                    "display_name": r"NTFS:\sample",
                    "tag": "-",
                }
            )


def test_week4_final_smoke(tmp_path):
    primary_csv = tmp_path / "primary.csv"
    secondary_csv = tmp_path / "secondary.csv"
    primary_runs_root = tmp_path / "week4_primary_runs"
    week4_dir = tmp_path / "week4"
    _write_tiny_timeline(primary_csv, scenario="primary")
    _write_tiny_timeline(secondary_csv, scenario="secondary")

    seeds = [42, 123]
    for seed in seeds:
        run_week2(
            input_path=primary_csv,
            output_dir=primary_runs_root / f"primary_optimal_seed{seed}",
            sample_size=80,
            seed=seed,
            representations=["tfidf"],
            clustering_methods=["kmeans", "agglomerative"],
            pca_components=5,
            cluster_counts=[2, 3],
            metric_sample_size=40,
            verbose=False,
        )

    result = run_week4(
        primary_input=primary_csv,
        secondary_input=secondary_csv,
        primary_runs_root=primary_runs_root,
        output_dir=week4_dir,
        week3_dir=tmp_path / "missing_week3",
        seeds=seeds,
        sample_size=80,
        top_representations=1,
        representations=["tfidf"],
        clustering_methods=["kmeans", "agglomerative"],
        metric_sample_size=40,
        verbose=False,
    )

    expected_files = [
        "metrics_primary_runs.csv",
        "metrics_primary_summary.csv",
        "metrics_k_sensitivity.csv",
        "metrics_cross_dataset.csv",
        "cross_dataset_comparison.csv",
        "best_methods_week4.csv",
        "run_config_week4.json",
        "WEEK4_FINAL_REPORT.md",
        "plots/primary_silhouette_mean.png",
        "plots/k_sensitivity_silhouette_mean.png",
        "plots/cross_dataset_silhouette.png",
        "cd_friedman_nemenyi_week4.csv",
        "multi_metric_scores_week4.csv",
        "pareto_front_week4.csv",
        "visualization_times_week4.csv",
        "wordcloud_index_week4.csv",
        "plots/critical_difference_primary_silhouette.png",
        "plots/radar_multi_metric_top5.png",
        "plots/pareto_silhouette_runtime.png",
    ]
    for filename in expected_files:
        path = week4_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0

    primary_runs = pd.read_csv(week4_dir / "metrics_primary_runs.csv")
    assert len(primary_runs) == len(seeds) * 1 * 2

    primary_summary = pd.read_csv(week4_dir / "metrics_primary_summary.csv")
    assert set(primary_summary["seed_count"]) == {len(seeds)}

    sensitivity = pd.read_csv(week4_dir / "metrics_k_sensitivity.csv")
    assert set(sensitivity["representation"]) == {"tfidf"}
    assert set(sensitivity["clustering_method"]) == {"kmeans", "agglomerative"}

    comparison = pd.read_csv(week4_dir / "cross_dataset_comparison.csv")
    assert not comparison.empty
    assert result["run_config"]["best_representations"] == ["tfidf"]
    report_text = (week4_dir / "WEEK4_FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "Week 4 Pilot Report" not in report_text
    assert "Literal PDF Visualization Polish" in report_text


def test_week4_visual_polish_wordcloud(tmp_path):
    week4_dir = tmp_path / "week4"
    week3_dir = tmp_path / "week3"
    week4_dir.mkdir()
    week3_dir.mkdir()

    primary_runs = pd.DataFrame(
        [
            {
                "seed": seed,
                "representation": representation,
                "clustering_method": method,
                "silhouette": silhouette,
            }
            for seed in [42, 123, 456]
            for representation, method, silhouette in [
                ("tfidf", "hdbscan", 0.91 + seed / 100000),
                ("tfidf", "dbscan", 0.88 + seed / 100000),
                ("word2vec", "hdbscan", 0.82 + seed / 100000),
            ]
        ]
    )
    primary_runs.to_csv(week4_dir / "metrics_primary_runs.csv", index=False)

    primary_summary = pd.DataFrame(
        [
            {
                "dataset": "scenario-1",
                "experiment": "optimal",
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "seed_count": 3,
                "sample_size": 80,
                "silhouette_mean": 0.92,
                "calinski_harabasz_mean": 100.0,
                "davies_bouldin_mean": 0.2,
                "n_clusters_mean": 4,
                "noise_ratio_mean": 0.05,
                "embedding_time_s_mean": 1.0,
                "clustering_time_s_mean": 2.0,
                "peak_memory_mb_mean": 40.0,
            },
            {
                "dataset": "scenario-1",
                "experiment": "optimal",
                "representation": "tfidf",
                "clustering_method": "dbscan",
                "seed_count": 3,
                "sample_size": 80,
                "silhouette_mean": 0.88,
                "calinski_harabasz_mean": 80.0,
                "davies_bouldin_mean": 0.3,
                "n_clusters_mean": 5,
                "noise_ratio_mean": 0.1,
                "embedding_time_s_mean": 1.0,
                "clustering_time_s_mean": 1.0,
                "peak_memory_mb_mean": 35.0,
            },
            {
                "dataset": "scenario-1",
                "experiment": "optimal",
                "representation": "word2vec",
                "clustering_method": "hdbscan",
                "seed_count": 3,
                "sample_size": 80,
                "silhouette_mean": 0.82,
                "calinski_harabasz_mean": 70.0,
                "davies_bouldin_mean": 0.4,
                "n_clusters_mean": 3,
                "noise_ratio_mean": 0.2,
                "embedding_time_s_mean": 2.0,
                "clustering_time_s_mean": 2.5,
                "peak_memory_mb_mean": 55.0,
            },
        ]
    )
    primary_summary.to_csv(week4_dir / "metrics_primary_summary.csv", index=False)
    primary_summary.head(1).to_csv(week4_dir / "metrics_k_sensitivity_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "primary_silhouette_mean": 0.92,
                "secondary_silhouette_mean": 0.90,
            }
        ]
    ).to_csv(week4_dir / "cross_dataset_comparison.csv", index=False)

    pd.DataFrame(
        [
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "cluster_id": 1,
                "size": 50,
                "is_noise": False,
                "top_tokens": "path_token; registry; malware; startup; runkey",
            },
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "cluster_id": 2,
                "size": 30,
                "is_noise": False,
                "top_tokens": "browser; history; url_token; download; payload",
            },
        ]
    ).to_csv(week3_dir / "cluster_interpretability_week3.csv", index=False)

    result = generate_week4_visual_polish(output_dir=week4_dir, week3_dir=week3_dir, verbose=False)

    wordcloud_index = pd.read_csv(week4_dir / "wordcloud_index_week4.csv")
    assert not wordcloud_index.empty
    assert (wordcloud_index["status"] == "ok").any()
    assert (week4_dir / "plots" / "wordclouds" / "tfidf_hdbscan_aggregate.png").exists()
    assert (week4_dir / "plots" / "wordclouds" / "tfidf_hdbscan_aggregate.png").stat().st_size > 0
    assert not result["visualization_times"].empty


def test_week4_pdf_literal_completion(tmp_path):
    week4_dir = tmp_path / "week4"
    week3_dir = tmp_path / "week3"
    (week4_dir / "primary_k_sensitivity_seed42").mkdir(parents=True)
    (week4_dir / "primary_optimal_seed42").mkdir(parents=True)
    (week3_dir / "labels").mkdir(parents=True)
    week4_dir.mkdir(exist_ok=True)

    primary_summary = pd.DataFrame(
        [
            {
                "dataset": "scenario-1",
                "experiment": "optimal",
                "representation": "tfidf",
                "clustering_method": "kmeans",
                "seed_count": 1,
                "sample_size": 40,
                "silhouette_mean": 0.71,
                "calinski_harabasz_mean": 50.0,
                "davies_bouldin_mean": 0.5,
                "n_clusters_mean": 2,
                "noise_ratio_mean": 0.0,
                "embedding_time_s_mean": 0.2,
                "clustering_time_s_mean": 0.1,
                "peak_memory_mb_mean": 10.0,
            },
            {
                "dataset": "scenario-1",
                "experiment": "optimal",
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "seed_count": 1,
                "sample_size": 40,
                "silhouette_mean": 0.82,
                "calinski_harabasz_mean": 60.0,
                "davies_bouldin_mean": 0.4,
                "n_clusters_mean": 2,
                "noise_ratio_mean": 0.05,
                "embedding_time_s_mean": 0.2,
                "clustering_time_s_mean": 0.2,
                "peak_memory_mb_mean": 11.0,
            },
        ]
    )
    primary_summary.to_csv(week4_dir / "metrics_primary_summary.csv", index=False)
    pd.DataFrame(
        [
            {"seed": 42, "representation": "tfidf", "clustering_method": "kmeans", "silhouette": 0.71},
            {"seed": 42, "representation": "tfidf", "clustering_method": "hdbscan", "silhouette": 0.82},
        ]
    ).to_csv(week4_dir / "metrics_primary_runs.csv", index=False)
    pd.DataFrame(
        [
            {"representation": "tfidf", "clustering_method": "hdbscan", "primary_silhouette_mean": 0.82, "secondary_silhouette_mean": 0.8}
        ]
    ).to_csv(week4_dir / "cross_dataset_comparison.csv", index=False)
    (week4_dir / "WEEK4_FINAL_REPORT.md").write_text("# Week 4 Final Report\n", encoding="utf-8")

    trial_rows = []
    for k in [10, 20, 50]:
        trial_rows.append(
            {
                "representation": "tfidf",
                "clustering_method": "kmeans",
                "seed": 42,
                "params": f'{{"n_clusters": {k}}}',
                "silhouette": 0.5 + k / 1000,
                "davies_bouldin": 0.9,
                "n_clusters": k,
                "noise_ratio": 0.0,
                "clustering_time_s": 0.1,
                "inertia": 1000 / k,
            }
        )
    pd.DataFrame(trial_rows).to_csv(week4_dir / "primary_k_sensitivity_seed42" / "trials_week2.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(week4_dir / "primary_optimal_seed42" / "trials_week2.csv", index=False)

    pd.DataFrame(
        [
            {"sample_index": 0, "clean_text": "source_file path_token invoice download"},
            {"sample_index": 1, "clean_text": "source_file path_token invoice created"},
            {"sample_index": 2, "clean_text": "source_reg runkey registry startup"},
            {"sample_index": 3, "clean_text": "source_reg registry startup runkey"},
            {"sample_index": 4, "clean_text": "source_evt event login session"},
            {"sample_index": 5, "clean_text": "source_evt event service started"},
        ]
    ).to_csv(week3_dir / "sample_metadata_week3.csv", index=False)
    pd.DataFrame(
        [
            {"sample_index": 0, "cluster_id": 1},
            {"sample_index": 1, "cluster_id": 1},
            {"sample_index": 2, "cluster_id": 1},
            {"sample_index": 3, "cluster_id": 1},
            {"sample_index": 4, "cluster_id": 1},
            {"sample_index": 5, "cluster_id": 1},
        ]
    ).to_csv(week3_dir / "labels" / "tfidf_hdbscan_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "representation": "tfidf",
                "clustering_method": "hdbscan",
                "cluster_id": 1,
                "size": 10,
                "is_noise": False,
                "forensic_cluster_score": 5,
                "top_tokens": "path_token; invoice; download",
                "top_sources": "FILE:10",
                "example_messages": "invoice created",
            }
        ]
    ).to_csv(week3_dir / "cluster_interpretability_week3.csv", index=False)
    pd.DataFrame(
        [
            {
                "representation": "tfidf",
                "clustering_method": "kmeans",
                "visualization_method": "pca",
                "stage": "reduction",
                "elapsed_s": 0.01,
                "output_path": "x",
                "status": "ok",
            }
        ]
    ).to_csv(week3_dir / "visualization_times_week3.csv", index=False)
    pd.DataFrame([{"visualization_method": "pca", "plot_count": 1, "output_path": "x", "status": "ok"}]).to_csv(
        week3_dir / "scatter_grid_index_week3.csv",
        index=False,
    )

    result = generate_pdf_literal_completion(output_dir=week4_dir, week3_dir=week3_dir, verbose=False)

    expected_files = [
        "PDF_REQUIREMENT_AUDIT.md",
        "pdf_requirement_traceability_week4.csv",
        "k_sensitivity_trials_week4.csv",
        "k_sensitivity_by_k_week4.csv",
        "k_optimal_diagnostics_week4.csv",
        "primary_silhouette_heatmap_mean.csv",
        "manual_interpretability_review_week4.csv",
        "wordcloud_full_index_week4.csv",
        "plots/k_sensitivity_by_k_silhouette.png",
        "plots/k_diagnostics_tfidf.png",
        "plots/primary_silhouette_heatmap_mean.png",
    ]
    for filename in expected_files:
        path = week4_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0
    assert "Final Literal PDF Completion" in (week4_dir / "WEEK4_FINAL_REPORT.md").read_text(encoding="utf-8")
    assert not result["traceability"].empty
