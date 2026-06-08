# Log Clustering for Forensic Timeline Analysis

Repository ini berisi implementasi mini project **Log Clustering for Forensic Timeline Analysis** pada output CSV log2timeline/Plaso. Saat ini repository sudah mencakup pipeline Pekan 1, Pekan 2, Pekan 3, dan finalisasi Pekan 4.

## Dataset

Dataset yang digunakan adalah `scenario-1` dari Zenodo:

- Judul: [Digital forensic timeline dataset extracted using log2timeline/Plaso](https://zenodo.org/records/15493424)
- DOI: `10.5281/zenodo.15493424`
- Lisensi: MIT License
- File lokal yang dipakai: `timeline.csv`

`timeline.csv` tidak dimasukkan ke Git karena ukurannya besar. Letakkan file tersebut di root repository sebelum menjalankan pipeline.
Dataset kedua untuk cross-dataset validation diletakkan secara lokal di `data/external/timeline.csv`; folder `data/external/` juga di-ignore dari Git.

## Pekan 1

Target Pekan 1:

- profiling dataset Plaso CSV
- preprocessing pesan log
- sampling stratified 50.000 event valid berdasarkan `source`
- baseline TF-IDF
- reduksi dimensi 50D dengan TruncatedSVD
- baseline K-Means untuk `k=10,20,50`
- export metrik, ringkasan cluster, dan plot kecil

Catatan progress detail tersedia secara lokal di `week_progress/`, tetapi folder tersebut di-ignore dari Git.

## Pekan 2

Target Pekan 2:

- representasi teks dense dengan TF-IDF, Word2Vec, Doc2Vec, dan Sentence-BERT
- reduksi dimensi PCA 50D sebelum clustering
- clustering dengan K-Means, DBSCAN, HDBSCAN, Agglomerative Clustering, dan GMM
- tuning parameter sederhana per algoritma
- evaluasi internal dengan Silhouette, Calinski-Harabasz, Davies-Bouldin, jumlah cluster, noise ratio, waktu komputasi, dan peak memory
- export metrik, seluruh trial tuning, ringkasan cluster, dan plot perbandingan awal

Catatan progress detail tersedia secara lokal di `week_progress/`, tetapi folder tersebut di-ignore dari Git.

## Pekan 3

Target Pekan 3:

- rekonstruksi label cluster terbaik dari hasil Pekan 2
- visualisasi PCA, UMAP, dan t-SNE untuk setiap kombinasi representasi dan clustering
- dendrogram untuk Agglomerative Clustering
- visualisasi timeline event terhadap cluster
- visualisasi distribusi source dan top terms untuk membaca makna cluster
- interpretability assessment untuk menilai apakah cluster bermakna secara forensik
- laporan Markdown yang merangkum hasil visualisasi, analisis komparatif, batasan, dan rekomendasi inspeksi

Catatan progress detail tersedia secara lokal di `week_progress/`, tetapi folder tersebut di-ignore dari Git.

## Status Scope

| Scope | Status | Output Utama |
| --- | --- | --- |
| Pekan 1 | Selesai | `reports/week1/` |
| Pekan 2 | Selesai | `reports/week2/` |
| Pekan 3 | Selesai | `reports/week3/` |
| Pekan 4 | Selesai | `reports/week4/` |

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Baseline

```bash
python -m log_clustering.week1 --input timeline.csv --sample-size 50000 --seed 42 --k-values 10,20,50 --output reports/week1
```

Output utama:

- `reports/week1/dataset_profile.json`
- `reports/week1/metrics_week1.csv`
- `reports/week1/cluster_summary_week1.csv`
- `reports/week1/source_by_cluster.csv`
- `reports/week1/elbow_inertia.png`
- `reports/week1/silhouette_scores.png`
- `reports/week1/source_distribution_sample.png`

## Menjalankan Eksperimen Pekan 2

Run cepat tanpa SBERT, cocok untuk validasi lokal:

```bash
python -m log_clustering.week2 --input timeline.csv --sample-size 50000 --representations tfidf,word2vec,doc2vec --clustering-methods kmeans,dbscan,hdbscan,agglomerative,gmm --output reports/week2_quick
```

Run sesuai scope Pekan 2 yang diminta:

```bash
python -m log_clustering.week2 --input timeline.csv --sample-size 50000 --representations tfidf,word2vec,doc2vec,sbert --clustering-methods kmeans,dbscan,hdbscan,agglomerative,gmm --experiment-mode optimal --sbert-device auto --seed 42 --output reports/week2
```

Eksperimen juga bisa dijalankan bertahap ke folder output yang sama. Output Week 2 akan otomatis digabung dan plot perbandingan akan dibuat ulang dari hasil gabungan:

```bash
python -m log_clustering.week2 --input timeline.csv --sample-size 50000 --representations tfidf --clustering-methods kmeans,dbscan,hdbscan,agglomerative,gmm --seed 42 --output reports/week2
python -m log_clustering.week2 --input timeline.csv --sample-size 50000 --representations word2vec --clustering-methods kmeans,dbscan,hdbscan,agglomerative,gmm --seed 42 --output reports/week2
python -m log_clustering.week2 --input timeline.csv --sample-size 50000 --representations doc2vec --clustering-methods kmeans,dbscan,hdbscan,agglomerative,gmm --seed 42 --output reports/week2
python -m log_clustering.week2 --input timeline.csv --sample-size 50000 --representations sbert --clustering-methods kmeans,dbscan,hdbscan,agglomerative,gmm --sbert-device auto --seed 42 --output reports/week2
```

Embedding otomatis disimpan sebagai cache di `reports/week2/cache/`. Jika representasi, dataset, sample size, seed, dan parameter embedding sama, run berikutnya akan memakai cache tersebut sehingga bisa menjalankan clustering tambahan tanpa menghitung ulang Word2Vec, Doc2Vec, atau SBERT. Gunakan `--no-embedding-cache` jika ingin memaksa embedding dihitung ulang.

Mode default `--experiment-mode optimal` mencari parameter terbaik untuk skenario inti. Untuk Agglomerative/GMM, kandidat default-nya adalah `2,3,4,5,8,10,15,20,30,40,50,75,100`. Mode `--experiment-mode k-sensitivity` disiapkan untuk eksperimen tambahan nanti dan memakai kandidat `10,20,50`.

Output utama:

- `reports/week2/metrics_week2.csv`
- `reports/week2/trials_week2.csv`
- `reports/week2/embedding_profile_week2.csv`
- `reports/week2/cluster_summary_week2.csv`
- `reports/week2/source_by_cluster_week2.csv`
- `reports/week2/run_config_week2.json`
- `reports/week2/silhouette_heatmap_week2.png`
- `reports/week2/quality_runtime_week2.png`

## Menjalankan Analisis Pekan 3

Pipeline Pekan 3 memakai output Pekan 2 sebagai input. Jalankan setelah `reports/week2` berisi metrik, ringkasan cluster, source distribution, dan cache embedding:

```bash
python -m log_clustering.week3 --input timeline.csv --week2-dir reports/week2 --output reports/week3
```

Output utama:

- `reports/week3/interpretability_assessment_week3.csv`
- `reports/week3/cluster_interpretability_week3.csv`
- `reports/week3/scatter/pca/*.png`
- `reports/week3/scatter/umap/*.png`
- `reports/week3/scatter/tsne/*.png`
- `reports/week3/dendrograms/*.png`
- `reports/week3/timelines/*.png`
- `reports/week3/source_distribution/*.png`
- `reports/week3/top_terms/*.png`
- `reports/week3/scatter_grids/*.png`
- `reports/week3/visualization_times_week3.csv`

Artefak besar/regenerable seperti `reports/week3/sample_metadata_week3.csv` dan `reports/week3/labels/*.csv` dibuat saat pipeline berjalan, tetapi di-ignore dari Git agar repository tetap ringan.

Catatan: SBERT memakai model `all-MiniLM-L6-v2` dan akan mengunduh model dari Hugging Face pada run pertama jika belum ada di cache lokal. Opsi `--sbert-device auto` akan memakai GPU CUDA jika PyTorch mendeteksinya; gunakan `--sbert-device cuda` jika ingin memaksa GPU dan gagal cepat ketika CUDA belum aktif.

Jika PC memiliki NVIDIA GPU tetapi `torch.cuda.is_available()` masih `False`, install PyTorch CUDA build terlebih dahulu. Contoh untuk CUDA 12.8:

```bash
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Menjalankan Finalisasi Pekan 4

Pipeline Pekan 4 mengagregasi lima primary optimal seed, memilih dua representasi terbaik untuk k-sensitivity, menjalankan best quality pipeline pada dataset kedua, dan menulis artefak laporan final. Best quality pipeline dipilih dari `silhouette_mean` tertinggi pada `reports/week4/metrics_primary_summary.csv`; skor multi-metrik di `reports/week4/multi_metric_scores_week4.csv` dipakai sebagai analisis balanced quality-efficiency, bukan sebagai kriteria utama cross-dataset validation.

```bash
python -m log_clustering.week4 --primary-input timeline.csv --secondary-input data/external/timeline.csv --output reports/week4 --seeds 42,123,456,789,101 --sbert-device auto
```

Jika eksperimen final sudah selesai dan hanya ingin membuat ulang visual polish literal PDF tanpa mengulang run berat:

```bash
python -m log_clustering.week4 --polish-only --output reports/week4 --week3-dir reports/week3
```

Untuk menutup audit literal PDF terakhir dari artefak yang sudah ada:

```bash
python -m log_clustering.week4 --literal-pdf-completion --output reports/week4 --week3-dir reports/week3
```

Output utama:

- `reports/week4/metrics_primary_runs.csv`
- `reports/week4/metrics_primary_summary.csv`
- `reports/week4/metrics_k_sensitivity.csv`
- `reports/week4/metrics_k_sensitivity_summary.csv`
- `reports/week4/metrics_cross_dataset.csv`
- `reports/week4/cross_dataset_comparison.csv`
- `reports/week4/best_methods_week4.csv`
- `reports/week4/run_config_week4.json`
- `reports/week4/PDF_REQUIREMENT_AUDIT.md`
- `reports/week4/pdf_requirement_traceability_week4.csv`
- `notebooks/final_result.ipynb`

Notebook final `notebooks/final_result.ipynb` adalah notebook ringan untuk menyusun tabel bagian 4.4, performa komputasi 4.4b, visualisasi 4.5, dan traceability requirement PDF dari artefak final. Output cell notebook sengaja dikosongkan agar aman untuk GitHub; jalankan notebook untuk merender ulang tabel dan gambar.

Polish literal PDF:

- Critical Difference/Friedman-Nemenyi: `reports/week4/cd_friedman_nemenyi_week4.csv`, `reports/week4/plots/critical_difference_primary_silhouette.png`
- Radar/spider chart multi-metrik: `reports/week4/multi_metric_scores_week4.csv`, `reports/week4/plots/radar_multi_metric_top5.png`
- Pareto front Silhouette vs computation time: `reports/week4/pareto_front_week4.csv`, `reports/week4/plots/pareto_silhouette_runtime.png`
- Word cloud literal best pipeline `tfidf + hdbscan`: `reports/week4/wordcloud_index_week4.csv`, `reports/week4/plots/wordclouds/*.png`
- Visualization time metric: `reports/week4/visualization_times_week4.csv`

Completion literal PDF:

- Traceability audit: `reports/week4/PDF_REQUIREMENT_AUDIT.md`, `reports/week4/pdf_requirement_traceability_week4.csv`
- Visualization time PCA/UMAP/t-SNE: `reports/week3/visualization_times_week3.csv`
- Scatter montage/grid: `reports/week3/scatter_grids/*.png`, `reports/week3/scatter_grid_index_week3.csv`
- K-sensitivity k=10/20/50 summary: `reports/week4/k_sensitivity_trials_week4.csv`, `reports/week4/k_sensitivity_by_k_week4.csv`
- K-optimal diagnostic: `reports/week4/k_optimal_diagnostics_week4.csv`, `reports/week4/plots/k_diagnostics_*.png`
- Final mean heatmap: `reports/week4/primary_silhouette_heatmap_mean.csv`, `reports/week4/plots/primary_silhouette_heatmap_mean.png`
- Full best-pipeline word cloud: `reports/week4/wordcloud_full_index_week4.csv`; PNG lengkap di `reports/week4/plots/wordclouds_full/*.png` bersifat lokal/regenerable dan di-ignore dari Git
- Interpretability review-ready artifact: `reports/week4/manual_interpretability_review_week4.csv`

Artefak kerja Pekan 4 seperti `reports/week4/primary_*_seed*/`, `reports/week4/secondary_best_seed*/`, cache embedding, log, dan `reports/week4_pilot/` sengaja di-ignore. Artefak tersebut bisa dibuat ulang dari pipeline dan tidak perlu ikut commit.

## Preprocessing

Pipeline memakai kolom `message` sebagai teks utama dan `source` sebagai metadata tambahan. Teks input TF-IDF dibentuk sebagai:

```text
source_<source> + cleaned_message
```

Normalisasi yang dilakukan:

- lowercase
- URL menjadi `url_token`
- path menjadi `path_token`
- timestamp menjadi `time_token`
- hex panjang menjadi `hex_token`
- angka menjadi `num_token`
- tokenisasi regex
- stopword English dari scikit-learn + stopword forensik custom

## Testing

```bash
pytest
```

Test mencakup preprocessing dan smoke test pipeline dengan CSV mini.

## Hasil Pekan 1

Dataset lokal berisi 2.233.799 event. Dengan filter tahun `2000-2026` dan pesan tidak kosong, terdapat 2.175.779 event valid. Pipeline memakai stratified sample 50.000 event untuk baseline awal.

Distribusi sample:

| Source | Jumlah |
| --- | ---: |
| FILE | 68.591 |
| REG | 26.963 |
| EVT | 2.741 |
| PE | 1.494 |
| WEBHIST | 80 |
| OLECF | 78 |
| LOG | 27 |
| AMCACHE | 18 |
| LNK | 8 |

Ringkasan baseline TF-IDF + K-Means:

| k | Silhouette | Calinski-Harabasz | Davies-Bouldin | Inertia |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.5586 | 24013.23 | 1.5412 | 24232.81 |
| 20 | 0.6312 | 24522.99 | 1.3693 | 13534.56 |
| 50 | 0.7261 | 26010.45 | 0.8435 | 5571.01 |

Pada baseline Pekan 1, `k=50` memberi Silhouette tertinggi dan Davies-Bouldin terendah dari tiga nilai `k` yang diuji. Detail lengkap tersedia di `reports/week1/metrics_week1.csv` dan `reports/week1/cluster_summary_week1.csv`.


