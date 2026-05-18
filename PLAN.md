# Pekan 1: Log Clustering Baseline

## Summary
- Siapkan repository hybrid: script reproducible, notebook ringkas, README, requirements, `.gitignore`, dan output eksperimen kecil.
- Dataset utama: `timeline.csv` dari Zenodo scenario-1, sumber [Digital forensic timeline dataset extracted using log2timeline/Plaso](https://zenodo.org/records/15493424), DOI `10.5281/zenodo.15493424`, lisensi MIT.
- CSV lokal berisi 2.233.799 event; baseline Pekan 1 memakai stratified sample 100.000 event valid agar cepat dan tetap memenuhi minimum 50.000 event dari PDF.

## Key Changes
- Buat package Python untuk pipeline Pekan 1: profiling CSV, preprocessing teks, TF-IDF, reduksi dimensi 50D dengan TruncatedSVD, K-Means baseline, evaluasi, dan export laporan.
- CLI utama:
  `python -m log_clustering.week1 --input timeline.csv --sample-size 100000 --seed 42 --k-values 10,20,50 --output reports/week1`
- Preprocessing:
  lowercase, normalisasi path/hex panjang/timestamp/angka, tokenisasi regex, stopword English dari scikit-learn + stopword forensik custom, dan input teks berupa `source_<source> + cleaned_message`.
- Sampling:
  filter tahun `2000-2026`, pesan tidak kosong, sampling stratified by `source`, seed `42`, dengan sumber log kecil tetap terwakili.
- Output tracked ringkas:
  `metrics_week1.csv`, `cluster_summary_week1.csv`, `source_by_cluster.csv`, `dataset_profile.json`, dan plot kecil seperti silhouette/elbow serta source distribution.
- `.gitignore` mengecualikan `timeline.csv`, raw data, cache, model/intermediate besar, tetapi tetap mengizinkan output ringkas di `reports/week1`.

## Interfaces And Schemas
- Input wajib CSV memiliki kolom: `datetime`, `timestamp_desc`, `source`, `source_long`, `message`, `parser`, `display_name`, `tag`.
- `metrics_week1.csv`: `sample_size`, `seed`, `k`, `tfidf_features`, `svd_components`, `silhouette`, `calinski_harabasz`, `davies_bouldin`, `inertia`, `embedding_time_s`, `clustering_time_s`.
- `cluster_summary_week1.csv`: `k`, `cluster_id`, `size`, `top_terms`, `top_sources`, `example_messages`.
- README menjelaskan sumber dataset, alasan dataset tidak ikut di-commit, cara menjalankan pipeline, dan ringkasan hasil Pekan 1 dalam Bahasa Indonesia.

## Test Plan
- Unit test preprocessing untuk memastikan path, hex panjang, timestamp, dan angka ternormalisasi konsisten.
- Smoke test pipeline pada sample kecil 1.000 baris dan memastikan semua file output terbentuk.
- Run utama pada stratified 100k dengan `k=10,20,50`; validasi metrik finite, jumlah cluster sesuai `k`, dan plot tidak kosong.
- Verifikasi README command dapat dijalankan dari fresh checkout setelah `pip install -r requirements.txt`.

## Assumptions
- Artifact final untuk Pekan 1 berbentuk hybrid: script + notebook + README.
- Baseline resmi memakai stratified sample 100k, bukan full 2.23M, agar realistis untuk laptop/Google Colab.
- Output GitHub dibuat ringkas; dataset besar tetap lokal dan dicantumkan via sumber Zenodo.
