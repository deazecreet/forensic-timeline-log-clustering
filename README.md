# Log Clustering for Forensic Timeline Analysis

Repository ini berisi implementasi Pekan 1 untuk mini project **Log Clustering for Forensic Timeline Analysis** pada output CSV log2timeline/Plaso.

## Dataset

Dataset yang digunakan adalah `scenario-1` dari Zenodo:

- Judul: [Digital forensic timeline dataset extracted using log2timeline/Plaso](https://zenodo.org/records/15493424)
- DOI: `10.5281/zenodo.15493424`
- Lisensi: MIT License
- File lokal yang dipakai: `timeline.csv`

`timeline.csv` tidak dimasukkan ke Git karena ukurannya besar. Letakkan file tersebut di root repository sebelum menjalankan pipeline.

## Pekan 1

Target Pekan 1:

- profiling dataset Plaso CSV
- preprocessing pesan log
- sampling stratified 100.000 event valid berdasarkan `source`
- baseline TF-IDF
- reduksi dimensi 50D dengan TruncatedSVD
- baseline K-Means untuk `k=10,20,50`
- export metrik, ringkasan cluster, dan plot kecil

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Baseline

```bash
python -m log_clustering.week1 --input timeline.csv --sample-size 100000 --seed 42 --k-values 10,20,50 --output reports/week1
```

Output utama:

- `reports/week1/dataset_profile.json`
- `reports/week1/metrics_week1.csv`
- `reports/week1/cluster_summary_week1.csv`
- `reports/week1/source_by_cluster.csv`
- `reports/week1/elbow_inertia.png`
- `reports/week1/silhouette_scores.png`
- `reports/week1/source_distribution_sample.png`

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

Dataset lokal berisi 2.233.799 event. Dengan filter tahun `2000-2026` dan pesan tidak kosong, terdapat 2.175.779 event valid. Pipeline memakai stratified sample 100.000 event untuk baseline awal.

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
| 10 | 0.6105 | 26164.24 | 1.3492 | 22887.27 |
| 20 | 0.6438 | 25298.43 | 1.2739 | 13221.70 |
| 50 | 0.7325 | 27389.34 | 0.8898 | 5322.28 |

Pada baseline Pekan 1, `k=50` memberi Silhouette tertinggi dan Davies-Bouldin terendah dari tiga nilai `k` yang diuji. Detail lengkap tersedia di `reports/week1/metrics_week1.csv` dan `reports/week1/cluster_summary_week1.csv`.
