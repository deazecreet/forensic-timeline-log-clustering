# Progress Pekan 1: Log Clustering Baseline

## Identitas Kelompok

**Kelompok 3**

| No. | Nama | NRP |
| ---: | --- | --- |
| 1 | Fransisco Allenxeon | 6025252006 |
| 2 | Azel Rizki Nasution | 6025252009 |
| 3 | Muhammad Aldi Firmansyah | 6025252011 |

## Ringkasan

Pekan 1 berfokus pada pembuatan pipeline dasar untuk eksperimen log clustering pada forensic timeline hasil log2timeline/Plaso. Target utama pekan ini adalah memastikan dataset bisa dibaca, diprofilkan, dibersihkan, direpresentasikan sebagai fitur numerik, lalu diuji dengan baseline clustering sederhana.

Secara ilmiah, Pekan 1 diposisikan sebagai tahap baseline. Baseline diperlukan agar eksperimen lanjutan pada Pekan 2-4 memiliki pembanding awal yang jelas. Sebelum memakai metode embedding dan clustering yang lebih kompleks, pipeline sederhana berbasis TF-IDF dan K-Means perlu dibuat terlebih dahulu untuk memastikan bahwa data, preprocessing, sampling, evaluasi, dan mekanisme export hasil sudah berjalan end-to-end.

Pipeline yang sudah dibuat mencakup:

- profiling dataset `timeline.csv`
- preprocessing teks pada kolom `message`
- stratified sampling berdasarkan kolom `source`
- representasi teks menggunakan TF-IDF
- reduksi dimensi menggunakan TruncatedSVD
- baseline clustering menggunakan K-Means
- evaluasi clustering menggunakan metrik internal
- export hasil dalam format CSV, JSON, dan PNG

Dataset utama berasal dari Zenodo scenario-1:

- sumber: <https://zenodo.org/records/15493424>
- DOI: `10.5281/zenodo.15493424`
- file lokal: `timeline.csv`

## Progress Implementasi

### 1. Struktur Project

Project sudah disiapkan sebagai repository Python sederhana dan reproducible.

| Path | Fungsi |
| --- | --- |
| `log_clustering/preprocessing.py` | Fungsi preprocessing teks log |
| `log_clustering/week1.py` | Pipeline utama Pekan 1 |
| `notebooks/week1_baseline.ipynb` | Notebook ringkas untuk menjalankan baseline |
| `tests/` | Unit test dan smoke test pipeline |
| `reports/week1/` | Output hasil eksperimen Pekan 1 |
| `README.md` | Dokumentasi cara menjalankan project |
| `requirements.txt` | Daftar dependency Python |
| `.gitignore` | Mengecualikan dataset besar dan cache dari Git |

Struktur ini dibuat agar eksperimen tidak hanya bisa dijalankan sekali secara lokal, tetapi juga dapat direproduksi oleh anggota kelompok lain atau pembaca repository. Pemisahan antara `preprocessing.py`, `week1.py`, `tests/`, dan `reports/` juga membantu membedakan antara kode pipeline, kode pengujian, dan hasil eksperimen.

### 2. Profiling Dataset

Dataset diprofilkan untuk memahami kondisi awal data sebelum masuk ke proses clustering.

| Informasi | Nilai |
| --- | ---: |
| Total event | 2.233.799 |
| Event valid | 2.175.779 |
| Event invalid / di luar filter | 58.020 |
| Message kosong | 0 |
| Rata-rata panjang message | 171,61 karakter |
| Panjang message maksimum | 2.304.741 karakter |
| Rentang waktu valid | 2000-01-07 sampai 2026-12-24 |

Event dianggap valid jika:

- kolom `message` tidak kosong
- tahun pada `datetime` berada pada rentang `2000-2026`

Alasan profiling dilakukan:

- memastikan jumlah event yang tersedia memenuhi kebutuhan eksperimen
- mengetahui distribusi source agar sampling tidak bias
- menemukan nilai timestamp tidak wajar
- mengetahui panjang message karena message yang sangat panjang dapat memengaruhi waktu preprocessing dan fitur TF-IDF
- mendokumentasikan kondisi dataset agar eksperimen dapat diaudit

Filter tahun `2000-2026` digunakan karena dataset forensic timeline dapat memuat timestamp placeholder atau tidak wajar, misalnya tahun `1601`, `0000`, atau timestamp di luar rentang analisis. Jika timestamp seperti ini tidak difilter, sebagian event dapat memberi noise pada analisis temporal dan membuat profil dataset kurang representatif terhadap aktivitas sistem modern.

### 3. Distribusi Source

Distribusi source valid pada dataset:

| Source | Jumlah Event Valid | Makna Umum |
| --- | ---: | --- |
| `FILE` | 1.492.389 | Artefak file system, file stat, NTFS USN change |
| `REG` | 586.662 | Windows Registry |
| `EVT` | 59.632 | Windows Event Log |
| `PE` | 32.507 | Portable Executable, seperti EXE/DLL/SYS |
| `WEBHIST` | 1.742 | Browser history, cache, cookie |
| `OLECF` | 1.703 | OLE Compound File / metadata dokumen |
| `LOG` | 586 | Prefetch, SetupAPI log, PCA log, dan log lain |
| `AMCACHE` | 384 | Jejak executable dari Amcache |
| `LNK` | 174 | Windows Shortcut |

Distribusi ini menunjukkan dataset sangat didominasi oleh source `FILE` dan `REG`. Karena itu, pipeline menggunakan stratified sampling agar source kecil seperti `WEBHIST`, `AMCACHE`, dan `LNK` tetap terwakili.

Alasan source penting:

- kolom `source` menunjukkan jenis artefak forensik asal event
- event dari source berbeda dapat memiliki makna investigatif yang berbeda walaupun teks message mirip
- source dapat membantu interpretasi cluster, misalnya cluster yang didominasi `WEBHIST` dapat berkaitan dengan aktivitas browser, sedangkan cluster yang didominasi `REG` dapat berkaitan dengan perubahan registry

Dominasi `FILE` dan `REG` juga menjadi alasan mengapa sampling acak biasa kurang ideal. Jika random sampling dilakukan tanpa memperhatikan source, source kecil berisiko hilang atau terlalu sedikit muncul pada sample. Hal ini dapat membuat baseline clustering hanya menggambarkan pola file system dan registry, bukan forensic timeline secara lebih luas.

### 4. Preprocessing Teks

Preprocessing dilakukan pada kolom `message` dan digabung dengan metadata `source`.

Langkah preprocessing:

- mengubah teks menjadi lowercase
- mengganti URL menjadi `url_token`
- mengganti path menjadi `path_token`
- mengganti timestamp menjadi `time_token`
- mengganti hex panjang menjadi `hex_token`
- mengganti angka menjadi `num_token`
- tokenisasi berbasis regex
- menghapus stopwords English dari scikit-learn
- menghapus custom forensic stopwords
- menambahkan token source, misalnya `source_file`, `source_reg`, atau `source_webhist`

Contoh:

```text
Source asli:
FILE

Message asli:
NTFS:\Windows\SoftwareDistribution\SLS\...\sls.cab Type: file

Clean text:
source_file path_token
```

Penambahan token source dilakukan agar model tetap mengetahui asal event walaupun isi message sudah dinormalisasi.

Alasan preprocessing dilakukan:

- log forensic mengandung banyak nilai unik seperti path, URL, hash, timestamp, GUID, dan angka
- nilai unik tersebut dapat membuat TF-IDF terlalu fokus pada detail spesifik, bukan pola umum event
- normalisasi membantu menyamakan event yang secara struktur mirip tetapi memiliki nilai detail berbeda
- stopword removal mengurangi kata umum yang sering muncul tetapi kurang membedakan cluster

Contoh, dua path berbeda seperti:

```text
NTFS:\Windows\System32\cmd.exe
NTFS:\Users\User\AppData\Local\Temp\a.exe
```

keduanya sama-sama dinormalisasi menjadi:

```text
path_token
```

Dengan cara ini, model tetap mengetahui bahwa event mengandung path, tetapi tidak terlalu terpengaruh oleh variasi path spesifik yang jumlahnya sangat banyak.

Custom forensic stopwords digunakan karena stopword umum bahasa Inggris belum cukup untuk dataset forensic timeline. Kata seperti `file`, `type`, `windows`, `microsoft`, `user`, `system`, dan `flags` sering muncul di artefak Windows, tetapi tidak selalu membantu membedakan pola cluster. Informasi penting seperti asal event tetap dipertahankan melalui token source, misalnya `source_file` atau `source_reg`.

### 4.1 Contoh Data dan Hasil Cleaning

Berikut adalah contoh satu event asli dari dataset `timeline.csv` dan bagaimana event tersebut diubah menjadi teks bersih untuk TF-IDF.

Data asli:

| Kolom | Nilai |
| --- | --- |
| `datetime` | `2000-01-07T16:20:34.000000+00:00` |
| `timestamp_desc` | `Content Modification Time` |
| `source` | `PE` |
| `source_long` | `PE/COFF file` |
| `parser` | `pe` |
| `display_name` | `NTFS:\Windows\System32\dxpps.dll` |
| `tag` | `-` |
| `message` | `PE Type: Dynamic Link Library (DLL) Import hash: ace64f92359bb54a9f6d5167468fbf7f Export DLL name: dxpps.dll` |

Tahap 1: normalisasi source.

```text
Input source:
PE

Output normalize_source:
source_pe
```

Tahap 2: cleaning message.

```text
Input message:
PE Type: Dynamic Link Library (DLL) Import hash: ace64f92359bb54a9f6d5167468fbf7f Export DLL name: dxpps.dll

Output clean_message:
pe dynamic link library dll import hash hex_token export dll dxpps dll
```

Penjelasan perubahan:

- `PE` menjadi `pe` karena semua teks diubah menjadi lowercase
- `Type` dihapus karena termasuk custom stopword
- hash panjang `ace64f92359bb54a9f6d5167468fbf7f` diganti menjadi `hex_token`
- tanda baca seperti `:` dan `()` tidak dipertahankan sebagai token
- kata penting seperti `dynamic`, `link`, `library`, `dll`, `import`, dan `export` tetap dipertahankan karena membantu menggambarkan karakteristik file PE/DLL

Tahap 3: membentuk final text untuk TF-IDF.

```text
Output build_clustering_text:
source_pe pe dynamic link library dll import hash hex_token export dll dxpps dll
```

Final text tersebut adalah teks yang masuk ke TF-IDF. Dengan format ini, model tidak hanya melihat isi `message`, tetapi juga mengetahui bahwa event berasal dari source `PE`. Token `hex_token` menjaga informasi bahwa ada hash panjang tanpa membuat model terlalu bergantung pada nilai hash yang spesifik dan unik.

### 5. Sampling

Baseline Pekan 1 menggunakan stratified sample sebanyak 100.000 event valid.

| Source | Jumlah Sample |
| --- | ---: |
| `FILE` | 68.591 |
| `REG` | 26.963 |
| `EVT` | 2.741 |
| `PE` | 1.494 |
| `WEBHIST` | 80 |
| `OLECF` | 78 |
| `LOG` | 27 |
| `AMCACHE` | 18 |
| `LNK` | 8 |

Sampling menggunakan seed `42` agar hasil dapat direproduksi.

Alasan menggunakan stratified sampling:

- dataset terlalu besar untuk eksperimen baseline ringan jika seluruh 2,2 juta event langsung digunakan
- jumlah 100.000 event tetap berada di atas minimum 50.000 event yang diminta pada detail project
- stratifikasi menjaga proporsi source agar sample tetap mencerminkan struktur dataset
- source kecil tetap diberi kesempatan muncul sehingga interpretasi cluster tidak sepenuhnya didominasi oleh source besar
- seed `42` memastikan eksperimen dapat diulang dengan hasil sampling yang sama

Dengan pendekatan ini, pipeline menjadi cukup ringan untuk laptop atau Google Colab, tetapi tetap menggunakan jumlah data yang besar dan representatif untuk baseline awal.

### 6. Representasi dan Clustering

Representasi teks menggunakan TF-IDF dengan konfigurasi:

| Parameter | Nilai |
| --- | ---: |
| `max_features` | 5.000 |
| `min_df` | 5 |
| `max_df` | 0,95 |
| `ngram_range` | 1 sampai 2 |

Alasan menggunakan TF-IDF:

- TF-IDF merupakan baseline kuat dan umum untuk representasi teks klasik
- TF-IDF mudah diinterpretasikan karena fitur berasal dari token atau pasangan token yang muncul pada log
- bobot TF-IDF menurunkan pengaruh kata yang terlalu umum dan menaikkan pengaruh kata yang lebih khas pada event tertentu
- cocok sebagai pembanding awal sebelum memakai embedding yang lebih kompleks seperti Word2Vec, Doc2Vec, atau Sentence-BERT

Parameter `max_features=5000` digunakan untuk membatasi jumlah fitur agar matrix tidak terlalu besar. Parameter `min_df=5` membuang token yang terlalu jarang muncul, sedangkan `max_df=0,95` membuang token yang muncul terlalu sering pada hampir semua dokumen. `ngram_range=(1, 2)` digunakan agar model dapat menangkap token tunggal dan frasa pendek, misalnya kombinasi seperti `source_file path_token`.

Hasil TF-IDF direduksi menggunakan `TruncatedSVD + Normalizer` menjadi 50 dimensi. TruncatedSVD dipakai karena output TF-IDF berbentuk sparse matrix, yaitu matrix yang sebagian besar nilainya nol. PCA biasa cenderung membutuhkan dense matrix sehingga dapat lebih boros memori. TruncatedSVD lebih sesuai untuk matrix teks sparse dan umum digunakan dalam pendekatan Latent Semantic Analysis.

Jumlah 50 dimensi dipilih karena sesuai dengan detail task project yang menetapkan dimensionality reduction 50 dimensi untuk representasi high-dimensional sebelum clustering.

Clustering dilakukan menggunakan K-Means dengan nilai `k`:

```text
10, 20, 50
```

Alasan menggunakan K-Means:

- K-Means sederhana, cepat, dan cocok sebagai baseline awal
- hasilnya mudah dibandingkan untuk beberapa nilai `k`
- bekerja cukup baik setelah fitur high-dimensional direduksi ke 50 dimensi
- dapat menjadi pembanding untuk metode Pekan 2 seperti DBSCAN, HDBSCAN, Agglomerative Clustering, dan GMM

Nilai `k=10`, `k=20`, dan `k=50` digunakan untuk melihat sensitivitas awal terhadap jumlah cluster. Nilai yang lebih kecil memberi cluster yang lebih umum, sedangkan nilai yang lebih besar memberi cluster yang lebih spesifik.

## Hasil Eksperimen

### Metrik Clustering

| k | Silhouette | Calinski-Harabasz | Davies-Bouldin | Inertia | Waktu Clustering |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0,5586 | 24013,23 | 1,5412 | 24232,81 | 0,629 s |
| 20 | 0,6312 | 24522,99 | 1,3693 | 13534,56 | 0,972 s |
| 50 | 0,7261 | 26010,45 | 0,8435 | 5571,01 | 2,516 s |

Interpretasi awal:

- `k=50` menghasilkan Silhouette Score tertinggi, yaitu `0,7261`.
- `k=50` juga menghasilkan Davies-Bouldin Index terendah, yaitu `0,8435`.
- Inertia menurun ketika jumlah cluster bertambah, sesuai perilaku umum K-Means.
- Waktu clustering masih relatif ringan untuk sample 100.000 event.

Makna metrik:

- Silhouette Score mengukur seberapa dekat event dengan cluster sendiri dibanding cluster lain. Semakin tinggi semakin baik.
- Calinski-Harabasz mengukur rasio separasi antar-cluster terhadap kepadatan intra-cluster. Semakin tinggi semakin baik.
- Davies-Bouldin mengukur kemiripan antar-cluster. Semakin rendah semakin baik.
- Inertia mengukur total jarak data ke centroid cluster. Semakin rendah berarti cluster makin rapat, tetapi nilainya cenderung turun saat `k` bertambah sehingga perlu dibaca bersama metrik lain.

Berdasarkan metrik internal, konfigurasi terbaik pada baseline Pekan 1 adalah:

```text
TF-IDF + TruncatedSVD 50D + K-Means k=50
```

Walaupun `k=50` terlihat paling baik berdasarkan metrik internal, hasil ini belum otomatis berarti paling baik secara forensik. Pada tahap berikutnya, cluster tetap perlu diperiksa interpretabilitasnya, misalnya dengan melihat `top_terms`, `top_sources`, dan contoh message pada `cluster_summary_week1.csv`. Dalam konteks digital forensics, cluster yang baik tidak hanya tinggi secara metrik, tetapi juga harus membantu investigator memahami pola aktivitas.

### Output Artefak

Output hasil eksperimen tersimpan di folder `reports/week1/`.

| File | Isi |
| --- | --- |
| `dataset_profile.json` | Profil dataset, konfigurasi sampling, TF-IDF, SVD, dan clustering |
| `metrics_week1.csv` | Tabel metrik clustering untuk setiap nilai `k` |
| `cluster_summary_week1.csv` | Ringkasan tiap cluster, top terms, top sources, dan contoh message |
| `source_by_cluster.csv` | Distribusi source pada setiap cluster |
| `elbow_inertia.png` | Plot inertia terhadap nilai `k` |
| `silhouette_scores.png` | Plot Silhouette Score terhadap nilai `k` |
| `source_distribution_sample.png` | Plot distribusi source pada sample |

## Validasi

Validasi kode dilakukan dengan test otomatis.

Test yang tersedia:

- unit test preprocessing
- smoke test pipeline Pekan 1 menggunakan CSV mini

Command:

```bash
pytest
```

Hasil validasi terakhir:

```text
4 passed
```

Validasi ini penting karena pipeline terdiri dari beberapa tahap yang saling bergantung. Unit test memastikan fungsi preprocessing tetap bekerja sesuai ekspektasi, sedangkan smoke test memastikan pipeline end-to-end dapat berjalan pada dataset kecil dan menghasilkan file output yang diperlukan. Dengan demikian, perubahan kode di masa depan dapat dicek lebih cepat.

## Kesimpulan Pekan 1

Pekan 1 berhasil menyelesaikan pipeline baseline end-to-end untuk log clustering forensic timeline. Dataset berhasil diprofilkan, data valid berhasil difilter, preprocessing teks berhasil diterapkan, sample 100.000 event berhasil dibuat secara stratified, dan baseline TF-IDF + K-Means berhasil dijalankan.

Hasil awal menunjukkan bahwa konfigurasi `k=50` memberikan performa terbaik di antara nilai `k` yang diuji berdasarkan Silhouette Score dan Davies-Bouldin Index. Output yang dihasilkan sudah cukup untuk menjadi dasar eksperimen lanjutan pada Pekan 2, seperti membandingkan representasi Word2Vec, Doc2Vec, Sentence-BERT, serta algoritma clustering lain seperti DBSCAN, HDBSCAN, Agglomerative Clustering, dan GMM.

Secara metodologis, Pekan 1 sudah memenuhi fungsi baseline: data sudah dipahami, preprocessing sudah terdokumentasi, representasi numerik sudah terbentuk, model clustering awal sudah berjalan, dan evaluasi awal sudah tersedia. Dengan baseline ini, peningkatan pada pekan berikutnya dapat dinilai secara lebih objektif karena ada pembanding awal yang jelas.

## Catatan Lanjutan

Beberapa hal yang dapat dikembangkan pada pekan berikutnya:

- menambahkan representasi embedding selain TF-IDF
- melakukan tuning parameter clustering
- menambahkan visualisasi PCA, UMAP, dan t-SNE
- melakukan interpretasi cluster secara forensik
- membandingkan kualitas cluster antar metode
- memperluas custom stopwords berdasarkan hasil observasi cluster
