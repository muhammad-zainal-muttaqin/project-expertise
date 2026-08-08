# CLAUDE.md — Panduan Kerja

Baca seluruhnya sebelum mengubah apa pun.

## Bahasa

Seluruh isi repo dan percakapan memakai **Bahasa Indonesia**.
Istilah teknis asing ditulis apa adanya tanpa diterjemahkan.

## Apa Ini

Repo eksperimen untuk membandingkan tiga arsitektur detektor (YOLO26l,
RT-DETR-L, RF-DETR-L) pada dataset RGB dan RGB+Depth (4-kanal), lalu
mengukur dampaknya terhadap deteksi, klasifikasi kematangan, dan counting
tandan sawit per pohon.

**Bukan** repo tinjauan pustaka — itu ada di
[Research-Pipeline](https://github.com/muhammad-zainal-muttaqin/Research-Pipeline).

## Dataset

Dua dataset, keduanya CC BY-NC 4.0, dari ULM-DS-Lab:

| | SawitMVC | SawitMVC-Depth |
|---|---|---|
| Sumber | [HuggingFace](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO) | [HuggingFace](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-Depth) |
| Pohon | 953 | 352 |
| Citra | 3.992 | 1.408 |
| Resolusi | 960 x 1280 | 1.280 x 800 |
| Bbox | 18.540 | 2.299 |
| Depth | Tidak | Ya (Orbbec, uint16 mm) |
| Split | 716 / 96 / 141 | Perlu dibuat |

Detail lengkap: [docs/DATASET.md](docs/DATASET.md).

## Metrik yang berlaku saat ini

Satu-satunya angka deteksi yang boleh dikutip (dari E-021, SawitMVC 953 pohon):

| Model | Test mAP50 | Test mAP50-95 |
|---|---|---|
| YOLO26l @ 1280 | 0,5300 | 0,2568 |
| RT-DETR-L @ 1280 | 0,5784 | 0,2707 |
| RF-DETR-L @ 1280 | 0,6038 | 0,2770 |

Angka counting terbaik (Baseline-SawitMVC, YOLO26m):

| Counter | Class &plusmn;1 Acc | Tree &plusmn;1 Acc | Macro MAE |
|---|---|---|---|
| Ridge + F_all | 77,48% | 32,62% | 1,036 |

Angka counting untuk YOLO26l, RT-DETR-L, RF-DETR-L **belum ada**.

## Aturan eksperimen

- Satu entri = satu hipotesis yang falsifiable.
- Append-only. Jangan edit entri lama.
- Hasil negatif wajib dicatat dengan bobot yang sama.
- Angka apa adanya. Jangan dibungkus.
- Setiap klaim numerik harus terlacak ke sumber (skrip, JSON, log).
- Evaluasi deteksi: `pycocotools` (mengikat dari E-025).
- Evaluasi counting: pipeline dari Baseline-SawitMVC.
- Setiap angka menyebut dataset dan split.

## Hal yang sudah dicoba dan GAGAL (jangan diulang)

Daftar lengkap: [docs/REKAP.md](docs/REKAP.md) bagian "Percobaan Gagal".
Ringkasan singkat:

1. **Early fusion (depth sebagai kanal ke-4 langsung)** — regresi, bukan
   peningkatan (E-022, E-027). Depth merugikan YOLO26n sebesar −0,0230 mAP.
2. **Tuning hyperparameter** — sudah habis dijalankan, tidak naik lagi.
3. **SAHI dan teknik siap-pakai** — sudah dicoba sendiri oleh pengguna, tidak
   satu pun menaikkan mAP.
4. **Gate init-nol pada cabang samping** — gate tidak pernah terbuka, γ ≈ 0 (F-007).
5. **Konsistensi lintas-sisi** — plafon hanya 0,2794 (F-003).
6. **Fusi menengah/akhir dari nol** — tidak konklusif, semua CI memuat nol (E-032).

## Hal yang BERHASIL (boleh dibangun di atasnya)

1. **RF-DETR-L** adalah detektor terbaik saat ini (mAP50 0,6038).
2. **Pipeline counting Ridge + F_all** sudah modular dan established.
3. **Reproyeksi depth** ke RGB sudah tervalidasi untuk SawitMVC-Depth.
4. **Frekuensi tinggi memisahkan tandan dari pelepah** (+0,0731 B4, F-002).

## Cara kerja

- Paralelisme dibatasi VRAM, bukan slot tetap. Baca `nvidia-smi` sebelum
  menyalakan run berikutnya.
- Jangan mengarang eksperimen tambahan untuk "mengisi GPU".
- Laporkan hasil apa adanya.
