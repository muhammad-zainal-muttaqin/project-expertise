"""Render contoh kotak pembatas (GT vs prediksi) untuk laporan garis waktu eksperimen.

Skrip ini murni untuk visualisasi/pelaporan, bukan bagian dari pipeline evaluasi
resmi. Ia membaca dump prediksi mentah (`.npz`, format `[x1,y1,x2,y2,skor,kelas]`
pada ambang conf=0,001, per V2-E-034/eval_new763_pycoco.py), label GT berformat
YOLO ternormalisasi, dan citra uji asli new763 (`SawitMVC-Depth-YOLO`), lalu
memilih secara otomatis (bukan dipilih manual/cherry-picked) beberapa citra yang
paling representatif untuk dua kategori: "deteksi bersih" (semua kotak GT
cocok kelas dan lokasinya) dan "kekeliruan klasifikasi" (kotak terdeteksi pada
lokasi yang benar tetapi kelas kematangan tertukar, mis. B2/B3/B4).

Pencocokan GT-vs-prediksi memakai IoU tunggal per kotak dengan strategi greedy
per-kelas-mentah: untuk setiap grup prediksi pada lokasi yang sama (npz
menyimpan banyak hipotesis kelas per lokasi), hanya kandidat berskor tertinggi
yang dipakai sebagai "kelas terpilih" model pada lokasi itu — pendekatan ini
meniru argmax softmax final tanpa perlu menjalankan ulang model.

Jalankan:
    py -3 scripts/render_bbox_examples.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path(r"D:\Work\Assisten-Dosen\SawitMVC-Depth\SawitMVC-Depth-YOLO\test")
NPZ_PATH = Path("results/new763/predictions/rfdetr_l_rgb_s42_i1280__test.npz")
OUT_DIR = Path(
    r"C:\Users\Zainal\AppData\Local\Temp\claude\D--Work-Assisten-Dosen-project-expertise"
    r"\395755d6-7463-4ee1-a643-cd7ebbff3bbd\scratchpad\bbox_examples"
)

CLASS_NAMES = ["B1", "B2", "B3", "B4"]
# Hijau -> merah mengikuti urutan kematangan (BGR untuk cv2).
CLASS_COLORS = [(90, 200, 60), (0, 200, 230), (0, 140, 255), (30, 30, 220)]
CONF_MIN = 0.25
NMS_IOU = 0.5
MATCH_IOU = 0.5


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matriks antara dua kumpulan kotak xyxy piksel, bentuk (Na,4) dan (Nb,4)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def top1_per_lokasi(pred: np.ndarray) -> np.ndarray:
    """Kolaps banyak hipotesis kelas pada lokasi kotak yang sama menjadi satu
    baris (kelas berskor tertinggi), lalu NMS greedy sederhana lintas kelas."""
    if pred.shape[0] == 0:
        return pred
    kunci = np.round(pred[:, :4] / 2.0).astype(np.int64)
    kelompok: dict[tuple, np.ndarray] = {}
    for i, k in enumerate(map(tuple, kunci)):
        if k not in kelompok or pred[i, 4] > pred[kelompok[k], 4]:
            kelompok[k] = i
    idx = np.array(sorted(kelompok.values()))
    top1 = pred[idx]
    top1 = top1[top1[:, 4] >= CONF_MIN]
    if top1.shape[0] == 0:
        return top1
    urut = np.argsort(-top1[:, 4])
    top1 = top1[urut]
    dipakai = np.ones(len(top1), bool)
    if len(top1) > 1:
        ious = iou_xyxy(top1[:, :4], top1[:, :4])
        for i in range(len(top1)):
            if not dipakai[i]:
                continue
            supresi = (ious[i] > NMS_IOU) & (np.arange(len(top1)) > i)
            dipakai[supresi] = False
    return top1[dipakai]


def baca_label_yolo(path_txt: Path, w: int, h: int) -> np.ndarray:
    if not path_txt.exists():
        return np.zeros((0, 5))
    baris = []
    for ln in path_txt.read_text().strip().splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = map(float, ln.split())
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        baris.append([x1, y1, x2, y2, c])
    return np.array(baris) if baris else np.zeros((0, 5))


def evaluasi_citra(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Cocokkan tiap kotak GT ke prediksi terdekat (IoU>=0,5, greedy skor
    tertinggi dulu) dan klasifikasikan sebagai benar/salah-kelas/tidak terdeteksi."""
    hasil = {"benar": 0, "salah_kelas": 0, "hilang": 0, "salah_tetangga": 0, "detail": []}
    if gt.shape[0] == 0:
        return hasil
    dipakai_pred = np.zeros(len(pred), bool)
    if len(pred):
        ious = iou_xyxy(gt[:, :4], pred[:, :4])
    for gi in range(len(gt)):
        if len(pred) == 0:
            hasil["hilang"] += 1
            continue
        kandidat = np.where(~dipakai_pred & (ious[gi] >= MATCH_IOU))[0]
        if len(kandidat) == 0:
            hasil["hilang"] += 1
            continue
        pi = kandidat[np.argmax(pred[kandidat, 4])]
        dipakai_pred[pi] = True
        kelas_gt, kelas_pred = int(gt[gi, 4]), int(pred[pi, 5])
        if kelas_gt == kelas_pred:
            hasil["benar"] += 1
        else:
            hasil["salah_kelas"] += 1
            if abs(kelas_gt - kelas_pred) == 1:
                hasil["salah_tetangga"] += 1
        hasil["detail"].append(
            {"gt": kelas_gt, "pred": kelas_pred, "skor": float(pred[pi, 4]), "iou": float(ious[gi, pi])}
        )
    return hasil


def gambar_kotak(citra: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = citra.copy()
    for x1, y1, x2, y2, c in gt:
        c = int(c)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, CLASS_COLORS[c], 3)
        cv2.putText(out, f"GT:{CLASS_NAMES[c]}", (p1[0], max(0, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS[c], 2, cv2.LINE_AA)
    for x1, y1, x2, y2, skor, c in pred:
        c = int(c)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, CLASS_COLORS[c], 2)
        teks = f"PR:{CLASS_NAMES[c]} {skor:.2f}"
        cv2.putText(out, teks, (p1[0], min(citra.shape[0] - 4, p2[1] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[c], 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--npz", type=Path, default=NPZ_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--n-benar", type=int, default=4)
    ap.add_argument("--n-salah", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)

    skor_citra = []
    for nama in data.keys():
        img_path = args.dataset_root / "images" / f"{nama}.jpg"
        lbl_path = args.dataset_root / "labels" / f"{nama}.txt"
        if not img_path.exists():
            continue
        citra = cv2.imread(str(img_path))
        h, w = citra.shape[:2]
        gt = baca_label_yolo(lbl_path, w, h)
        pred = top1_per_lokasi(data[nama].astype(np.float64))
        ev = evaluasi_citra(gt, pred)
        n_kotak = gt.shape[0]
        skor_bersih = ev["benar"] - ev["salah_kelas"] - ev["hilang"]
        skor_citra.append({
            "nama": nama, "n_kotak": n_kotak, "benar": ev["benar"],
            "salah_kelas": ev["salah_kelas"], "salah_tetangga": ev["salah_tetangga"],
            "hilang": ev["hilang"], "skor_bersih": skor_bersih,
            "gt": gt, "pred": pred, "detail": ev["detail"],
        })

    kandidat_bersih = [s for s in skor_citra if s["n_kotak"] >= 3 and s["salah_kelas"] == 0 and s["hilang"] == 0]
    kandidat_bersih.sort(key=lambda s: (-s["benar"], s["nama"]))
    terpilih_benar = kandidat_bersih[: args.n_benar]

    kandidat_salah = [s for s in skor_citra if s["salah_tetangga"] > 0]
    kandidat_salah.sort(key=lambda s: (-s["salah_tetangga"], -s["salah_kelas"], s["nama"]))
    terpilih_salah = kandidat_salah[: args.n_salah]

    ringkasan = {"benar": [], "salah": []}
    for kategori, daftar in (("benar", terpilih_benar), ("salah", terpilih_salah)):
        for s in daftar:
            img_path = args.dataset_root / "images" / f"{s['nama']}.jpg"
            citra = cv2.imread(str(img_path))
            citra = gambar_kotak(citra, s["gt"], s["pred"])
            out_path = args.out / f"{kategori}_{s['nama']}.jpg"
            cv2.imwrite(str(out_path), citra, [cv2.IMWRITE_JPEG_QUALITY, 88])
            ringkasan[kategori].append({
                "nama": s["nama"], "file": out_path.name, "n_kotak": s["n_kotak"],
                "benar": s["benar"], "salah_kelas": s["salah_kelas"],
                "salah_tetangga": s["salah_tetangga"], "hilang": s["hilang"],
                "detail": s["detail"],
            })
            print(f"[{kategori}] {s['nama']}: benar={s['benar']} salah_kelas={s['salah_kelas']} "
                  f"hilang={s['hilang']} -> {out_path}")

    (args.out / "ringkasan.json").write_text(json.dumps(ringkasan, indent=2))
    print(f"\nTotal citra diproses: {len(skor_citra)}")
    print(f"Kandidat bersih (semua benar, >=3 kotak): {len(kandidat_bersih)}")
    print(f"Kandidat dengan kekeliruan kelas tetangga: {len(kandidat_salah)}")
    print(f"Ringkasan tersimpan di: {args.out / 'ringkasan.json'}")


if __name__ == "__main__":
    main()
