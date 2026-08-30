"""Render contoh nyata "crop lintas sudut pandang, lalu diklasifikasikan"
untuk laporan garis waktu eksperimen.

Kepala klasifikasi kematangan (dua-tahap di jalur utama, modul C di
`pipeline-pertandan`) tidak menerima citra penuh satu pohon. Ia menerima
potongan (*crop*) di sekitar satu kotak pembatas, satu potongan per kemunculan
tandan pada satu sisi foto. Skrip ini mengambil `_confirmedLinks`/`bunches`
pada berkas `linked/*.json` new763 (graf identitas fisik tandan lintas sisi
yang sudah diverifikasi, dipakai `pipeline-pertandan`) untuk menemukan dua
kasus nyata:

  1. Satu tandan fisik yang muncul di tiga sisi dan diberi label kematangan
     yang SAMA pada ketiganya (`class_mismatch=false`) -- kasus umum.
  2. Satu tandan fisik yang muncul di dua sisi tetapi diberi label kematangan
     BERBEDA pada tiap sisi (`class_mismatch=true`) -- satu-satunya kasus
     semacam ini pada seluruh split uji new763, dan alasan langsung kenapa
     agregasi lintas sudut pandang (bukan klasifikasi satu sisi) dibutuhkan.

Jalankan:
    py -3 scripts/render_multiview_crop_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

DATASET_ROOT = Path(r"D:\Work\Assisten-Dosen\SawitMVC-Depth\SawitMVC-Depth-YOLO\test")
LINKED_DIR = DATASET_ROOT / "linked"
OUT_DIR = Path(
    r"C:\Users\Zainal\AppData\Local\Temp\claude\D--Work-Assisten-Dosen-project-expertise"
    r"\395755d6-7463-4ee1-a643-cd7ebbff3bbd\scratchpad\crop_examples"
)
PAD_FRAC = 0.18


def potong(tree_id: str, side: str, bbox_pixel: list[int]) -> "cv2.typing.MatLike":
    img_path = DATASET_ROOT / "images" / f"{tree_id}_{side.split('_')[1]}.jpg"
    citra = cv2.imread(str(img_path))
    h, w = citra.shape[:2]
    x1, y1, x2, y2 = bbox_pixel
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * PAD_FRAC), int(bh * PAD_FRAC)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return citra[y1:y2, x1:x2]


DISPLAY_MIN_SIDE = 320


def simpan_bunch(tree_id: str, bunch: dict, prefix: str) -> list[dict]:
    hasil = []
    for a in bunch["appearances"]:
        crop = potong(tree_id, a["side"], a["bbox_pixel"])
        h, w = crop.shape[:2]
        asli_w, asli_h = w, h
        skala = max(1.0, DISPLAY_MIN_SIDE / min(h, w))
        if skala > 1.0:
            crop = cv2.resize(crop, (int(w * skala), int(h * skala)), interpolation=cv2.INTER_CUBIC)
        nama = f"{prefix}_{tree_id}_{a['side']}.jpg"
        cv2.imwrite(str(OUT_DIR / nama), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        hasil.append({
            "side": a["side"], "kelas": a["class_name"], "file": nama,
            "resolusi_asli": f"{asli_w}x{asli_h}px",
        })
        print(f"  {tree_id} {a['side']}: kelas={a['class_name']} asli={asli_w}x{asli_h}px -> {nama}")
    return hasil


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    setuju = json.loads((LINKED_DIR / "DAMIMAS_A21B_0012.json").read_text())
    bunch_setuju = next(b for b in setuju["bunches"] if b["bunch_id"] == 2)
    assert bunch_setuju["appearance_count"] == 3 and not bunch_setuju["class_mismatch"]
    print("Kasus 1 -- label konsisten lintas sisi:")
    r1 = simpan_bunch("DAMIMAS_A21B_0012", bunch_setuju, "setuju")

    beda = json.loads((LINKED_DIR / "DAMIMAS_A21B_0037.json").read_text())
    bunch_beda = next(b for b in beda["bunches"] if b["bunch_id"] == 1)
    assert bunch_beda["class_mismatch"]
    print("Kasus 2 -- label BERBEDA lintas sisi (satu-satunya di split uji):")
    r2 = simpan_bunch("DAMIMAS_A21B_0037", bunch_beda, "beda")

    ringkasan = {
        "kasus_1_konsisten": {"tree_id": "DAMIMAS_A21B_0012", "bunch_id": 2, "kelas_final": bunch_setuju["class"], "sisi": r1},
        "kasus_2_berbeda": {"tree_id": "DAMIMAS_A21B_0037", "bunch_id": 1, "kelas_final": bunch_beda["class"], "sisi": r2},
    }
    (OUT_DIR / "ringkasan.json").write_text(json.dumps(ringkasan, indent=2))
    print(f"\nRingkasan tersimpan di: {OUT_DIR / 'ringkasan.json'}")


if __name__ == "__main__":
    main()
