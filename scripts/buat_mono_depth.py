"""Bangkitkan peta monocular-depth untuk SawitMVC-Depth (352) dan SawitMVC-YOLO (953).

Keluarannya mengikuti KONTRAK YANG SAMA dengan depth sensor
(`reproject_depth.py` -> `depth_png_352/`): PNG uint8 satu kanal, 0 = tidak ada
data, 1..255 = inverse depth pada rentang metrik TETAP [z_near, z_far].
Transformnya bukan tulisan ulang — `encode_inverse` diimpor langsung dari
`Research-Pipeline/experiments/code/build/depth_calib.py` supaya kanal mono dan
kanal sensor benar-benar sebanding piksel-per-piksel, bukan sekadar mirip.

Dua perbedaan yang MELEKAT pada monocular dan harus diingat saat membaca hasil:

1. Tidak ada piksel invalid. Sensor Orbbec meninggalkan ~29% lubang (0);
   monocular selalu mengisi penuh, jadi nilai 0 praktis tidak pernah muncul.
   Itu keunggulan sekaligus perbedaan distribusi — bukan bug.
2. Skalanya metrik menurut model, bukan menurut pengukuran. Model bisa keliru
   soal skala absolut sementara struktur relatifnya benar. Karena itu
   `probe_mono_vs_sensor.py` membandingkan keduanya sebelum dipakai melatih.

Model: yolo26l-depth.pt (ultralytics >= 8.4.104, task `depth`). Venv training
di-pin 8.4.103 yang BELUM punya task itu, jadi skrip ini dijalankan dengan
ultralytics yang dipasang berdampingan:

    PYTHONPATH=/workspace/.mono_ul .venv/bin/python scripts/buat_mono_depth.py \
        --src /workspace/SawitMVC-Depth/images --out /workspace/mono_png_352

    PYTHONPATH=/workspace/.mono_ul .venv/bin/python scripts/buat_mono_depth.py \
        --src /workspace/SawitMVC-YOLO/images --out /workspace/mono_png_953
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CALIB = Path("/workspace/Research-Pipeline/experiments/code/build")
if not (CALIB / "depth_calib.py").exists():
    sys.exit(f"FATAL: {CALIB}/depth_calib.py tidak ada — encoding sensor tidak bisa direuse. "
             "Jangan tulis ulang formulanya; clone Research-Pipeline dulu.")
sys.path.insert(0, str(CALIB))
from depth_calib import encode_inverse  # noqa: E402  (satu sumber kebenaran)

Z_NEAR, Z_FAR = 0.8, 15.0  # beku, dari depth_png_352/depth_meta.json
EKSTENSI = (".jpg", ".jpeg", ".png")


def muat_model(bobot: str):
    try:
        from ultralytics import YOLO
        import ultralytics
        import ultralytics.cfg as cfg
    except ImportError as e:
        sys.exit(f"FATAL: ultralytics tidak bisa diimpor ({e})")
    if "depth" not in cfg.TASKS:
        sys.exit(f"FATAL: ultralytics {ultralytics.__version__} belum punya task `depth`. "
                 "Jalankan dengan PYTHONPATH=/workspace/.mono_ul (lihat docstring).")
    print(f"ultralytics {ultralytics.__version__} | bobot {bobot}")
    return YOLO(bobot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="direktori citra RGB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bobot", default="yolo26l-depth.pt")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="disamakan dengan resolusi latih detektor (1280), bukan default 640")
    ap.add_argument("--z-near", type=float, default=Z_NEAR)
    ap.add_argument("--z-far", type=float, default=Z_FAR)
    ap.add_argument("--batas", type=int, default=0, help="hanya N citra pertama (uji cepat)")
    ap.add_argument("--paksa", action="store_true", help="tulis ulang PNG yang sudah ada")
    ap.add_argument("--batch", type=int, default=8,
                    help="citra per batch GPU; jangan dinaikkan tanpa cek VRAM")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        sys.exit(f"FATAL: {src} tidak ada")
    # rglob, bukan iterdir: SawitMVC-Depth datar (images/*.jpg) sedangkan
    # SawitMVC-YOLO sudah dimaterialisasi (images/{train,val,test}/*.jpg).
    # Keluaran tetap datar — stem unik lintas split (nama pohon + sisi).
    citra = sorted(p for p in src.rglob("*") if p.suffix.lower() in EKSTENSI)
    stem_unik = {p.stem for p in citra}
    if len(stem_unik) != len(citra):
        sys.exit(f"FATAL: stem tidak unik ({len(citra)} berkas, {len(stem_unik)} stem) — "
                 "keluaran datar akan saling menimpa")
    if args.batas:
        citra = citra[: args.batas]
    if not citra:
        sys.exit(f"FATAL: tidak ada citra di {src}")
    out.mkdir(parents=True, exist_ok=True)
    n_semua = len(citra)
    if not args.paksa:
        # idempotent: aman dijalankan ulang kalau proses terputus di tengah
        citra = [p for p in citra if not (out / f"{p.stem}.png").exists()]
    print(f"{n_semua} citra, {n_semua - len(citra)} sudah ada, {len(citra)} dikerjakan -> {out}")
    if not citra:
        print("tidak ada yang perlu dikerjakan")
        return 0

    model = muat_model(args.bobot)
    stat_m, n_jauh, n_dekat = [], 0, 0
    # Potong sendiri jadi batch kecil. JANGAN oper seluruh daftar ke predict():
    # ultralytics memperlakukan satu daftar sebagai SATU batch, jadi 1.408 citra
    # @1280 langsung minta ~16 GiB VRAM dan OOM. Dipotong begini, VRAM tetap
    # datar berapa pun jumlah citranya.
    def aliran_hasil():
        for j in range(0, len(citra), args.batch):
            potong = [str(p) for p in citra[j: j + args.batch]]
            for r in model.predict(source=potong, imgsz=args.imgsz, verbose=False):
                yield r

    for i, r in enumerate(aliran_hasil(), 1):
        p = Path(r.path)
        d = r.depth.data
        d = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
        d = np.squeeze(d).astype(np.float32)

        h, w = r.orig_shape  # tanpa membaca ulang citranya dari disk
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        n_dekat += int((d < args.z_near).mean() > 0.5)
        n_jauh += int((d > args.z_far).mean() > 0.5)
        stat_m.append([float(d.min()), float(np.median(d)), float(d.max())])

        # encode_inverse minta mm dengan 0=invalid; monocular selalu valid,
        # jadi cukup pastikan tidak ada nilai <=0 yang tersamar jadi "lubang".
        png = encode_inverse(np.maximum(d, 1e-3) * 1000.0, args.z_near, args.z_far)
        if not cv2.imwrite(str(out / f"{p.stem}.png"), png):
            sys.exit(f"FATAL: gagal menulis {out / (p.stem + '.png')}")
        if i % 200 == 0 or i == len(citra):
            print(f"  {i}/{len(citra)}")

    s = np.array(stat_m)
    meta = {
        "sumber": str(src),
        "model": args.bobot,
        "imgsz": args.imgsz,
        "z_near_m": args.z_near,
        "z_far_m": args.z_far,
        "kontrak": "PNG uint8 1 kanal; 1..255 inverse depth pada [z_near, z_far]; "
                   "0 tidak dipakai (monocular tidak punya lubang)",
        "encoding": "encode_inverse() dari Research-Pipeline/.../depth_calib.py — identik dengan kanal sensor",
        "n_berkas": len(list(out.glob("*.png"))),
        "n_dikerjakan_sesi_ini": len(citra),  # statistik meter di bawah hanya dari sesi ini
        "meter": {
            "min_rerata": round(float(s[:, 0].mean()), 3),
            "median_rerata": round(float(s[:, 1].mean()), 3),
            "maks_rerata": round(float(s[:, 2].mean()), 3),
        },
        "citra_mayoritas_di_luar_rentang": {"lebih_dekat_dari_z_near": n_dekat,
                                            "lebih_jauh_dari_z_far": n_jauh},
    }
    (out / "mono_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["meter"], indent=2))
    print(f"selesai: {len(citra)} berkas + mono_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
