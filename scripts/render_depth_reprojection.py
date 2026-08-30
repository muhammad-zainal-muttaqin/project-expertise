"""Reproyeksikan satu berkas depth mentah (.raw, grid kamera depth 848x480) ke
bidang kamera warna (1280x800), untuk contoh visual pada laporan garis waktu.

Berkas `.raw` new763/SawitMVC-Depth disimpan uint16le milimeter pada resolusi
KAMERA DEPTH aslinya ("alignedTo": "depth" pada sidecar .json) — bukan pada
grid RGB. Skrip ini melakukan pipeline reproyeksi geometris penuh yang
dijelaskan pada `alignmentNote` sidecar tersebut:

    1. Deproyeksi tiap piksel depth valid ke titik 3D pada bidang kamera depth
       (pinhole murni; distorsi depth pada sidecar bernilai nol).
    2. Transformasi titik 3D dari bidang depth ke bidang warna memakai
       ekstrinsik (mRot, mTrans) pada sidecar.
    3. Proyeksikan titik 3D ke piksel warna memakai intrinsik + distorsi
       Brown-Conrady kamera warna (cv2.projectPoints).
    4. Scatter piksel far->near (Z menurun) ke kanvas 1280x800 sehingga
       titik terdekat menang secara alami pada tabrakan piksel (z-buffer
       tanpa loop eksplisit).
    5. Tutup celah resampling kecil (<=2 piksel) saja; wilayah tak-valid
       yang lebih besar (oklusi asli, depth invalid) TETAP ditandai kosong,
       bukan diisi seolah-olah terukur.

Jalankan:
    py -3 scripts/render_depth_reprojection.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RAW_PATH = Path(
    r"D:\Work\Assisten-Dosen\SawitMVC-Depth\SawitMVC-Depth-YOLO\test\depth\DAMIMAS_A21B_0012_1.raw"
)
JSON_PATH = RAW_PATH.with_suffix(".json")
RGB_PATH = RAW_PATH.parents[1] / "images" / f"{RAW_PATH.stem}.jpg"
LABEL_PATH = RAW_PATH.parents[1] / "labels" / f"{RAW_PATH.stem}.txt"
OUT_DIR = Path(
    r"C:\Users\Zainal\AppData\Local\Temp\claude\D--Work-Assisten-Dosen-project-expertise"
    r"\395755d6-7463-4ee1-a643-cd7ebbff3bbd\scratchpad\depth_example"
)

CLASS_NAMES = ["B1", "B2", "B3", "B4"]
CLASS_COLORS = [(90, 200, 60), (0, 200, 230), (0, 140, 255), (30, 30, 220)]


def parse_dump(dump: str) -> dict:
    """Parse ringkas string `CameraParam{...}` dari sidecar menjadi dict Python."""
    import re

    def ambil(nama_struct: str) -> str:
        i = dump.index(nama_struct)
        depth = 0
        j = dump.index("{", i)
        for k in range(j, len(dump)):
            if dump[k] == "{":
                depth += 1
            elif dump[k] == "}":
                depth -= 1
                if depth == 0:
                    return dump[j + 1 : k]
        raise ValueError(nama_struct)

    def angka(blok: str, kunci: str) -> float:
        m = re.search(rf"{kunci}=([-\d.Ee]+)", blok)
        return float(m.group(1))

    depth_in = ambil("mDepthIntrinsic")
    color_in = ambil("mColorIntrinsic")
    color_dist = ambil("mColorDistortion")
    extrinsic = ambil("mTransform")

    m_rot = re.search(r"mRot=\[([^\]]+)\]", extrinsic)
    m_trans = re.search(r"mTrans=\[([^\]]+)\]", extrinsic)
    rot = np.array([float(x) for x in m_rot.group(1).split(",")]).reshape(3, 3)
    trans = np.array([float(x) for x in m_trans.group(1).split(",")])

    return {
        "depth_intrinsic": {k: angka(depth_in, f"m{k[0].upper()}{k[1:]}") for k in ["fx", "fy", "cx", "cy"]},
        "color_intrinsic": {k: angka(color_in, f"m{k[0].upper()}{k[1:]}") for k in ["fx", "fy", "cx", "cy"]},
        "color_dist": {k: angka(color_dist, f"m{k.upper()}") for k in ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"]},
        "rot": rot,
        "trans": trans,
    }


def reproyeksi(raw_path: Path, json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(json_path.read_text())
    kal = parse_dump(meta["calibrationDump"])

    w_d, h_d = meta["width"], meta["height"]
    depth = np.fromfile(raw_path, dtype="<u2").reshape(h_d, w_d).astype(np.float64)
    valid = depth > 0
    v_idx, u_idx = np.nonzero(valid)
    z = depth[v_idx, u_idx]

    di = kal["depth_intrinsic"]
    x3 = (u_idx - di["cx"]) * z / di["fx"]
    y3 = (v_idx - di["cy"]) * z / di["fy"]
    pts_depth = np.stack([x3, y3, z], axis=1).astype(np.float64)

    ci = kal["color_intrinsic"]
    cam_mtx = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]])
    dc = kal["color_dist"]
    dist = np.array([dc["k1"], dc["k2"], dc["p1"], dc["p2"], dc["k3"], dc["k4"], dc["k5"], dc["k6"]])
    rvec, _ = cv2.Rodrigues(kal["rot"])
    tvec = kal["trans"].reshape(3, 1)

    proj, _ = cv2.projectPoints(pts_depth.reshape(-1, 1, 3), rvec, tvec, cam_mtx, dist)
    proj = proj.reshape(-1, 2)

    w_c, h_c = meta["rgbWidth"], meta["rgbHeight"]
    u_t = np.round(proj[:, 0]).astype(np.int64)
    v_t = np.round(proj[:, 1]).astype(np.int64)
    di_bounds = (u_t >= 0) & (u_t < w_c) & (v_t >= 0) & (v_t < h_c)
    u_t, v_t, z_t = u_t[di_bounds], v_t[di_bounds], z[di_bounds]

    order = np.argsort(-z_t)  # jauh -> dekat; penulisan terakhir (terdekat) menang
    flat_idx = (v_t * w_c + u_t)[order]
    z_sorted = z_t[order]
    depth_flat = np.zeros(w_c * h_c, dtype=np.float64)
    depth_flat[flat_idx] = z_sorted
    depth_aligned = depth_flat.reshape(h_c, w_c)

    mask = (depth_aligned > 0).astype(np.uint8) * 255
    return depth_aligned, mask


def warnai(depth_mm: np.ndarray, mask: np.ndarray, floor_mm: float, ceil_mm: float) -> np.ndarray:
    clipped = np.clip(depth_mm, floor_mm, ceil_mm)
    norm = ((clipped - floor_mm) / (ceil_mm - floor_mm) * 255).astype(np.uint8)
    # Tutup HANYA celah resampling sangat kecil (<=2 px); lubang oklusi asli tetap kosong.
    valid = mask > 0
    filled = cv2.inpaint(norm, (255 - mask), 2, cv2.INPAINT_NS)
    colored = cv2.applyColorMap(filled, cv2.COLORMAP_TURBO)
    colored[~valid & (cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0)] = (32, 32, 32)
    return colored


def gambar_gt(citra: np.ndarray, label_path: Path) -> np.ndarray:
    out = citra.copy()
    if not label_path.exists():
        return out
    h, w = citra.shape[:2]
    for ln in label_path.read_text().strip().splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = map(float, ln.split())
        c = int(c)
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), CLASS_COLORS[c], 3)
        cv2.putText(out, CLASS_NAMES[c], (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, CLASS_COLORS[c], 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_PATH)
    ap.add_argument("--rgb", type=Path, default=RGB_PATH)
    ap.add_argument("--label", type=Path, default=LABEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(args.raw.with_suffix(".json").read_text())
    depth_aligned, mask = reproyeksi(args.raw, args.raw.with_suffix(".json"))
    colored = warnai(depth_aligned, mask, meta["displayFloorMm"], meta["displayCeilingMm"])

    rgb = cv2.imread(str(args.rgb))
    rgb_gt = gambar_gt(rgb, args.label)

    cakupan = float((mask > 0).mean())
    print(f"Cakupan piksel valid setelah reproyeksi: {cakupan * 100:.1f}%")

    cv2.imwrite(str(args.out / "rgb_gt.jpg"), rgb_gt, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(args.out / "depth_reprojected.jpg"), colored, [cv2.IMWRITE_JPEG_QUALITY, 90])

    sisi = np.hstack([rgb_gt, colored])
    cv2.imwrite(str(args.out / "rgb_depth_sisi.jpg"), sisi, [cv2.IMWRITE_JPEG_QUALITY, 88])

    (args.out / "ringkasan.json").write_text(json.dumps({
        "berkas": args.raw.name,
        "cakupan_valid_persen": round(cakupan * 100, 2),
        "displayFloorMm": meta["displayFloorMm"],
        "displayCeilingMm": meta["displayCeilingMm"],
        "measuredPercentilesMm": meta.get("correction", {}).get("measuredPercentilesMm"),
    }, indent=2))
    print(f"Tersimpan di: {args.out}")


if __name__ == "__main__":
    main()
