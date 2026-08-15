"""Bangun dataset deteksi N-kanal untuk matriks monocular-depth (V2-E-027 dst).

Satu skrip untuk seluruh sel matriks, supaya tidak ada perbedaan tak sengaja
antar-lengan perbandingan. Urutan kanal MENGIKUTI konvensi yang sudah ada
(`build_4ch_dataset.py`: TIFF [B,G,R,D]), jadi:

    rgb            -> [B,G,R]              3 kanal (tidak perlu skrip ini)
    rgb+edge       -> [B,G,R,edge]         4 kanal  = SawitMVC-Depth-4ch-edge
    rgb+mono       -> [B,G,R,mono]         4 kanal  BARU
    rgb+edge+mono  -> [B,G,R,edge,mono]    5 kanal  BARU

`edge` = magnitudo Sobel depth sensor (pemenang Fase 5), hanya ada untuk 352.
`mono` = monocular-depth ter-encode inverse-uint8 dengan kontrak yang sama
persis dengan kanal sensor (lihat `buat_mono_depth.py`).

MODE MONO — kontrol untuk membuktikan gain berasal dari keselarasan peta, bukan
dari sekadar tambahan kanal:

  asli        peta milik citra itu sendiri
  cross_tree  peta milik POHON LAIN (geser siklik daftar pohon, di dalam split
              masing-masing, tanpa titik tetap). Statistik nilai dan struktur
              spasialnya tetap peta depth yang sah; yang hilang hanya
              keselarasan dengan RGB. Ini kontrol utama.
  spatial     piksel peta sendiri diacak posisinya. Distribusi nilai identik
              persis, struktur spasial hancur total. Ini analisis sensitivitas,
              bukan kontrol utama — ia merusak keselarasan DAN kelayakan citra
              sekaligus, jadi penurunan tidak bisa diatribusikan ke satu sebab.

Geseran cross_tree dilakukan DI DALAM split (train donor train, test donor
test) supaya tidak ada informasi yang menyeberang antar-split.

Usage:
    # 352, class-aware 4 kelas, RGB+mono
    .venv/bin/python scripts/buat_dataset_nch.py --dataset 352 \
        --kanal mono --out /workspace/d352_rgbmono

    # 352, RGB+edge+mono 5 kanal
    .venv/bin/python scripts/buat_dataset_nch.py --dataset 352 \
        --kanal edge mono --out /workspace/d352_rgbedgemono

    # 953, RGB+mono
    .venv/bin/python scripts/buat_dataset_nch.py --dataset 953 \
        --kanal mono --out /workspace/d953_rgbmono

    # kontrol
    .venv/bin/python scripts/buat_dataset_nch.py --dataset 953 --kanal mono \
        --mono-mode cross_tree --out /workspace/d953_rgbmono_xtree
"""

from __future__ import annotations

import argparse
import json
import zlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Kedua dataset punya tata letak BERBEDA, dan ini bukan detail kosmetik:
# split yang dipakai harus sama persis dengan yang dipakai sel pembandingnya,
# kalau tidak angkanya tidak sebanding.
#   352 -> citra/label datar, split dari berkas .txt kanonik. Sudah
#          diverifikasi identik dengan splits_abs_rgb (sel 1) dan
#          SawitMVC-Depth-4ch-edge/splits_abs (sel 2).
#   953 -> citra/label SUDAH dimaterialisasi ke {train,val,test}/. Itulah yang
#          dibaca eval_all_pycoco_v2repro.py untuk menghasilkan angka sel 5
#          (0,5435). Jangan bangun ulang split dari split_manifest.csv —
#          kebetulan isinya cocok, tapi direktori ini yang jadi sumber
#          kebenaran karena dia yang benar-benar dievaluasi.
SUMBER = {
    "352": {
        "citra": Path("/workspace/SawitMVC-Depth/images"),
        "label": Path("/workspace/SawitMVC-Depth/labels"),
        "split_dir": Path("/workspace/SawitMVC-Depth/splits/canonical_70_15_15"),
        "edge_tiff": Path("/workspace/SawitMVC-Depth-4ch-edge/images"),
        "mono": Path("/workspace/mono_png_352"),
        "materialisasi": False,
    },
    "953": {
        "citra": Path("/workspace/SawitMVC-YOLO/images"),
        "label": Path("/workspace/SawitMVC-YOLO/labels"),
        "edge_tiff": None,
        "mono": Path("/workspace/mono_png_953"),
        "materialisasi": True,
    },
}
KELAS = ["B1", "B2", "B3", "B4"]


def pohon(stem: str) -> str:
    return "_".join(stem.split("_")[:-1])


def tampilan(stem: str) -> str:
    return stem.split("_")[-1]


def baca_split(S: dict) -> dict[str, list[str]]:
    """stem -> split, dari sumber kebenaran masing-masing dataset (lihat SUMBER)."""
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sp in out:
        if S["materialisasi"]:
            d = S["citra"] / sp
            if not d.is_dir():
                sys.exit(f"FATAL: {d} tidak ada")
            out[sp] = sorted(p.stem for p in d.iterdir() if p.suffix.lower() == ".jpg")
        else:
            fp = S["split_dir"] / f"{sp}.txt"
            if not fp.exists():
                sys.exit(f"FATAL: {fp} tidak ada")
            out[sp] = [Path(l.strip()).stem for l in fp.read_text().splitlines() if l.strip()]
        if not out[sp]:
            sys.exit(f"FATAL: split {sp} kosong")
    tumpang = set(out["train"]) & (set(out["val"]) | set(out["test"]))
    if tumpang:
        sys.exit(f"FATAL: {len(tumpang)} stem muncul di train DAN val/test")
    return out


def jalur_sumber(S: dict, sp: str, stem: str) -> tuple[Path, Path]:
    if S["materialisasi"]:
        return S["citra"] / sp / f"{stem}.jpg", S["label"] / sp / f"{stem}.txt"
    return S["citra"] / f"{stem}.jpg", S["label"] / f"{stem}.txt"


def peta_donor(stems: list[str]) -> dict[str, str]:
    """Geser siklik daftar pohon: pohon ke-i menyumbang peta ke pohon ke-(i+1).

    Tampilan dicocokkan (sisi 1 dapat sisi 1) supaya geometri kamera tetap
    masuk akal; kalau donor tidak punya tampilan itu, ambil tampilan pertamanya.
    """
    per_pohon: dict[str, dict[str, str]] = defaultdict(dict)
    for s in stems:
        per_pohon[pohon(s)][tampilan(s)] = s
    daftar = sorted(per_pohon)
    if len(daftar) < 2:
        sys.exit("FATAL: butuh >=2 pohon untuk cross_tree")
    donor = {}
    for i, t in enumerate(daftar):
        d = daftar[(i + 1) % len(daftar)]
        for v, s in per_pohon[t].items():
            kandidat = per_pohon[d]
            donor[s] = kandidat.get(v) or kandidat[sorted(kandidat)[0]]
    assert all(pohon(k) != pohon(v) for k, v in donor.items()), "cross_tree punya titik tetap"
    return donor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["352", "953"], required=True)
    ap.add_argument("--kanal", nargs="+", choices=["edge", "mono"], required=True,
                    help="kanal tambahan setelah B,G,R — urutannya menentukan urutan kanal")
    ap.add_argument("--mono-mode", choices=["asli", "cross_tree", "spatial"], default="asli")
    ap.add_argument("--agnostik", action="store_true", help="lipat 4 kelas jadi 1 (`tandan`)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42, help="hanya dipakai mode spatial")
    ap.add_argument("--paksa", action="store_true",
                    help="tulis ulang TIFF yang sudah ada (default: lanjutkan, lewati yang ada)")
    args = ap.parse_args()

    S = SUMBER[args.dataset]
    if "edge" in args.kanal and S["edge_tiff"] is None:
        sys.exit(f"FATAL: dataset {args.dataset} tidak punya depth sensor — kanal `edge` mustahil")
    for k, p in (("citra", S["citra"]), ("label", S["label"]), ("mono", S["mono"])):
        if k == "mono" and "mono" not in args.kanal:
            continue
        if not p.is_dir():
            sys.exit(f"FATAL: {p} tidak ada ({k})")

    # Tata letak: images/{split}/ dan labels/{split}/, PLUS rak symlink
    # {split}/images -> images/{split}. Ini bukan selera, ini keharusan:
    # proyek ini punya DUA konvensi eval yang berbeda dan keduanya harus bisa
    # dipakai tanpa menulis evaluator baru —
    #   eval_all_pycoco_v2repro.py (sel 5, 953) membaca  ds_root/images/{split}
    #   eval_pycoco_352.py / _rgbd352.py (sel 1, 2)      membaca ds_root/{split}/images
    # Ultralytics sendiri memetakan /images/ -> /labels/ saat latih, jadi tata
    # letak utama juga sah untuk training lewat daftar .txt.
    akar = Path(args.out)
    (akar / "splits").mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        (akar / "images" / sp).mkdir(parents=True, exist_ok=True)
        (akar / "labels" / sp).mkdir(parents=True, exist_ok=True)

    split = baca_split(S)
    n_kanal = 3 + len(args.kanal)
    donor = {}
    if "mono" in args.kanal and args.mono_mode == "cross_tree":
        for sp, stems in split.items():
            donor.update(peta_donor(stems))

    ringkas, hilang = {}, []
    for sp, stems in split.items():
        jalur, n_box = [], 0
        for stem in stems:
            src_img, src_lbl = jalur_sumber(S, sp, stem)
            if not (src_img.exists() and src_lbl.exists()):
                hilang.append(stem)
                continue

            dst = akar / "images" / sp / f"{stem}.tiff"
            dst_lbl = akar / "labels" / sp / f"{stem}.txt"

            # Label SELALU ditulis ulang dari sumber, tidak pernah dipercaya dari
            # keluaran sebelumnya. Ini menutup kegagalan senyap: kalau proses
            # terbunuh di antara penulisan citra dan penulisan label, jalur
            # resume yang mempercayai label lama akan meninggalkan citra dengan
            # label kosong — ultralytics memperlakukannya sebagai latar tanpa
            # objek, jadi datanya salah TANPA pesan error apa pun. Menulis ulang
            # label itu murah; salah diam-diam itu mahal.
            baris = []
            for ln in src_lbl.read_text().splitlines():
                p = ln.split()
                if len(p) < 5 or int(p[0]) < 0:
                    continue
                c = 0 if args.agnostik else int(p[0])
                baris.append(" ".join([str(c), *p[1:5]]))
            dst_lbl.write_text("\n".join(baris) + ("\n" if baris else ""))
            n_box += len(baris)

            if dst.exists() and dst.stat().st_size > 0 and not args.paksa:
                jalur.append(str(dst))  # citra sudah ada, cukup lewati encoding-nya
                continue

            bgr = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
            if bgr is None:
                sys.exit(f"FATAL: gagal membaca {src_img}")
            h, w = bgr.shape[:2]
            tumpuk = [bgr]

            for k in args.kanal:
                if k == "edge":
                    t = cv2.imread(str(S["edge_tiff"] / f"{stem}.tiff"), cv2.IMREAD_UNCHANGED)
                    if t is None or t.ndim != 3 or t.shape[2] < 4:
                        sys.exit(f"FATAL: TIFF edge {stem} tidak punya kanal ke-4")
                    ch = t[:, :, 3]
                else:
                    asal = donor.get(stem, stem) if args.mono_mode == "cross_tree" else stem
                    fp = S["mono"] / f"{asal}.png"
                    if not fp.exists():
                        sys.exit(f"FATAL: peta mono {fp} tidak ada — jalankan buat_mono_depth.py dulu")
                    ch = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                    if ch is None:
                        sys.exit(f"FATAL: gagal membaca {fp}")
                    if args.mono_mode == "spatial":
                        # RNG diturunkan per-citra dari seed + stem, BUKAN satu
                        # aliran global: hasilnya tidak bergantung urutan proses,
                        # jadi tetap identik kalau pembangunan dilanjut setelah
                        # terputus. Kontrol harus reproducible.
                        r_px = np.random.default_rng(args.seed + zlib.crc32(stem.encode()))
                        bentuk = ch.shape
                        ch = r_px.permutation(ch.ravel()).reshape(bentuk)
                if ch.shape[:2] != (h, w):
                    ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_NEAREST)
                tumpuk.append(ch[:, :, None] if ch.ndim == 2 else ch)

            keluar = np.dstack(tumpuk).astype(np.uint8)
            if keluar.shape[2] != n_kanal:
                sys.exit(f"FATAL: {stem} menghasilkan {keluar.shape[2]} kanal, harusnya {n_kanal}")
            # cv2.imwrite HANYA sanggup 1/3/4 kanal — 5 kanal melempar
            # cv2.error, bukan mengembalikan False. Untuk 5+ kanal simpan
            # sebagai TIFF MULTI-HALAMAN: ultralytics membacanya lewat
            # cv2.imdecodemulti(IMREAD_UNCHANGED) lalu np.stack(frames, axis=2),
            # jadi hasilnya (H,W,5) dengan urutan kanal terjaga (diverifikasi
            # roundtrip). 4 kanal tetap satu halaman supaya byte-compatible
            # dengan SawitMVC-Depth-4ch-edge yang jadi pembanding sel 2.
            if n_kanal > 4:
                ok = cv2.imwritemulti(str(dst), [keluar[:, :, c] for c in range(n_kanal)])
            else:
                ok = cv2.imwrite(str(dst), keluar)
            if not ok:
                sys.exit(f"FATAL: gagal menulis {dst}")

            jalur.append(str(dst))  # label sudah ditulis di atas, sebelum cek resume

        (akar / "splits" / f"{sp}.txt").write_text("\n".join(jalur) + "\n")
        ringkas[sp] = {"citra": len(jalur), "pohon": len({pohon(Path(j).stem) for j in jalur}),
                       "kotak": n_box}
        print(f"{sp:6s} citra={len(jalur):5d} pohon={ringkas[sp]['pohon']:4d} kotak={n_box:6d}")

    # Rak symlink untuk konvensi eval yang satunya ({split}/images).
    # Symlink, bukan salinan — 5-kanal TIFF terlalu besar untuk digandakan.
    for sp in ("train", "val", "test"):
        (akar / sp).mkdir(exist_ok=True)
        for jenis in ("images", "labels"):
            tautan = akar / sp / jenis
            if tautan.is_symlink() or tautan.exists():
                tautan.unlink()
            tautan.symlink_to(akar / jenis / sp)

    nc = 1 if args.agnostik else 4
    nama = ["tandan"] if args.agnostik else KELAS
    (akar / "data.yaml").write_text("\n".join([
        f"path: {akar}",
        f"train: {akar}/splits/train.txt",
        f"val: {akar}/splits/val.txt",
        f"test: {akar}/splits/test.txt",
        f"channels: {n_kanal}",   # WAJIB: tanpa ini ultralytics memuat 3 kanal saja
        f"nc: {nc}",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(nama)],
    ]) + "\n")

    meta = {"dataset": args.dataset, "kanal": ["B", "G", "R", *args.kanal],
            "n_kanal": n_kanal, "mono_mode": args.mono_mode, "agnostik": args.agnostik,
            "seed": args.seed, "split": ringkas, "stem_hilang": hilang[:20],
            "n_stem_hilang": len(hilang)}
    (akar / "meta.json").write_text(json.dumps(meta, indent=2))
    if hilang:
        print(f"PERINGATAN: {len(hilang)} stem dilewati (citra/label tidak lengkap)")
    print(f"selesai -> {akar} ({n_kanal} kanal, mode mono={args.mono_mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
