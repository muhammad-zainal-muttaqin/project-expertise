"""Fusi CLASS-AWARE lintas-jalur: dua-tahap (Fase 6) + detektor end-to-end (Fase 1-5).

Dua jalur ini memberi kelas dengan mekanisme yang sama sekali berbeda:

  jalur A  detektor class-agnostic -> crop -> classifier kematangan (ConvNeXt)
  jalur B  detektor 4-kelas end-to-end (YOLO26l), termasuk varian RGB+D `edge`

Karena galatnya tidak berkorelasi penuh, fusi per-kelas bisa melewati keduanya.
Bukti komplementaritas pada test (V2-E-020 vs V2-E-010): jalur A unggul di B3
(0,3212 vs 0,2240), jalur B unggul di B2 (0,5031 vs 0,4683).

PEMILIHAN KONFIGURASI SELALU DI SPLIT VAL. Test hanya dievaluasi sekali dengan
konfigurasi yang sudah terkunci dari val — supaya angka test tetap sah.

Usage:
    .venv/bin/python scripts/fuse_final.py \
        --sumber results/twostage_final_v6.pred.npz results/pred_edge.npz \
        --nama dua_tahap edge_rgbd \
        --out results/fusi_final.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_twostage import ap50, iou_mat  # noqa: E402

D352 = Path("/workspace/SawitMVC-Depth")
SPLIT = D352 / "splits" / "canonical_70_15_15"
W, H = 1280, 800
K = 4


def muat_gt(split: str):
    stems = [Path(l.strip()).stem
             for l in (SPLIT / f"{split}.txt").read_text().splitlines() if l.strip()]
    gt = {}
    for s in stems:
        g = []
        for ln in (D352 / "labels" / f"{s}.txt").read_text().splitlines():
            p = ln.split()
            if len(p) < 5 or int(p[0]) < 0:
                continue
            c = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
            g.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H])
        gt[s] = np.array(g, float) if g else np.zeros((0, 5))
    return stems, gt


def muat_dump(jalur: str) -> dict:
    z = np.load(jalur)
    return {k: np.asarray(z[k], float) for k in z.files}


def fuse_gambar(per_sumber: list[np.ndarray], bobot: np.ndarray,
                iou_th: float, mode: str) -> np.ndarray:
    """Fusi kotak SATU kelas pada SATU citra. Tiap elemen: Nx5 (xyxy+skor).

    Kotak dikelompokkan greedy berdasarkan IoU terhadap anggota pertama gugus
    (skor tertinggi lebih dulu), lalu tiap gugus diringkas jadi satu kotak.
    """
    baris, src = [], []
    for i, a in enumerate(per_sumber):
        if len(a):
            baris.append(a[:, :5]); src.append(np.full(len(a), i))
    if not baris:
        return np.zeros((0, 5))
    kotak = np.concatenate(baris, 0)
    sidx = np.concatenate(src, 0).astype(int)

    # urutkan pakai skor BERBOBOT supaya sumber kuat memimpin gugus
    skor_b = kotak[:, 4] * bobot[sidx]
    urut = np.argsort(-skor_b)
    kotak, sidx = kotak[urut], sidx[urut]

    gugus_kotak: list[list[np.ndarray]] = []
    gugus_src: list[list[int]] = []
    for k, si in zip(kotak, sidx):
        for gk, gs in zip(gugus_kotak, gugus_src):
            if iou_mat(k[None, :4], gk[0][None, :4])[0, 0] >= iou_th:
                gk.append(k); gs.append(si)
                break
        else:
            gugus_kotak.append([k]); gugus_src.append([si])

    total_bobot = bobot.sum()
    keluar = []
    for gk, gs in zip(gugus_kotak, gugus_src):
        a = np.stack(gk)
        wb = bobot[np.array(gs)]
        wpos = a[:, 4] * wb                      # bobot koordinat: skor x bobot sumber
        xy = (a[:, :4] * wpos[:, None]).sum(0) / max(wpos.sum(), 1e-9)
        if mode == "max":
            sk = float((a[:, 4] * wb).max() / bobot.max())
        elif mode == "mean":                     # rata-rata berbobot ANGGOTA saja
            sk = float(wpos.sum() / max(wb.sum(), 1e-9))
        else:                                    # "wbf": sumber yang absen dihitung 0
            uniq = {}
            for s_, k_ in zip(gs, a[:, 4]):      # satu suara per sumber (yang tertinggi)
                uniq[s_] = max(uniq.get(s_, 0.0), float(k_))
            sk = sum(bobot[s_] * v for s_, v in uniq.items()) / total_bobot
        keluar.append([*xy, sk])
    return np.array(keluar, float)


def evaluasi(dumps: list[dict], stems: list[str], gt: dict, cfg):
    """cfg = satu (bobot, iou, mode) untuk semua kelas, ATAU daftar K konfigurasi.

    Bobot per-kelas penting karena kekuatan tiap sumber berbeda drastis antar
    kelas: pada uji 2-sumber, `edge` jauh unggul di B4 (0,2711 vs 0,1508)
    sehingga bobot global memaksa mencampur sumber lemah dan merusak B4.
    """
    if not isinstance(cfg, list):
        cfg = [cfg] * K
    pred = {s: [] for s in stems}
    for c in range(K):
        bobot, iou_th, mode = cfg[c]
        for s in stems:
            per_sumber = []
            for d in dumps:
                a = d.get(s, np.zeros((0, 6)))
                per_sumber.append(a[a[:, 5] == c] if len(a) else a)
            f = fuse_gambar(per_sumber, bobot, iou_th, mode)
            if len(f):
                pred[s].append(np.concatenate([f, np.full((len(f), 1), c)], 1))
    pred = {s: (np.concatenate(v, 0) if v else np.zeros((0, 6))) for s, v in pred.items()}
    per = [ap50(gt, pred, c) for c in range(K)]
    return float(np.mean(per)), per, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", nargs="+", required=True,
                    help="npz dump per sumber, untuk split VAL (pemilihan)")
    ap.add_argument("--sumber-test", nargs="+",
                    help="npz dump yang SAMA urutannya untuk split test (angka akhir)")
    ap.add_argument("--nama", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.sumber) != len(args.nama):
        print("FATAL: jumlah --sumber dan --nama harus sama"); return 1
    if args.sumber_test and len(args.sumber_test) != len(args.sumber):
        print("FATAL: jumlah --sumber-test harus sama dengan --sumber"); return 1
    n = len(args.sumber)

    stems_v, gt_v = muat_gt("val")
    dumps_v = [muat_dump(j) for j in args.sumber]

    print("=== baseline tiap sumber di VAL ===")
    dasar = {}
    for nm, d in zip(args.nama, dumps_v):
        per = [ap50(gt_v, {k: v for k, v in d.items()}, c) for c in range(K)]
        dasar[nm] = {"mAP50": round(float(np.mean(per)), 4),
                     "per_kelas": {f"B{i+1}": round(float(per[i]), 4) for i in range(K)}}
        print(f"  {nm:14s} {dasar[nm]['mAP50']:.4f}  {dasar[nm]['per_kelas']}")

    # --- sweep di VAL saja ------------------------------------------------
    # Kandidat "solo" (bobot one-hot) ikut disertakan supaya pemilihan per-kelas
    # boleh memutuskan bahwa untuk kelas tertentu fusi justru merugikan dan satu
    # sumber saja lebih baik.
    kandidat = []
    for i in range(n):
        b = np.zeros(n); b[i] = 1.0
        kandidat.append({"label": f"solo:{args.nama[i]}", "bobot": b,
                         "iou": 0.6, "mode": "wbf"})
    for wx, iou_th, mode in itertools.product([0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                                              [0.5, 0.6, 0.7], ["wbf", "mean", "max"]):
        kandidat.append({"label": f"w={wx}", "bobot": np.array([1.0] + [wx] * (n - 1)),
                         "iou": iou_th, "mode": mode})

    riwayat = []
    print(f"\n=== sweep {len(kandidat)} konfigurasi fusi (VAL) ===")
    for kd in kandidat:
        m, per, _ = evaluasi(dumps_v, stems_v, gt_v, (kd["bobot"], kd["iou"], kd["mode"]))
        riwayat.append({"label": kd["label"], "bobot": [round(float(x), 3) for x in kd["bobot"]],
                        "iou": kd["iou"], "mode": kd["mode"], "mAP50_val": round(m, 4),
                        "per_kelas": {f"B{i+1}": round(float(per[i]), 4) for i in range(K)},
                        "_cfg": (kd["bobot"], kd["iou"], kd["mode"])})
    riwayat.sort(key=lambda r: -r["mAP50_val"])
    for r in riwayat[:6]:
        print(f"  {r['label']:<22} iou={r['iou']} mode={r['mode']:<5} -> {r['mAP50_val']:.4f}")
    terbaik = riwayat[0]
    print(f"\nTERPILIH GLOBAL (val): {terbaik['label']} iou={terbaik['iou']} "
          f"mode={terbaik['mode']} -> {terbaik['mAP50_val']:.4f}")

    # Pemilihan PER-KELAS: gratis, karena tiap konfigurasi sudah menyimpan AP
    # tiap kelas. Perhatian jujur: ini 4 parameter yang dipilih di val (bukan 1),
    # jadi risiko overfit val lebih besar — terutama B4 yang instansnya paling
    # sedikit. Dilaporkan terpisah, bukan menggantikan angka global.
    cfg_kelas, pilihan_kelas = [], []
    for c in range(K):
        r = max(riwayat, key=lambda r: r["per_kelas"][f"B{c+1}"])
        cfg_kelas.append(r["_cfg"])
        pilihan_kelas.append({"kelas": f"B{c+1}", "label": r["label"], "iou": r["iou"],
                              "mode": r["mode"], "AP50_val": r["per_kelas"][f"B{c+1}"]})
        print(f"  per-kelas B{c+1}: {r['label']:<22} iou={r['iou']} mode={r['mode']:<5} "
              f"-> {r['per_kelas'][f'B{c+1}']:.4f}")
    m_pk_val = float(np.mean([p["AP50_val"] for p in pilihan_kelas]))
    print(f"  (batas atas val bobot per-kelas: {m_pk_val:.4f})")

    for r in riwayat:
        r.pop("_cfg", None)
    hasil = {"n_sumber": n, "nama": args.nama, "sumber_val": args.sumber,
             "baseline_val": dasar, "sweep_val": riwayat, "terpilih": terbaik,
             "terpilih_per_kelas": pilihan_kelas}

    # --- terapkan SEKALI ke test -----------------------------------------
    if args.sumber_test:
        stems_t, gt_t = muat_gt("test")
        dumps_t = [muat_dump(j) for j in args.sumber_test]
        print("\n=== baseline tiap sumber di TEST ===")
        dasar_t = {}
        for nm, d in zip(args.nama, dumps_t):
            per = [ap50(gt_t, {k: v for k, v in d.items()}, c) for c in range(K)]
            dasar_t[nm] = {"mAP50": round(float(np.mean(per)), 4),
                           "per_kelas": {f"B{i+1}": round(float(per[i]), 4) for i in range(K)}}
            print(f"  {nm:14s} {dasar_t[nm]['mAP50']:.4f}  {dasar_t[nm]['per_kelas']}")

        hasil["baseline_test"] = dasar_t
        print("\n=== FUSI di TEST (konfigurasi terkunci dari val) ===")
        for nama_cfg, cfg in (("global", (np.array(terbaik["bobot"]), terbaik["iou"], terbaik["mode"])),
                              ("per_kelas", cfg_kelas)):
            m, per, pred = evaluasi(dumps_t, stems_t, gt_t, cfg)
            agn = ap50(gt_t, {k: v[np.argsort(-v[:, 4])] for k, v in pred.items()}, None)
            hasil[f"test_{nama_cfg}"] = {
                "mAP50": round(m, 4),
                "AP50_per_kelas": {f"B{i+1}": round(float(per[i]), 4) for i in range(K)},
                "AP50_class_agnostic": round(float(agn), 4)}
            print(f"  bobot {nama_cfg:9s} -> mAP50 = {m:.4f}  "
                  f"{hasil[f'test_{nama_cfg}']['AP50_per_kelas']}")
            np.savez_compressed(
                f"{str(Path(args.out).with_suffix(''))}_test_{nama_cfg}.pred.npz", **pred)

        terbaik_tunggal = max(d["mAP50"] for d in dasar_t.values())
        delta = hasil["test_global"]["mAP50"] - terbaik_tunggal
        print(f"  sumber tunggal terbaik di test = {terbaik_tunggal:.4f}; "
              f"fusi global {'+' if delta >= 0 else ''}{delta:.4f}")
        hasil["delta_vs_sumber_terbaik_test"] = round(delta, 4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hasil, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
