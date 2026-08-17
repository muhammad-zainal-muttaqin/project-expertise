"""PT-E-001 — Rangka evaluasi per-TANDAN + plafon tautan oracle.

Menjawab gerbang G0 (`docs/PROPOSAL.md` §7): dengan tautan lintas-sisi yang
SEMPURNA (diambil dari GT), apakah menggabungkan tampak benar-benar menaikkan
akurasi kelas per tandan dibanding melihat satu foto saja?

Kalau dengan tautan sempurna pun tidak naik, membangun penaut tidak ada
gunanya, dan separuh-kelas dari proposal gugur di sini — sebelum satu jam GPU
pun terpakai.

## Definisi satuan

- **kemunculan** (appearance) = satu kotak GT di satu citra. Ini satuan lama,
  yang dipakai mAP dan seluruh evaluasi Fase 1-6.
- **tandan** (bunch) = satu buah fisik di pohon, bisa terlihat di 1-6 sisi.
  Ini satuan baru. Kelasnya satu, dan GT-nya konsisten di semua sisi
  (`class_mismatch` = 0 untuk seluruh 9.823 tandan).

## Himpunan evaluasi

Semua aturan dinilai pada himpunan yang SAMA: tandan yang punya >=1 deteksi
tercocokkan (IoU>=0,5). Tandan yang tidak terdeteksi sama sekali adalah
kegagalan DETEKSI, bukan kegagalan aturan agregasi — dilaporkan terpisah
sebagai cakupan, tidak dicampur ke akurasi.

## Aturan yang dibandingkan

  R0    satu tampak, argmax       <- baseline "tanpa pipeline": Anda cuma punya satu foto
  R0cal satu tampak, ambang ordinal
  R1    keyakinan tertinggi       <- usulan awal
  R2    argmax rerata softmax     <- yang sudah dicoba E-016 (+0,66 pp)
  R3    argmax rerata berbobot mutu
  R4    ekspektasi ordinal        <- rekomendasi proposal; ambang dilatih di val

## Kenapa R0cal ada — koreksi terhadap rancangan awal

Versi pertama rangka ini membandingkan R4 langsung ke R0 dan mendapat +4,9 pp
di val. Angka itu MENYESATKAN: R4 juga menaikkan akurasi pool yang cuma punya
SATU tampak (val 0,6917 -> 0,7222), padahal di sana tidak ada penggabungan
apa pun. Artinya sebagian "gain" itu berasal dari **rekalibrasi ambang kelas**,
bukan dari multi-tampak.

Karena itu selisihnya dipecah dua:

    R0cal - R0   = untung dari REKALIBRASI (bisa didapat tanpa pipeline ini)
    R4    - R0cal = untung dari PENGGABUNGAN (yang pipeline ini benar-benar klaim)

Gerbang G0 dinilai pada suku kedua saja.

## Baseline satu-tampak dihitung sebagai EKSPEKTASI, bukan satu undian

Mengundi satu tampak secara acak menambahkan derau sampling yang besarnya
sebanding dengan efek yang dicari (terlihat: selisih R4-R0 berubah dari 2,38 pp
menjadi 1,03 pp hanya karena undian berbeda). Jadi R0 dan R0cal dihitung
sebagai rata-rata kebenaran atas SELURUH tampak dalam pool — itu persis nilai
harapan "ambil satu foto acak", tanpa derau.

Vektor kelas datang dari `infer_skor_penuh.py` (cabang one2one YOLO26,
4 logit per anchor -> sigmoid -> dinormalkan jadi distribusi).

## Protokol

Pemilihan (skema bobot R3, ambang R4) SELALU di val. Test dievaluasi sekali
dengan konfigurasi yang sudah dikunci. Bootstrap di tingkat POHON, bukan
tandan — tandan dalam satu pohon berkorelasi kuat.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/eval_pertandan.py --conf 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DS = Path("/workspace/SawitMVC-YOLO")
SUB = Path(__file__).resolve().parents[1]
KELAS = ["B1", "B2", "B3", "B4"]
K = 4
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# pemuatan
# --------------------------------------------------------------------------
def muat_manifest() -> dict[str, str]:
    man = {}
    with (DS / "split_manifest.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            man[r["tree_id"]] = r["new_split"]
    return man


def muat_pohon(tree: str) -> dict:
    d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
    b2 = {}
    for b in d["bunches"]:
        for ap in b["appearances"]:
            b2[(ap["side_index"], ap["box_index"])] = b["bunch_id"]
    sisi = []
    for im in d["images"].values():
        sisi.append({
            "stem": im["filename"].rsplit(".", 1)[0],
            "si": im["side_index"],
            "wh": (im["width"], im["height"]),
            "gt": [{"bi": a["box_index"], "kelas": KELAS.index(a["class_name"]),
                    "box": np.array(a["bbox_pixel"], float),
                    "bid": b2.get((im["side_index"], a["box_index"]))}
                   for a in im["annotations"]],
        })
    return {"tree": tree, "n_sisi": len(sisi), "sisi": sisi,
            "tandan": {b["bunch_id"]: KELAS.index(b["class"]) for b in d["bunches"]},
            "n_app": {b["bunch_id"]: b["appearance_count"] for b in d["bunches"]}}


def iou_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (a[:, None] + b[None, :] - inter + 1e-9)


# --------------------------------------------------------------------------
# pencocokan deteksi -> kemunculan GT
# --------------------------------------------------------------------------
def cocokkan_pohon(P: dict, z, conf: float, iou_min: float = 0.5) -> list[dict]:
    """Kembalikan daftar deteksi tercocokkan, masing-masing membawa bunch_id GT.

    Serakah menurut keyakinan menurun, satu-ke-satu (satu kotak GT dipakai
    sekali). Deteksi yang tidak cocok ke GT mana pun dibuang di sini — ia
    positif palsu, urusan metrik deteksi, bukan metrik agregasi kelas.
    """
    hasil = []
    for s in P["sisi"]:
        D = z[s["stem"]] if s["stem"] in z.files else np.zeros((0, 11))
        if len(D):
            # satu anchor = satu kotak = satu distribusi. YOLO26 memancarkan
            # beberapa baris per anchor (satu per kelas yang lolos), jadi baris
            # duplikat harus disatukan dulu — kalau tidak, satu tandan yang sama
            # masuk pool berkali-kali dan menggandakan suaranya.
            _, uniq = np.unique(D[:, 10], return_index=True)
            D = D[np.sort(uniq)]
            D = D[D[:, 6:10].max(1) >= conf]
        if len(D) == 0 or not s["gt"]:
            continue
        G = np.stack([g["box"] for g in s["gt"]])
        M = iou_mat(D[:, :4], G)
        urut = np.argsort(-D[:, 6:10].max(1))
        dipakai = set()
        w, h = s["wh"]
        for di in urut:
            kand = [(M[di, gi], gi) for gi in range(len(G))
                    if gi not in dipakai and M[di, gi] >= iou_min]
            if not kand:
                continue
            _, gi = max(kand)
            dipakai.add(gi)
            g = s["gt"][gi]
            if g["bid"] is None:
                continue
            p = D[di, 6:10].astype(float)
            box = D[di, :4]
            luas = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            tepi = min(box[0], box[1], w - box[2], h - box[3]) / max(w, h)
            hasil.append({
                "bid": g["bid"], "si": s["si"], "gt_kelas": g["kelas"],
                "p": p / max(p.sum(), 1e-9),          # dinormalkan jadi distribusi
                "conf": float(p.max()),
                "luas": float(luas), "tepi": float(max(tepi, 0.0)),
            })
    return hasil


# --------------------------------------------------------------------------
# aturan agregasi
# --------------------------------------------------------------------------
def bobot_mutu(pool: list[dict], skema: str) -> np.ndarray:
    c = np.array([d["conf"] for d in pool])
    a = np.array([d["luas"] for d in pool])
    t = np.array([d["tepi"] for d in pool])
    if skema == "conf":
        w = c
    elif skema == "conf_luas":
        w = c * np.sqrt(a + 1e-9)
    elif skema == "conf_luas_tepi":
        w = c * np.sqrt(a + 1e-9) * (0.5 + t)
    elif skema == "luas":
        w = np.sqrt(a + 1e-9)
    else:
        w = np.ones(len(pool))
    return w / max(w.sum(), 1e-9)


def benar(pool: list[dict], gt: int, aturan: str, skema: str = "conf_luas",
          tau: tuple[float, float, float] = (0.5, 1.5, 2.5)) -> float:
    """Kebenaran sebuah pool di bawah satu aturan.

    Mengembalikan float, bukan bool: untuk baseline satu-tampak nilainya adalah
    FRAKSI tampak yang benar — nilai harapan "ambil satu foto acak", tanpa
    derau undian.
    """
    Pm = np.stack([d["p"] for d in pool])
    t = np.asarray(tau)
    if aturan == "R0":                                  # satu tampak, argmax
        return float((Pm.argmax(1) == gt).mean())
    if aturan == "R0cal":                               # satu tampak, ambang ordinal
        gv = Pm @ np.arange(K)
        return float((np.searchsorted(t, gv) == gt).mean())
    if aturan == "R1":
        return float(int(np.argmax(Pm[int(np.argmax([d["conf"] for d in pool]))])) == gt)
    if aturan == "R2":
        return float(int(np.argmax(Pm.mean(0))) == gt)
    if aturan == "R3":
        return float(int(np.argmax((bobot_mutu(pool, skema)[:, None] * Pm).sum(0))) == gt)
    if aturan == "R4":
        return float(int(np.searchsorted(t, skor_ordinal(pool, skema))) == gt)
    raise ValueError(aturan)


def prediksi(pool: list[dict], aturan: str, skema: str, tau) -> int:
    """Kelas tunggal terprediksi (untuk matriks per-kelas). Aturan ekspektasi
    R0/R0cal dipetakan ke tampak berkeyakinan tertinggi supaya tetap terdefinisi."""
    Pm = np.stack([d["p"] for d in pool])
    t = np.asarray(tau)
    j = int(np.argmax([d["conf"] for d in pool]))
    if aturan == "R0":
        return int(np.argmax(Pm[j]))
    if aturan == "R0cal":
        return int(np.searchsorted(t, float(Pm[j] @ np.arange(K))))
    if aturan == "R1":
        return int(np.argmax(Pm[j]))
    if aturan == "R2":
        return int(np.argmax(Pm.mean(0)))
    if aturan == "R3":
        return int(np.argmax((bobot_mutu(pool, skema)[:, None] * Pm).sum(0)))
    if aturan == "R4":
        return int(np.searchsorted(t, skor_ordinal(pool, skema)))
    raise ValueError(aturan)


def skor_ordinal(pool: list[dict], skema: str) -> float:
    Pm = np.stack([d["p"] for d in pool])
    w = bobot_mutu(pool, skema)
    return float((w[:, None] * Pm).sum(0) @ np.arange(K))


# --------------------------------------------------------------------------
# evaluasi
# --------------------------------------------------------------------------
def bangun_pool(pohon: list[dict], z, conf: float,
                klaster: dict | None = None) -> list[dict]:
    """Satu entri per pool. `klaster` = None berarti tautan ORACLE (bunch_id GT).

    Kalau `klaster` diberikan (PT-E-003), ia memetakan (tree, si, bid) -> id pool
    hasil penaut; kebenaran pool diambil dari kelas GT mayoritas anggotanya.
    """
    keluar = []
    for P in pohon:
        det = cocokkan_pohon(P, z, conf)
        if klaster is None:
            g = defaultdict(list)
            for d in det:
                g[d["bid"]].append(d)
            for bid, anggota in g.items():
                keluar.append({"tree": P["tree"], "kunci": bid,
                               "gt": P["tandan"][bid], "n_app": P["n_app"][bid],
                               "pool": anggota})
        else:
            g = defaultdict(list)
            for d in det:
                cid = klaster.get((P["tree"], d["si"], d["bid"]))
                if cid is None:
                    continue
                g[cid].append(d)
            for cid, anggota in g.items():
                kelas = [a["gt_kelas"] for a in anggota]
                keluar.append({"tree": P["tree"], "kunci": cid,
                               "gt": max(set(kelas), key=kelas.count),
                               "n_app": len(anggota), "pool": anggota})
    return keluar


def nilai(pools: list[dict], aturan: str, skema: str, tau) -> dict:
    if not pools:
        return {"akurasi": None, "n": 0}
    y = np.array([p["gt"] for p in pools])
    c = np.array([benar(p["pool"], p["gt"], aturan, skema, tau) for p in pools])
    yh = np.array([prediksi(p["pool"], aturan, skema, tau) for p in pools])
    f1, rec = [], {}
    for k in range(K):
        tp = int(((yh == k) & (y == k)).sum()); fp = int(((yh == k) & (y != k)).sum())
        fn = int(((yh != k) & (y == k)).sum())
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        f1.append(2 * pr * rc / (pr + rc + 1e-9))
        rec[KELAS[k]] = round(rc, 4)
    return {"akurasi": round(float(c.mean()), 4),
            "akurasi_pm1": round(float((np.abs(y - yh) <= 1).mean()), 4),
            "mae_ordinal": round(float(np.abs(y - yh).mean()), 4),
            "macro_f1": round(float(np.mean(f1)), 4),
            "recall_per_kelas": rec, "n": len(pools)}


def cari_tau(pools: list[dict], skema: str) -> tuple[float, float, float]:
    """Ambang ordinal dilatih di val: cari (t1,t2,t3) yang memaksimalkan akurasi R4."""
    g = np.array([skor_ordinal(p["pool"], skema) for p in pools])
    y = np.array([p["gt"] for p in pools])
    kisi = np.round(np.arange(0.10, 3.00, 0.05), 2)
    terbaik, skor = (0.5, 1.5, 2.5), -1.0
    for t1 in kisi:
        for t2 in kisi[kisi > t1]:
            for t3 in kisi[kisi > t2]:
                a = float((np.searchsorted(np.array([t1, t2, t3]), g) == y).mean())
                if a > skor:
                    skor, terbaik = a, (float(t1), float(t2), float(t3))
    return terbaik


def bootstrap_pohon(pools: list[dict], aturan_a: str, aturan_b: str,
                    skema: str, tau, n: int = 2000) -> dict:
    """CI selisih akurasi (a - b), resampling di tingkat POHON. Deterministik."""
    if not pools:
        return {"delta_pp": None, "n_pohon": 0}
    per = defaultdict(list)
    for p in pools:
        per[p["tree"]].append(p)
    pohon = list(per)
    cache = {t: (sum(benar(p["pool"], p["gt"], aturan_a, skema, tau) for p in per[t]),
                 sum(benar(p["pool"], p["gt"], aturan_b, skema, tau) for p in per[t]),
                 len(per[t])) for t in pohon}
    rng = np.random.default_rng(42)
    d = []
    for _ in range(n):
        s = rng.choice(len(pohon), len(pohon), replace=True)
        A = B = N = 0.0
        for i in s:
            a, b, m = cache[pohon[i]]
            A += a; B += b; N += m
        d.append((A - B) / max(N, 1))
    d = np.array(d)
    return {"delta_pp": round(float(d.mean() * 100), 3),
            "ci95_pp": [round(float(np.percentile(d, 2.5) * 100), 3),
                        round(float(np.percentile(d, 97.5) * 100), 3)],
            "P_delta_gt_0": round(float((d > 0).mean()), 4),
            "n_pohon": len(pohon), "n_pool": len(pools)}


def cakupan(pohon: list[dict], pools: list[dict]) -> dict:
    total = sum(len(P["tandan"]) for P in pohon)
    app = sum(len(s["gt"]) for P in pohon for s in P["sisi"])
    app_kena = sum(len(p["pool"]) for p in pools)
    return {"n_tandan_gt": total, "n_tandan_terdeteksi": len(pools),
            "recall_per_tandan": round(len(pools) / total, 4),
            "n_kemunculan_gt": app, "n_kemunculan_terdeteksi": app_kena,
            "recall_per_kemunculan": round(app_kena / app, 4)}


def akurasi_per_kemunculan(pools: list[dict]) -> dict:
    """PEMBANDING PIPELINE LAMA: kelas dinilai per kotak per citra."""
    b = n = 0
    for p in pools:
        for d in p["pool"]:
            n += 1
            b += int(np.argmax(d["p"]) == d["gt_kelas"])
    return {"n_kemunculan": n, "akurasi": round(b / max(n, 1), 4)}


# --------------------------------------------------------------------------
ATURAN = ["R0", "R0cal", "R1", "R2", "R3", "R4"]


def blok_split(pohon, pools, skema, tau) -> dict:
    multi = [p for p in pools if len(p["pool"]) >= 2]
    blok = {"cakupan": cakupan(pohon, pools),
            "PEMBANDING_per_kemunculan": akurasi_per_kemunculan(pools),
            "n_pool": len(pools), "n_pool_multi_tampak": len(multi),
            "semua_pool": {a: nilai(pools, a, skema, tau) for a in ATURAN},
            "HANYA_pool_multi_tampak": {a: nilai(multi, a, skema, tau) for a in ATURAN}}
    per_n = defaultdict(list)
    for p in pools:
        per_n[min(len(p["pool"]), 4)].append(p)
    blok["menurut_jumlah_tampak"] = {
        str(k): {"n": len(v), **{a: nilai(v, a, skema, tau)["akurasi"]
                                 for a in ["R0", "R0cal", "R4"]}}
        for k, v in sorted(per_n.items())}
    blok["dekomposisi"] = {
        "rekalibrasi_R0cal_vs_R0": bootstrap_pohon(pools, "R0cal", "R0", skema, tau),
        "PENGGABUNGAN_R4_vs_R0cal": bootstrap_pohon(pools, "R4", "R0cal", skema, tau),
        "PENGGABUNGAN_multi_R4_vs_R0cal": bootstrap_pohon(multi, "R4", "R0cal", skema, tau),
        "total_R4_vs_R0": bootstrap_pohon(pools, "R4", "R0", skema, tau),
        "R2_vs_R0_gaya_E016": bootstrap_pohon(pools, "R2", "R0", skema, tau),
    }
    return blok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, nargs="+", default=[0.10, 0.15, 0.25])
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_001_oracle.json"))
    args = ap.parse_args()

    man = muat_manifest()
    pohon = {s: [muat_pohon(t) for t, v in man.items() if v == s] for s in ["val", "test"]}
    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ["val", "test"]}

    # --- sapuan conf: dipilih di VAL pada suku PENGGABUNGAN, bukan akurasi total
    sapuan, terbaik = {}, (None, -9.9, None, None)
    for c in args.conf:
        pv = bangun_pool(pohon["val"], z["val"], c)
        skema = max(["seragam", "conf", "luas", "conf_luas", "conf_luas_tepi"],
                    key=lambda sk: nilai(pv, "R3", sk, (0.5, 1.5, 2.5))["akurasi"])
        tau = cari_tau(pv, skema)
        multi = [p for p in pv if len(p["pool"]) >= 2]
        d = bootstrap_pohon(multi, "R4", "R0cal", skema, tau, n=500)
        sapuan[f"{c:.2f}"] = {"skema": skema, "tau": tau, "n_pool": len(pv),
                              "n_multi": len(multi),
                              "penggabungan_pp_multi": d["delta_pp"]}
        print(f"[val] conf={c:.2f} skema={skema} tau={tau} "
              f"pool={len(pv)} multi={len(multi)} penggabungan={d['delta_pp']:+.2f} pp")
        if d["delta_pp"] > terbaik[1]:
            terbaik = (c, d["delta_pp"], skema, tau)
    conf, _, skema, tau = terbaik
    print(f"\n[val] TERKUNCI: conf={conf} skema={skema} tau={tau}")

    pools = {s: bangun_pool(pohon[s], z[s], conf) for s in ["val", "test"]}
    hasil = {"tautan": "ORACLE (GT _confirmedLinks)",
             "detektor": "YOLO26l @1280 sel5 (models/yolo26l_e60_i1280_v2repro)",
             "sapuan_conf_val": sapuan,
             "conf_dikunci": conf, "skema_bobot_dikunci": skema, "tau_R4_dikunci": tau,
             "split": {s: blok_split(pohon[s], pools[s], skema, tau)
                       for s in ["val", "test"]}}

    g = hasil["split"]["val"]["dekomposisi"]["PENGGABUNGAN_multi_R4_vs_R0cal"]
    hasil["gerbang_G0"] = {
        "syarat": ("suku PENGGABUNGAN (R4 vs R0cal, pool multi-tampak) >= +2,0 pp "
                   "di val, CI95 tidak memuat nol"),
        "terukur_pp": g["delta_pp"], "ci95_pp": g["ci95_pp"],
        "putusan": ("LOLOS" if (g["delta_pp"] >= 2.0 and g["ci95_pp"][0] > 0) else "GUGUR"),
        "catatan": ("dinilai pada suku penggabungan saja; suku rekalibrasi "
                    "dilaporkan terpisah karena bisa didapat tanpa pipeline ini"),
    }

    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["gerbang_G0"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
