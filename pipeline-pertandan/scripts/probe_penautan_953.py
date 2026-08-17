"""Probe kelayakan pipeline per-tandan pada SawitMVC 953 (vanilla).

Menyiapkan angka dasar untuk `pipeline-pertandan/docs/PROPOSAL.md`. Semuanya
dihitung dari dataset vanilla `/workspace/SawitMVC-YOLO` apa adanya, plus dump
prediksi detektor sel 5 yang sudah ada (dari repo induk:
`../results/pred_sel5_953_rgb_test.npz`, YOLO26l @1280, bobot
`../models/yolo26l_e60_i1280_v2repro/best.pt`).

Lima blok, semuanya cepat (CPU, ~2 menit):

  A. Struktur GT lintas-sisi   -- berapa tandan, berapa yang multi-sisi, plafon
                                  sisi per tandan, konsistensi kelas antar sisi.
  B. Integritas split          -- manifest vs folder vs field `split` di JSON.
  C. Biaya galat-gabung        -- kalau penaut salah menggabung dua kotak beda
                                  tandan, seberapa sering keduanya kebetulan
                                  sekelas (galat itu jadi tidak merusak kelas).
  D. Penaut geometri-saja      -- plafon penautan TANPA piksel, dua bentuk:
                                  (D1) ambang per-pasangan, (D2) penugasan
                                  global berkendala (Hungarian + plafon sisi).
  E. Selisih recall            -- recall per-KEMUNCULAN (satuan citra, cara
                                  lama) vs recall per-TANDAN (satuan pohon,
                                  cara pipeline ini). Plus laju ketidaksepakatan
                                  kelas antar sisi pada tandan yang sama.

Blok D melatih HistGradientBoosting pada pasangan kotak split train dan
mengevaluasinya di val; ambang dikunci di val lalu test dievaluasi SEKALI.

Pemakaian (dari akar repo `project-expertise`, venv-nya ada di sana):
    .venv/bin/python pipeline-pertandan/scripts/probe_penautan_953.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, precision_recall_curve, roc_auc_score

DS = Path("/workspace/SawitMVC-YOLO")
SUB = Path(__file__).resolve().parents[1]      # pipeline-pertandan/
REPO = SUB.parent                              # project-expertise/
NPZ_SEL5 = REPO / "results" / "pred_sel5_953_rgb_test.npz"
KELAS = ["B1", "B2", "B3", "B4"]
SEED = 0


# --------------------------------------------------------------------------
# pemuatan
# --------------------------------------------------------------------------
def muat_manifest() -> dict[str, str]:
    """tree_id -> split kanonik. `utf-8-sig`: berkasnya ber-BOM."""
    man = {}
    with (DS / "split_manifest.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            man[r["tree_id"]] = r["new_split"]
    return man


def muat_pohon(tree: str):
    """(n_sisi, daftar kotak GT). bid=None berarti kotak tak tertaut ke bunch."""
    d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
    b2 = {}
    for b in d["bunches"]:
        for ap in b["appearances"]:
            b2[(ap["side_index"], ap["box_index"])] = b["bunch_id"]
    kotak = []
    for im in d["images"].values():
        for a in im["annotations"]:
            cx, cy, w, h = a["bbox_yolo"]
            kotak.append(dict(s=im["side_index"], i=a["box_index"], c=a["class_id"],
                              cx=cx, cy=cy, w=w, h=h,
                              bid=b2.get((im["side_index"], a["box_index"]))))
    return len(d["images"]), kotak, d


# --------------------------------------------------------------------------
# A. struktur GT
# --------------------------------------------------------------------------
def blok_a(man):
    per = defaultdict(lambda: dict(pohon=0, kotak=0, tandan=0, multi=0,
                                   app=Counter(), nsisi=Counter()))
    app_glob, maks_per_nsisi = Counter(), defaultdict(Counter)
    mismatch = 0
    for tree, split in man.items():
        nv, _, d = muat_pohon(tree)
        r = per[split]
        r["pohon"] += 1
        r["kotak"] += d["summary"]["total_detections"]
        r["nsisi"][nv] += 1
        mx = 0
        for b in d["bunches"]:
            a = b["appearance_count"]
            r["tandan"] += 1
            r["app"][a] += 1
            r["multi"] += a >= 2
            app_glob[a] += 1
            mismatch += bool(b.get("class_mismatch"))
            mx = max(mx, a)
        maks_per_nsisi[nv][mx] += 1
    n_tandan = sum(app_glob.values())
    n_multi = sum(v for k, v in app_glob.items() if k >= 2)
    return {
        "per_split": {s: {"pohon": r["pohon"], "kotak": r["kotak"],
                          "tandan": r["tandan"], "tandan_multi_sisi": r["multi"],
                          "pohon_4sisi": r["nsisi"][4], "pohon_8sisi": r["nsisi"][8],
                          "appearance_count": dict(sorted(r["app"].items()))}
                      for s, r in sorted(per.items())},
        "total_tandan": n_tandan,
        "total_tandan_multi_sisi": n_multi,
        "frac_multi_sisi": round(n_multi / n_tandan, 4),
        "appearance_count_global": dict(sorted(app_glob.items())),
        "frac_multi_yang_tepat_2_sisi": round(app_glob[2] / n_multi, 4),
        "maks_appearance_per_pohon": {str(k): dict(sorted(v.items()))
                                      for k, v in sorted(maks_per_nsisi.items())},
        "n_class_mismatch": mismatch,
        "catatan_mismatch": ("0 berarti label kelas SELALU konsisten di semua sisi "
                             "untuk tandan yang sama -- setiap ketidaksepakatan "
                             "antar sisi murni galat model, bukan galat label"),
    }


# --------------------------------------------------------------------------
# B. integritas split
# --------------------------------------------------------------------------
def blok_b(man):
    folder = {}
    for s in ["train", "val", "test"]:
        for fn in (DS / "images" / s).iterdir():
            folder["_".join(fn.stem.split("_")[:-1])] = s
    js = {}
    for tree in man:
        d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
        js[tree] = d["split"]
    return {
        "manifest": dict(Counter(man.values())),
        "dari_folder_gambar": dict(Counter(folder.values())),
        "field_split_di_json": dict(Counter(js.values())),
        "beda_manifest_vs_folder": sum(man[t] != folder[t] for t in folder),
        "beda_manifest_vs_json": sum(man[t] != js[t] for t in js),
        "PERINGATAN": ("field `split` di dalam json/*.json BUKAN split kanonik. "
                       "Pakai split_manifest.csv (identik dengan tata letak folder)."),
    }


# --------------------------------------------------------------------------
# C. biaya galat-gabung
# --------------------------------------------------------------------------
def blok_c(man):
    pos = neg = pos_sekelas = neg_sekelas = 0
    for tree in man:
        _, kotak, _ = muat_pohon(tree)
        for p, q in itertools.combinations(kotak, 2):
            if p["s"] == q["s"]:
                continue
            sekelas = p["c"] == q["c"]
            if p["bid"] is not None and p["bid"] == q["bid"]:
                pos += 1
                pos_sekelas += sekelas
            else:
                neg += 1
                neg_sekelas += sekelas
    return {
        "n_pasangan_lintas_sisi": pos + neg,
        "n_setandan": pos,
        "frac_setandan": round(pos / (pos + neg), 4),
        "frac_setandan_yang_sekelas": round(pos_sekelas / pos, 4),
        "frac_bedatandan_yang_sekelas": round(neg_sekelas / neg, 4),
        "arti": ("frac_bedatandan_yang_sekelas = peluang sebuah galat-gabung "
                 "TIDAK merusak kelas (kedua kotak kebetulan sekelas)"),
    }


# --------------------------------------------------------------------------
# D. penaut geometri-saja
# --------------------------------------------------------------------------
NAMA_FITUR = ["gap_sisi", "gap_ternorm", "n_sisi", "abs_dcx", "abs_dcy",
              "rasio_area", "rasio_w", "rasio_h", "delta_aspek", "cy_rerata",
              "kelas_sama", "log_rasio_area"]


def fitur(p, q, nv):
    g = abs(p["s"] - q["s"])
    g = min(g, nv - g)                       # jarak sisi melingkar
    ap, aq = p["w"] * p["h"], q["w"] * q["h"]
    return [g, g / nv, nv, abs(p["cx"] - q["cx"]), abs(p["cy"] - q["cy"]),
            min(ap, aq) / max(ap, aq), min(p["w"], q["w"]) / max(p["w"], q["w"]),
            min(p["h"], q["h"]) / max(p["h"], q["h"]),
            abs(p["w"] / p["h"] - q["w"] / q["h"]), (p["cy"] + q["cy"]) / 2,
            int(p["c"] == q["c"]), float(np.log(ap / aq))]


def pasangan_split(pohon_split):
    X, y = [], []
    prev = Counter()
    tot = Counter()
    for tree in pohon_split:
        nv, kotak, _ = muat_pohon(tree)
        for p, q in itertools.combinations(kotak, 2):
            if p["s"] == q["s"]:
                continue
            g = abs(p["s"] - q["s"])
            g = min(g, nv - g)
            lab = int(p["bid"] is not None and p["bid"] == q["bid"])
            tot[(nv, g)] += 1
            prev[(nv, g)] += lab
            X.append(fitur(p, q, nv))
            y.append(lab)
    return np.array(X, float), np.array(y), prev, tot


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def cari(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def gabung(self, a, b):
        self.p[self.cari(a)] = self.cari(b)


def klaster_pohon(clf, nv, kotak, ambang, maks_sisi):
    """Penugasan global berkendala: Hungarian per pasangan-sisi, lalu union-find
    serakah dengan dua kendala keras -- satu kotak per sisi per tandan, dan
    plafon jumlah sisi per tandan."""
    n = len(kotak)
    if n == 0:
        return []
    per_sisi = defaultdict(list)
    for k, b in enumerate(kotak):
        per_sisi[b["s"]].append(k)
    kandidat = []
    for a, b in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[a], per_sisi[b]
        if not A or not B:
            continue
        F = np.array([fitur(kotak[i], kotak[j], nv) for i in A for j in B], float)
        S = clf.predict_proba(F)[:, 1].reshape(len(A), len(B))
        for i, j in zip(*linear_sum_assignment(-S)):
            if S[i, j] >= ambang:
                kandidat.append((float(S[i, j]), A[i], B[j]))
    kandidat.sort(reverse=True)
    uf = UF(n)
    ukuran = Counter({k: 1 for k in range(n)})
    sisi = {k: {b["s"]} for k, b in enumerate(kotak)}
    for _, i, j in kandidat:
        ri, rj = uf.cari(i), uf.cari(j)
        if ri == rj or (sisi[ri] & sisi[rj]) or ukuran[ri] + ukuran[rj] > maks_sisi:
            continue
        uf.gabung(ri, rj)
        rn = uf.cari(ri)
        ukuran[rn] = ukuran[ri] + ukuran[rj]
        sisi[rn] = sisi[ri] | sisi[rj]
    return [uf.cari(k) for k in range(n)]


def nilai_klaster(clf, pohon, ambang):
    tp = fp = fn = 0
    aris, selisih = [], []
    for tree in pohon:
        nv, kotak, _ = muat_pohon(tree)
        if not kotak:
            continue
        lab = klaster_pohon(clf, nv, kotak, ambang, 3 if nv == 4 else 6)
        gt = [b["bid"] if b["bid"] is not None else -1000 - k
              for k, b in enumerate(kotak)]
        for i, j in itertools.combinations(range(len(kotak)), 2):
            if kotak[i]["s"] == kotak[j]["s"]:
                continue
            P, G = lab[i] == lab[j], gt[i] == gt[j]
            tp += P and G
            fp += P and not G
            fn += G and not P
        aris.append(adjusted_rand_score(gt, lab))
        selisih.append(len(set(lab)) - len(set(gt)))
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return dict(presisi=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r + 1e-9), 4),
                ari=round(float(np.mean(aris)), 4),
                bias_jumlah=round(float(np.mean(selisih)), 3),
                mae_jumlah=round(float(np.mean(np.abs(selisih))), 3))


def blok_d(man):
    pohon = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    Xtr, ytr, _, _ = pasangan_split(pohon["train"])
    Xva, yva, prev, tot = pasangan_split(pohon["val"])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=SEED).fit(Xtr, ytr)
    pv = clf.predict_proba(Xva)[:, 1]
    pr, rc, th = precision_recall_curve(yva, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    k = int(np.nanargmax(f1))

    sapuan = {}
    terbaik = (None, -1.0)
    for a in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]:
        m = nilai_klaster(clf, pohon["val"], a)
        sapuan[f"{a:.2f}"] = m
        if m["f1"] > terbaik[1]:
            terbaik = (a, m["f1"])
    kunci = terbaik[0]
    return {
        "n_pasangan_train": int(len(ytr)), "n_positif_train": int(ytr.sum()),
        "n_pasangan_val": int(len(yva)), "n_positif_val": int(yva.sum()),
        "fitur": NAMA_FITUR,
        "D1_ambang_per_pasangan_val": {
            "roc_auc": round(float(roc_auc_score(yva, pv)), 4),
            "f1_terbaik": round(float(f1[k]), 4),
            "presisi": round(float(pr[k]), 4), "recall": round(float(rc[k]), 4),
            "ambang": round(float(th[k]), 4),
        },
        "D2_penugasan_global_val": sapuan,
        "D2_ambang_dikunci_dari_val": kunci,
        "D2_test_sekali": nilai_klaster(clf, pohon["test"], kunci),
        "prevalensi_positif_per_jarak_sisi_val": {
            f"nsisi{nv}_gap{g}": round(prev[(nv, g)] / tot[(nv, g)], 4)
            for (nv, g) in sorted(tot) if tot[(nv, g)] > 300},
    }


# --------------------------------------------------------------------------
# E. selisih recall per-kemunculan vs per-tandan
# --------------------------------------------------------------------------
def iou_satu(a, B):
    if len(B) == 0:
        return np.zeros(0)
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    luas_a = (a[2] - a[0]) * (a[3] - a[1])
    luas_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (luas_a + luas_b - inter + 1e-9)


def blok_e(man, ambang_conf=(0.15, 0.25, 0.35), iou_min=0.5):
    if not NPZ_SEL5.exists():
        return {"lewat": f"{NPZ_SEL5} tidak ada"}
    z = np.load(NPZ_SEL5, allow_pickle=True)
    test = [t for t, s in man.items() if s == "test"]
    hasil = {}
    for conf in ambang_conf:
        n_app = n_app_hit = n_app_benar = 0
        n_b = n_b_hit = n_b_benar_min1 = 0
        n_multi = n_beda = 0
        for tree in test:
            d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
            b2 = {}
            for b in d["bunches"]:
                for ap in b["appearances"]:
                    b2[(ap["side_index"], ap["box_index"])] = b["bunch_id"]
            kena = defaultdict(list)
            for im in d["images"].values():
                stem = im["filename"].rsplit(".", 1)[0]
                P = z[stem] if stem in z.files else np.zeros((0, 6))
                P = P[P[:, 4] >= conf]
                G = np.array([a["bbox_pixel"] for a in im["annotations"]],
                             float).reshape(-1, 4)
                dipakai, cocok = set(), {}
                for pi in (np.argsort(-P[:, 4]) if len(P) else []):
                    ious = iou_satu(P[pi, :4], G)
                    kand = [(ious[k], k) for k in range(len(G))
                            if k not in dipakai and ious[k] >= iou_min]
                    if not kand:
                        continue
                    _, k = max(kand)
                    dipakai.add(k)
                    cocok[k] = pi
                for k, a in enumerate(im["annotations"]):
                    n_app += 1
                    if k not in cocok:
                        continue
                    n_app_hit += 1
                    pc = KELAS[int(P[cocok[k], 5])]
                    n_app_benar += pc == a["class_name"]
                    bid = b2.get((im["side_index"], a["box_index"]))
                    if bid is not None:
                        kena[bid].append(pc)
            for b in d["bunches"]:
                n_b += 1
                h = kena.get(b["bunch_id"], [])
                if h:
                    n_b_hit += 1
                    n_b_benar_min1 += any(c == b["class"] for c in h)
                if len(h) >= 2:
                    n_multi += 1
                    n_beda += len(set(h)) > 1
        hasil[f"conf{conf:.2f}"] = {
            "recall_per_kemunculan": round(n_app_hit / n_app, 4),
            "kelas_benar_per_kemunculan": round(n_app_benar / n_app, 4),
            "recall_per_tandan": round(n_b_hit / n_b, 4),
            "kelas_benar_min1_sisi_per_tandan": round(n_b_benar_min1 / n_b, 4),
            "selisih_recall_pp": round(100 * (n_b_hit / n_b - n_app_hit / n_app), 2),
            "n_tandan_terdeteksi_min2_sisi": n_multi,
            "n_tandan_kelasnya_beda_antar_sisi": n_beda,
            "frac_tidak_sepakat": round(n_beda / max(n_multi, 1), 4),
        }
    return {"sumber_prediksi": str(NPZ_SEL5.relative_to(REPO)),
            "detektor": "YOLO26l @1280 (sel5 / yolo26l_e60_i1280_v2repro)",
            "split": "test (141 pohon, 588 citra)", "iou_min": iou_min,
            "per_ambang_conf": hasil}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default=str(SUB / "results" / "probe_penautan_953.json"))
    args = ap.parse_args()

    man = muat_manifest()
    out = {
        "dataset": str(DS),
        "seed": SEED,
        "A_struktur_gt": blok_a(man),
        "B_integritas_split": blok_b(man),
        "C_biaya_galat_gabung": blok_c(man),
        "D_penaut_geometri_saja": blok_d(man),
        "E_selisih_recall": blok_e(man),
    }
    Path(args.keluaran).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
