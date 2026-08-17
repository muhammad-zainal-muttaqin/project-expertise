"""PT-E-002 — Penaut tandan lintas-sisi: geometri + PENAMPILAN.

Probe PT-E-000 sudah menunjukkan geometri saja mentok: F1 pasangan 0,4282 dan
ARI 0,3912 di test. Sinyal peringkatnya kuat (ROC-AUC 0,9301) tetapi prevalensi
positif cuma 6,62%, jadi ambang per-pasangan selalu tenggelam oleh negatif —
dan penugasan global berkendala pun tidak menambalnya.

Yang ditambahkan di sini: **penampilan potongan**. E-007 menyebut varian
"hanya penampilan", tetapi isinya kelas + ukuran kotak — tidak ada satu piksel
pun. Skrip ini memakai piksel sungguhan.

Dua tingkat, sengaja dipisah supaya kontribusinya bisa dibaca terpisah:

  A. deskriptor tangan   histogram HSV + statistik warna + ketajaman.
                         Tanpa training, tanpa GPU.
  B. embedding terlatih  (belum diaktifkan) metric learning atas graf identitas
                         GT. Dikerjakan hanya kalau A belum melewati G1.

Penugasan tetap global dan berkendala, sama seperti PT-E-000: Hungarian per
pasangan-sisi, lalu union-find serakah dengan dua kendala keras — satu kotak
per sisi per tandan, dan plafon ukuran pool (3 untuk 4-sisi, 6 untuk 8-sisi).

## Yang diukur

Mutu PENAUTAN itu sendiri, di atas kotak GT — supaya galat deteksi tidak
mengotori angkanya. Galat deteksi masuk di PT-E-003 (end-to-end).

## Gerbang G1

F1 pasangan val >= 0,65 DAN ARI val >= 0,55.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/penaut_pertandan.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, precision_recall_curve, roc_auc_score

DS = Path("/workspace/SawitMVC-YOLO")
SUB = Path(__file__).resolve().parents[1]
KELAS = ["B1", "B2", "B3", "B4"]
SEED = 0
HB, HS, HV = 8, 4, 4                     # bin histogram HSV
TAG = ""                                 # akhiran nama dump prediksi (mis. "_352")
PAD = 0.10                               # padding potongan


# --------------------------------------------------------------------------
def muat_manifest() -> dict[str, str]:
    man = {}
    with (DS / "split_manifest.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            man[r["tree_id"]] = r["new_split"]
    return man


def cari_citra(stem: str):
    """Lokasi berkas citra untuk sebuah stem.

    Ada di satu tempat supaya dataset dengan tata letak berbeda bisa ditangani
    dengan menimpa fungsi ini, bukan dengan menduplikasi skrip. SawitMVC-YOLO
    memakai `images/{train,val,test}/`; SawitMVC-Depth memakai `images/` datar.
    """
    for sp in ("train", "val", "test"):
        q = DS / "images" / sp / f"{stem}.jpg"
        if q.exists():
            return q
    q = DS / "images" / f"{stem}.jpg"
    return q if q.exists() else None


def muat_pohon(tree: str):
    d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
    b2 = {}
    for b in d["bunches"]:
        for ap in b["appearances"]:
            b2[(ap["side_index"], ap["box_index"])] = b["bunch_id"]
    kotak = []
    for im in d["images"].values():
        w, h = im["width"], im["height"]
        for a in im["annotations"]:
            cx, cy, bw, bh = a["bbox_yolo"]
            kotak.append(dict(s=im["side_index"], i=a["box_index"], c=a["class_id"],
                              cx=cx, cy=cy, w=bw, h=bh,
                              stem=im["filename"].rsplit(".", 1)[0], wh=(w, h),
                              px=a["bbox_pixel"],
                              bid=b2.get((im["side_index"], a["box_index"]))))
    return len(d["images"]), kotak


# --------------------------------------------------------------------------
# deskriptor penampilan
# --------------------------------------------------------------------------
def deskriptor(img: np.ndarray, px) -> np.ndarray:
    """Histogram HSV ternormalkan + rerata/simpangan HSV + ketajaman.

    Panjang HB*HS*HV + 6 + 1. Histogram dinormalkan L1 supaya tahan terhadap
    perbedaan ukuran potongan antar sisi (tandan yang sama tampak besar dari
    satu sisi dan kecil dari sisi lain).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = px
    dx, dy = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1 = max(0, int(x1 - dx)); y1 = max(0, int(y1 - dy))
    x2 = min(w, int(x2 + dx)); y2 = min(h, int(y2 + dy))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros(HB * HS * HV + 7, np.float32)
    c = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [HB, HS, HV],
                        [0, 180, 0, 256, 0, 256]).ravel()
    hist = hist / max(hist.sum(), 1e-9)
    m = hsv.reshape(-1, 3).mean(0) / 255.0
    s = hsv.reshape(-1, 3).std(0) / 255.0
    g = cv2.cvtColor(cv2.resize(c, (64, 64)), cv2.COLOR_BGR2GRAY)
    tajam = float(cv2.Laplacian(g, cv2.CV_32F).var()) / 1000.0
    return np.concatenate([hist, m, s, [tajam]]).astype(np.float32)


def bangun_deskriptor(pohon_ids: list[str], cache: Path) -> dict:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = {}
    for n, tree in enumerate(pohon_ids, 1):
        _, kotak = muat_pohon(tree)
        per_stem = defaultdict(list)
        for b in kotak:
            per_stem[b["stem"]].append(b)
        for stem, bs in per_stem.items():
            f = cari_citra(stem)
            img = cv2.imread(str(f)) if f else None
            for b in bs:
                out[f"{tree}|{b['s']}|{b['i']}"] = (
                    deskriptor(img, b["px"]) if img is not None
                    else np.zeros(HB * HS * HV + 7, np.float32))
        if n % 100 == 0:
            print(f"  deskriptor: {n}/{len(pohon_ids)} pohon", flush=True)
    np.savez_compressed(cache, **out)
    return out


# --------------------------------------------------------------------------
# fitur pasangan
# --------------------------------------------------------------------------
NAMA_GEO = ["gap_sisi", "gap_ternorm", "n_sisi", "abs_dcx", "abs_dcy", "rasio_area",
            "rasio_w", "rasio_h", "delta_aspek", "cy_rerata", "kelas_sama",
            "log_rasio_area"]
NAMA_APP = ["hist_intersect", "hist_chi2", "hist_bhatt", "d_mean_hsv", "d_std_hsv",
            "d_log_tajam"]
NAMA_REID = ["cos_reid", "l2_reid"]
NAMA_PROB = ["argmax_pred_sama", "bhatt_prob", "d_ekspektasi_ordinal", "conf_min"]
NAMA_ARAH = ["offset_bertanda", "dx_bertanda", "dy_bertanda", "sisa_dx", "abs_sisa_dx"]

# Pergeseran horizontal yang DIHARAPKAN per (n_sisi, offset), dipas dari split
# train oleh `hitung_harapan_geser()`. Lihat catatan di fungsi itu.
#
# JEBAKAN YANG SUDAH MEMAKAN KORBAN: ini global modul. Versi pertama hanya
# mengisinya di dalam `main()` skrip ini, sehingga skrip lain yang meng-import
# modul ini mendapatkannya KOSONG — fitur arah tidak aktif, dan hasilnya diam-diam
# kembali ke fitur lama tanpa satu pun pesan galat. Gejalanya: F1 end-to-end
# 0,1766 -> 0,1761, identik sampai tiga desimal, padahal di kotak GT melonjak
# 0,3651 -> 0,6486. Angka yang "tidak berubah sama sekali" itu tandanya.
#
# Sekarang tabelnya di-cache ke berkas dan dimuat OTOMATIS saat modul di-import,
# jadi tidak ada lagi skrip yang bisa lupa mengisinya.
BERKAS_HARAP = SUB / "results" / "harapan_geser.json"
HARAP: dict = {}


def _muat_harap_dari_berkas() -> dict:
    if not BERKAS_HARAP.exists():
        return {}
    mentah = json.loads(BERKAS_HARAP.read_text())
    return {(int(k.split("|")[0]), int(k.split("|")[1])): float(v)
            for k, v in mentah.items()}


def hitung_harapan_geser(train_ids: list[str]) -> dict:
    """Median pergeseran x untuk pasangan setandan, per (n_sisi, offset bertanda).

    Kenapa ini ada. Foto diambil **memutari pohon searah jarum jam** (dikonfirmasi
    pemilik data). Konsekuensinya sebuah tandan tidak berpindah sembarangan antar
    sisi: ia bergeser ke arah yang selalu sama, dengan besar yang khas.

    Terukur di pohon 4-sisi, seluruh korpus:

        offset +1  ->  dx = +0,241  (98,6% bergeser ke kanan, simpangan 0,116)
        offset +3  ->  dx = -0,260  (99,7% bergeser ke kiri,  simpangan 0,109)
        pasangan SALAH -> dx ~ 0,000 (50/50, simpangan 0,213)

    Fitur lama memakai `abs_dcx` — nilai mutlak — sehingga populasi +0,24 dan
    -0,26 tergabung jadi satu dan sinyalnya hancur. Menambahkan versi BERTANDA
    menaikkan AUC val 0,8220 -> 0,9168 dan F1 pasangan 0,3221 -> 0,5019, lompatan
    terbesar dari fitur mana pun yang dicoba di sub-proyek ini.

    Konstanta dipas HANYA di split train supaya tidak bocor ke val/test.
    """
    global HARAP
    kum = defaultdict(list)
    for t in train_ids:
        nv, kotak = muat_pohon(t)
        for p, q in itertools.combinations(kotak, 2):
            if p["s"] == q["s"]:
                continue
            if p["bid"] is None or p["bid"] != q["bid"]:
                continue
            a, b = (p, q) if p["s"] < q["s"] else (q, p)
            kum[(nv, (b["s"] - a["s"]) % nv)].append(b["cx"] - a["cx"])
    HARAP = {k: float(np.median(v)) for k, v in kum.items() if len(v) >= 20}
    BERKAS_HARAP.parent.mkdir(parents=True, exist_ok=True)
    BERKAS_HARAP.write_text(json.dumps(
        {f"{k[0]}|{k[1]}": round(v, 5) for k, v in sorted(HARAP.items())}, indent=1))
    return HARAP
NB = HB * HS * HV


def bangun_prob_prediksi(ids_per_split: dict) -> dict:
    """(tree|sisi|box_index) -> vektor 4-kelas PREDIKSI detektor, via IoU>=0,5.

    Kenapa ini ada. Aturan fisik "beda kelas berarti bukan tandan yang sama"
    memang BENAR -- di GT ia berlaku 100% (`class_mismatch` = 0). Masalahnya,
    saat inferensi penaut tidak melihat kelas GT melainkan kelas PREDIKSI, dan
    di sana aturan itu hanya benar ~77%: 23,3% tandan multi-sisi punya prediksi
    kelas yang berbeda antar sisi. Melatih penaut dengan kelas GT lalu
    memakainya atas kelas prediksi memberi veto absolut kepada bukti yang
    berderau -- dan justru memecah pool pada 23% kasus yang ingin diperbaiki
    agregasi.

    Solusinya bukan membuang sinyalnya (itu sinyal sah), melainkan memakai
    besaran yang SAMA di latih dan inferensi, dalam bentuk LUNAK.

    Kotak GT tanpa deteksi yang cocok diberi vektor seragam -- artinya
    "tidak ada bukti kelas", bukan "kelasnya berbeda".
    """
    out = {}
    seragam = np.full(4, 0.25)
    for split, ids in ids_per_split.items():
        f = SUB / "results" / f"pred_skorpenuh{TAG}_{split}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        for tree in ids:
            d = json.loads((DS / "json" / f"{tree}.json").read_text(encoding="utf-8-sig"))
            for im in d["images"].values():
                stem = im["filename"].rsplit(".", 1)[0]
                si = im["side_index"]
                G = np.array([a["bbox_pixel"] for a in im["annotations"]],
                             float).reshape(-1, 4)
                D = z[stem] if stem in z.files else np.zeros((0, 11))
                if len(D):
                    _, u = np.unique(D[:, 10], return_index=True)
                    D = D[np.sort(u)]
                for a in im["annotations"]:
                    out[f"{tree}|{si}|{a['box_index']}"] = seragam
                if len(D) == 0 or len(G) == 0:
                    continue
                x1 = np.maximum(D[:, None, 0], G[None, :, 0])
                y1 = np.maximum(D[:, None, 1], G[None, :, 1])
                x2 = np.minimum(D[:, None, 2], G[None, :, 2])
                y2 = np.minimum(D[:, None, 3], G[None, :, 3])
                inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                ad = (D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1])
                ag = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
                M = inter / (ad[:, None] + ag[None, :] - inter + 1e-9)
                dipakai = set()
                for di in np.argsort(-D[:, 6:10].max(1)):
                    kand = [(M[di, gi], gi) for gi in range(len(G))
                            if gi not in dipakai and M[di, gi] >= 0.5]
                    if not kand:
                        continue
                    _, gi = max(kand)
                    dipakai.add(gi)
                    v = D[di, 6:10].astype(float)
                    bi = im["annotations"][gi]["box_index"]
                    out[f"{tree}|{si}|{bi}"] = v / max(v.sum(), 1e-9)
    return out


def fitur_prob(pa, pb):
    k = np.arange(4)
    return [int(np.argmax(pa) == np.argmax(pb)),
            float(np.sqrt(pa * pb).sum()),
            float(abs(pa @ k - pb @ k)),
            float(min(pa.max(), pb.max()))]


def fitur_geo(p, q, nv, pakai_kelas: bool = True):
    """Fitur geometri pasangan.

    `pakai_kelas=False` MEMBUANG `kelas_sama`. Itu bukan penghematan fitur
    melainkan perbaikan bug: karena `class_mismatch` = 0 di seluruh GT, setiap
    pasangan positif selalu sekelas, sehingga penaut belajar MENOLAK semua
    penggabungan lintas-kelas (terukur: 100,0% pool multi-anggota homogen
    kelasnya, dan `kelas_sama` menurunkan AUC 0,375 saat dipermutasi -- lima
    kali lipat fitur berikutnya). Akibatnya pool tidak pernah memuat tampak
    yang berbeda kelas, dan agregasi kehilangan satu-satunya hal yang bisa ia
    perbaiki. Ketergantungannya melingkar: kelas dipakai menentukan identitas,
    lalu identitas dipakai memperbaiki kelas.
    """
    # urutan kanonik: selalu dari sisi ber-indeks lebih kecil ke lebih besar,
    # supaya tanda pergeseran punya arti tetap
    a, b = (p, q) if p["s"] < q["s"] else (q, p)
    d = (b["s"] - a["s"]) % nv
    g = min(d, nv - d)
    ap, aq = a["w"] * a["h"], b["w"] * b["h"]
    f = [g, g / nv, nv, abs(a["cx"] - b["cx"]), abs(a["cy"] - b["cy"]),
         min(ap, aq) / max(ap, aq), min(a["w"], b["w"]) / max(a["w"], b["w"]),
         min(a["h"], b["h"]) / max(a["h"], b["h"]),
         abs(a["w"] / a["h"] - b["w"] / b["h"]), (a["cy"] + b["cy"]) / 2]
    if pakai_kelas:
        f.append(int(a["c"] == b["c"]))
    f.append(float(np.log(ap / aq)))
    if HARAP:
        dx = b["cx"] - a["cx"]
        sisa = dx - HARAP.get((nv, d), 0.0)
        f += [float(d), dx, b["cy"] - a["cy"], sisa, abs(sisa)]
    return f


def fitur_app(da, db):
    ha, hb = da[:NB], db[:NB]
    inter = float(np.minimum(ha, hb).sum())
    chi2 = float((((ha - hb) ** 2) / (ha + hb + 1e-9)).sum())
    bhat = float(np.sqrt(np.maximum(0.0, 1.0 - np.sqrt(ha * hb).sum())))
    dm = float(np.linalg.norm(da[NB:NB + 3] - db[NB:NB + 3]))
    ds = float(np.linalg.norm(da[NB + 3:NB + 6] - db[NB + 3:NB + 6]))
    dt = float(abs(np.log(da[NB + 6] + 1e-3) - np.log(db[NB + 6] + 1e-3)))
    return [inter, chi2, bhat, dm, ds, dt]


def fitur_reid(ea, eb):
    cos = float(np.dot(ea, eb))
    return [cos, float(np.linalg.norm(ea - eb))]


def fitur_pasangan(p, q, nv, tree, desk, pakai_app, emb, pakai_kelas=True, prob=None):
    f = fitur_geo(p, q, nv, pakai_kelas)
    ka, kb = f"{tree}|{p['s']}|{p['i']}", f"{tree}|{q['s']}|{q['i']}"
    if pakai_app:
        f = f + fitur_app(desk[ka], desk[kb])
    if emb is not None:
        f = f + fitur_reid(emb[ka], emb[kb])
    if prob is not None:
        f = f + fitur_prob(prob[ka], prob[kb])
    return f


def pasangan(pohon_ids, desk, pakai_app: bool, emb=None, pakai_kelas=True, prob=None):
    X, y = [], []
    for tree in pohon_ids:
        nv, kotak = muat_pohon(tree)
        for p, q in itertools.combinations(kotak, 2):
            if p["s"] == q["s"]:
                continue
            X.append(fitur_pasangan(p, q, nv, tree, desk, pakai_app, emb, pakai_kelas, prob))
            y.append(int(p["bid"] is not None and p["bid"] == q["bid"]))
    return np.array(X, float), np.array(y)


# --------------------------------------------------------------------------
# penugasan berkendala
# --------------------------------------------------------------------------
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


def klaster(clf, tree, desk, pakai_app, ambang, emb=None, pakai_kelas=True, prob=None):
    nv, kotak = muat_pohon(tree)
    n = len(kotak)
    if n == 0:
        return [], kotak
    per_sisi = defaultdict(list)
    for k, b in enumerate(kotak):
        per_sisi[b["s"]].append(k)
    kandidat = []
    for a, b in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[a], per_sisi[b]
        if not A or not B:
            continue
        F = [fitur_pasangan(kotak[i], kotak[j], nv, tree, desk, pakai_app, emb,
                            pakai_kelas, prob) for i in A for j in B]
        S = clf.predict_proba(np.array(F, float))[:, 1].reshape(len(A), len(B))
        for i, j in zip(*linear_sum_assignment(-S)):
            if S[i, j] >= ambang:
                kandidat.append((float(S[i, j]), A[i], B[j]))
    kandidat.sort(reverse=True)
    uf = UF(n)
    ukuran = Counter({k: 1 for k in range(n)})
    sisi = {k: {b["s"]} for k, b in enumerate(kotak)}
    maks = 3 if nv == 4 else 6
    for _, i, j in kandidat:
        ri, rj = uf.cari(i), uf.cari(j)
        if ri == rj or (sisi[ri] & sisi[rj]) or ukuran[ri] + ukuran[rj] > maks:
            continue
        uf.gabung(ri, rj)
        rn = uf.cari(ri)
        ukuran[rn] = ukuran[ri] + ukuran[rj]
        sisi[rn] = sisi[ri] | sisi[rj]
    return [uf.cari(k) for k in range(n)], kotak


def nilai_klaster(clf, pohon_ids, desk, pakai_app, ambang, emb=None, pakai_kelas=True, prob=None):
    tp = fp = fn = 0
    aris, selisih = [], []
    for tree in pohon_ids:
        lab, kotak = klaster(clf, tree, desk, pakai_app, ambang, emb, pakai_kelas, prob)
        if not kotak:
            continue
        gt = [b["bid"] if b["bid"] is not None else -1000 - k
              for k, b in enumerate(kotak)]
        for i, j in itertools.combinations(range(len(kotak)), 2):
            if kotak[i]["s"] == kotak[j]["s"]:
                continue
            P, G = lab[i] == lab[j], gt[i] == gt[j]
            tp += P and G; fp += P and not G; fn += G and not P
        aris.append(adjusted_rand_score(gt, lab))
        selisih.append(len(set(lab)) - len(set(gt)))
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return dict(presisi=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r + 1e-9), 4),
                ari=round(float(np.mean(aris)), 4),
                bias_jumlah=round(float(np.mean(selisih)), 3),
                mae_jumlah=round(float(np.mean(np.abs(selisih))), 3))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_002_penaut.json"))
    ap.add_argument("--hanya", nargs="*", default=None,
                    help="jalankan hanya varian bernama ini; sisanya diambil dari "
                         "keluaran sebelumnya kalau ada")
    args = ap.parse_args()

    man = muat_manifest()
    pohon = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    semua = pohon["train"] + pohon["val"] + pohon["test"]

    cache = SUB / "results" / "deskriptor_crop.npz"
    print("membangun deskriptor penampilan (sekali, lalu di-cache)...")
    desk = bangun_deskriptor(semua, cache)
    print(f"  {len(desk)} potongan")

    hasil = {"seed": SEED, "fitur_geometri": NAMA_GEO, "fitur_penampilan": NAMA_APP,
             "varian": {}}

    # Embedding OUT-OF-FOLD. Model penuh (dilatih di seluruh train) menghafal
    # identitas train -- cosine-nya AUC 1,0000 di train tapi 0,7564 di val.
    # Kalau penaut dilatih memakai embedding itu, ia belajar aturan yang hanya
    # benar di train dan AUC val-nya runtuh ke 0,578 (terukur). Karena itu
    # pasangan TRAIN memakai embedding dari model yang TIDAK melihat pohon
    # tersebut (fold), sedangkan val/test memakai model penuh.
    f_emb = SUB / "results" / "reid_embedding.npz"
    emb = None
    if f_emb.exists():
        z = np.load(f_emb, allow_pickle=True)
        emb = {k: z[k] for k in z.files}
        nf = 2
        oof = 0
        for fo in range(nf):
            fz = SUB / "results" / f"reid_embedding_f{fo}.npz"
            if not fz.exists():
                continue
            zz = np.load(fz, allow_pickle=True)
            ditahan = {t for i, t in enumerate(sorted(pohon["train"])) if i % nf == fo}
            for k in zz.files:
                if k.split("|")[0] in ditahan:
                    emb[k] = zz[k]
                    oof += 1
        print(f"embedding re-ID: {len(emb)} potongan, {oof} di antaranya out-of-fold")

    global HARAP
    HARAP = hitung_harapan_geser(pohon["train"])
    print("pergeseran x yang diharapkan per (n_sisi, offset), dipas di TRAIN:")
    for k in sorted(HARAP):
        print(f"    n_sisi={k[0]} offset={k[1]}: {HARAP[k]:+.3f}")

    print("memetakan distribusi kelas PREDIKSI ke kotak GT (IoU>=0,5)...")
    prob = bangun_prob_prediksi(pohon)
    print(f"  {len(prob)} kotak GT diberi vektor prediksi")

    #  nama                          app    emb   kelas_GT  prob_prediksi
    varian = [
        ("A_geometri_saja",          False, None, True,  None),
        ("B_geo_penampilan",         True,  None, True,  None),
        ("B2_tanpa_fitur_kelas",     True,  None, False, None),
        ("D_kelas_prediksi_lunak",   True,  None, False, prob),
    ]
    if emb is not None:
        varian.append(("E_reid_plus_kelas_prediksi", True, emb, False, prob))

    lama = {}
    if args.hanya:
        f = Path(args.keluaran)
        if f.exists():
            lama = json.loads(f.read_text()).get("varian", {})
        varian = [v for v in varian if v[0] in args.hanya]
        hasil["varian"].update({k: v for k, v in lama.items() if k not in args.hanya})

    for nama, pakai_app, E, pakai_kelas, PR in varian:
        print(f"\n=== {nama} ===")
        Xtr, ytr = pasangan(pohon["train"], desk, pakai_app, E, pakai_kelas, PR)
        Xva, yva = pasangan(pohon["val"], desk, pakai_app, E, pakai_kelas, PR)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             random_state=SEED).fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)[:, 1]
        pr, rc, th = precision_recall_curve(yva, pv)
        f1 = 2 * pr * rc / (pr + rc + 1e-12)
        k = int(np.nanargmax(f1))
        blok = {"n_pasangan_train": int(len(ytr)), "n_positif_train": int(ytr.sum()),
                "per_pasangan_val": {
                    "roc_auc": round(float(roc_auc_score(yva, pv)), 4),
                    "f1_terbaik": round(float(f1[k]), 4),
                    "presisi": round(float(pr[k]), 4), "recall": round(float(rc[k]), 4)}}
        print(f"  per-pasangan val: AUC {blok['per_pasangan_val']['roc_auc']} "
              f"F1 {blok['per_pasangan_val']['f1_terbaik']}")

        sapuan, terbaik = {}, (None, -1.0)
        for a in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            m = nilai_klaster(clf, pohon["val"], desk, pakai_app, a, E, pakai_kelas, PR)
            sapuan[f"{a:.2f}"] = m
            print(f"  ambang {a:.2f}: F1 {m['f1']} ARI {m['ari']} MAE_n {m['mae_jumlah']}")
            if m["f1"] > terbaik[1]:
                terbaik = (a, m["f1"])
        blok["sapuan_val"] = sapuan
        blok["ambang_dikunci_dari_val"] = terbaik[0]
        blok["val"] = sapuan[f"{terbaik[0]:.2f}"]
        blok["pakai_kelas_sama"] = pakai_kelas
        blok["pakai_prob_prediksi"] = PR is not None
        blok["pakai_reid"] = E is not None
        blok["test_sekali"] = nilai_klaster(clf, pohon["test"], desk, pakai_app,
                                            terbaik[0], E, pakai_kelas, PR)
        print(f"  TEST: {blok['test_sekali']}")
        hasil["varian"][nama] = blok

    V = hasil["varian"]
    hasil["kontribusi_val"] = {
        "penampilan_tangan_B_vs_A": round(V["B_geo_penampilan"]["val"]["f1"]
                                          - V["A_geometri_saja"]["val"]["f1"], 4),
        "buang_fitur_kelas_B2_vs_B": round(V["B2_tanpa_fitur_kelas"]["val"]["f1"]
                                           - V["B_geo_penampilan"]["val"]["f1"], 4),
        "kelas_prediksi_lunak_D_vs_B2": round(V["D_kelas_prediksi_lunak"]["val"]["f1"]
                                              - V["B2_tanpa_fitur_kelas"]["val"]["f1"], 4),
    }
    if "E_reid_plus_kelas_prediksi" in V:
        hasil["kontribusi_val"]["reid_E_vs_D"] = round(
            V["E_reid_plus_kelas_prediksi"]["val"]["f1"]
            - V["D_kelas_prediksi_lunak"]["val"]["f1"], 4)
    terbaik_nama = max(V, key=lambda k: V[k]["val"]["f1"])
    t = V[terbaik_nama]["val"]
    hasil["varian_terbaik_di_val"] = terbaik_nama
    # yang dipakai end-to-end: terbaik DI ANTARA yang tidak memakai kelas GT
    sah = [k for k in V if not V[k]["pakai_kelas_sama"]]
    hasil["varian_dipakai_endtoend"] = max(sah, key=lambda k: V[k]["val"]["f1"])
    hasil["gerbang_G1"] = {
        "syarat": "F1 pasangan val >= 0,65 DAN ARI val >= 0,55",
        "varian": terbaik_nama, "f1_val": t["f1"], "ari_val": t["ari"],
        "putusan": "LOLOS" if (t["f1"] >= 0.65 and t["ari"] >= 0.55) else "GUGUR"}

    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["gerbang_G1"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# Dimuat otomatis saat modul di-import, supaya tidak ada skrip yang bisa lupa.
HARAP = _muat_harap_dari_berkas()
if not HARAP:
    import warnings
    warnings.warn(
        f"HARAP kosong ({BERKAS_HARAP.name} belum ada): fitur arah-putar TIDAK "
        "aktif dan hasil akan diam-diam kembali ke fitur lama. Jalankan "
        "hitung_harapan_geser(train_ids) lebih dulu.", RuntimeWarning, stacklevel=2)
