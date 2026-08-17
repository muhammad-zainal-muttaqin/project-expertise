"""PT-E-003 — Pipeline UTUH: deteksi -> penaut nyata -> pool -> aturan keputusan.

PT-E-001 memakai tautan ORACLE (dari GT). Skrip ini menggantinya dengan penaut
sungguhan dari PT-E-002, jadi angkanya adalah angka yang benar-benar bisa
didapat tanpa GT sama sekali saat inferensi.

Perbedaan penting dari PT-E-002: di sana penaut diuji di atas kotak GT, supaya
mutu penautan terisolasi dari galat deteksi. Di sini ia bekerja di atas kotak
DETEKSI — termasuk positif palsu, yang di deployment memang ikut masuk pool.

## Gerbang G2

akurasi per-tandan dengan penaut nyata >= akurasi dengan tautan oracle - 2,0 pp.
Kalau gugur sementara G0 lolos, bottleneck-nya penaut (modul L), bukan aturan
keputusan (modul A) — hasil negatif yang terlokalisasi.

## Cara GT dipakai (dan tidak dipakai)

Penaut TIDAK melihat GT. GT hanya dipakai sesudahnya, untuk dua hal:
  1. memberi identitas sejati pada tiap deteksi (IoU>=0,5), supaya presisi/
     recall pasangan bisa dihitung;
  2. menentukan pool mana yang mewakili sebuah tandan GT (pool yang memuat
     anggota tercocokkan terbanyak dari tandan itu).

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/eval_endtoend.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import reid_pertandan as RD             # noqa: E402

DS = PP.DS
SUB = PP.SUB
KELAS = PP.KELAS


def deteksi_pohon(P: dict, z, conf: float, reid=None) -> list[dict]:
    """Deteksi per pohon (sudah disatukan per anchor), plus identitas GT-nya.

    `bid = None` berarti positif palsu: ikut masuk penautan (seperti di
    deployment) tetapi tidak punya kelas sejati.
    """
    out = []
    for s in P["sisi"]:
        D = z[s["stem"]] if s["stem"] in z.files else np.zeros((0, 11))
        if len(D):
            _, uniq = np.unique(D[:, 10], return_index=True)
            D = D[np.sort(uniq)]
            D = D[D[:, 6:10].max(1) >= conf]
        if len(D) == 0:
            continue
        G = np.stack([g["box"] for g in s["gt"]]) if s["gt"] else np.zeros((0, 4))
        M = EP.iou_mat(D[:, :4], G)
        dipakai, milik = set(), {}
        for di in np.argsort(-D[:, 6:10].max(1)):
            kand = [(M[di, gi], gi) for gi in range(len(G))
                    if gi not in dipakai and M[di, gi] >= 0.5]
            if kand:
                _, gi = max(kand)
                dipakai.add(gi)
                milik[di] = s["gt"][gi]
        w, h = s["wh"]
        f = PP.cari_citra(s["stem"])
        img = cv2.imread(str(f)) if f else None
        for di in range(len(D)):
            box = D[di, :4]
            g = milik.get(di)
            p = D[di, 6:10].astype(float)
            crop = None
            if reid is not None and img is not None:
                x1, y1, x2, y2 = box
                dx, dy = (x2 - x1) * RD.PAD, (y2 - y1) * RD.PAD
                a1 = max(0, int(x1 - dx)); b1 = max(0, int(y1 - dy))
                a2 = min(w, int(x2 + dx)); b2 = min(h, int(y2 + dy))
                crop = (cv2.resize(img[b1:b2, a1:a2], (RD.SISI, RD.SISI))
                        if a2 - a1 > 3 and b2 - b1 > 3
                        else np.zeros((RD.SISI, RD.SISI, 3), np.uint8))
            out.append({
                "crop": crop,
                "s": s["si"], "i": di, "px": box,
                "cx": float((box[0] + box[2]) / 2 / w), "cy": float((box[1] + box[3]) / 2 / h),
                "w": float((box[2] - box[0]) / w), "h": float((box[3] - box[1]) / h),
                "c": int(np.argmax(p)),
                "desk": PP.deskriptor(img, box) if img is not None
                        else np.zeros(PP.HB * PP.HS * PP.HV + 7, np.float32),
                "bid": g["bid"] if g else None,
                "gt_kelas": g["kelas"] if g else None,
                "p": p / max(p.sum(), 1e-9), "conf": float(p.max()),
                "luas": float((box[2] - box[0]) * (box[3] - box[1]) / (w * h)),
                "tepi": float(max(min(box[0], box[1], w - box[2], h - box[3]) / max(w, h), 0.0)),
            })
    if reid is not None:
        idx = [k for k, d in enumerate(out) if d["crop"] is not None]
        if idx:
            E = reid(np.stack([out[k]["crop"] for k in idx]))
            for k, e in zip(idx, E):
                out[k]["emb"] = e
        for d in out:
            d.setdefault("emb", np.zeros(128, np.float32))
            d.pop("crop", None)
    return out


def fitur_det(a, b, nv, pakai_reid, pakai_kelas=True, pakai_prob=False):
    """Fitur pasangan untuk DETEKSI. Urutannya harus sama persis dengan
    `PP.fitur_pasangan`, kalau tidak model dilatih pada susunan berbeda."""
    f = PP.fitur_geo(a, b, nv, pakai_kelas) + PP.fitur_app(a["desk"], b["desk"])
    if pakai_reid:
        f = f + PP.fitur_reid(a["emb"], b["emb"])
    if pakai_prob:
        f = f + PP.fitur_prob(a["p"], b["p"])
    return f


def klaster_deteksi(clf, nv: int, det: list[dict], ambang: float,
                    pakai_reid: bool = False, pakai_kelas: bool = True,
                    pakai_prob: bool = False) -> list[int]:
    n = len(det)
    if n == 0:
        return []
    per_sisi = defaultdict(list)
    for k, d in enumerate(det):
        per_sisi[d["s"]].append(k)
    kandidat = []
    for a, b in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[a], per_sisi[b]
        if not A or not B:
            continue
        F = [fitur_det(det[i], det[j], nv, pakai_reid, pakai_kelas, pakai_prob)
             for i in A for j in B]
        S = clf.predict_proba(np.array(F, float))[:, 1].reshape(len(A), len(B))
        for i, j in zip(*linear_sum_assignment(-S)):
            if S[i, j] >= ambang:
                kandidat.append((float(S[i, j]), A[i], B[j]))
    kandidat.sort(reverse=True)
    uf = PP.UF(n)
    ukuran = Counter({k: 1 for k in range(n)})
    sisi = {k: {d["s"]} for k, d in enumerate(det)}
    maks = 3 if nv == 4 else 6
    for _, i, j in kandidat:
        ri, rj = uf.cari(i), uf.cari(j)
        if ri == rj or (sisi[ri] & sisi[rj]) or ukuran[ri] + ukuran[rj] > maks:
            continue
        uf.gabung(ri, rj)
        rn = uf.cari(ri)
        ukuran[rn] = ukuran[ri] + ukuran[rj]
        sisi[rn] = sisi[ri] | sisi[rj]
    return [uf.cari(k) for k in range(n)]


def jalankan(split: str, pohon, z, clf, ambang, conf, skema, tau,
             reid=None, pakai_reid=False, pakai_kelas=True, pakai_prob=False) -> dict:
    tp = fp = fn = 0
    aris = []
    pools_nyata, pools_oracle = [], []
    n_pool_total = n_pool_palsu = 0
    for P in pohon:
        det = deteksi_pohon(P, z, conf, reid)
        if not det:
            continue
        lab = klaster_deteksi(clf, P["n_sisi"], det, ambang, pakai_reid,
                              pakai_kelas, pakai_prob)

        cocok = [k for k, d in enumerate(det) if d["bid"] is not None]
        for i, j in itertools.combinations(cocok, 2):
            if det[i]["s"] == det[j]["s"]:
                continue
            Pr, G = lab[i] == lab[j], det[i]["bid"] == det[j]["bid"]
            tp += Pr and G; fp += Pr and not G; fn += G and not Pr
        if cocok:
            aris.append(adjusted_rand_score([det[k]["bid"] for k in cocok],
                                            [lab[k] for k in cocok]))

        per_pool = defaultdict(list)
        for k, d in enumerate(det):
            per_pool[lab[k]].append(d)
        n_pool_total += len(per_pool)
        n_pool_palsu += sum(1 for v in per_pool.values()
                            if all(d["bid"] is None for d in v))

        # tiap tandan GT diwakili pool yang memuat anggota tercocokkannya terbanyak
        milik = defaultdict(Counter)
        for k, d in enumerate(det):
            if d["bid"] is not None:
                milik[d["bid"]][lab[k]] += 1
        for bid, c in milik.items():
            pid = c.most_common(1)[0][0]
            pools_nyata.append({"tree": P["tree"], "gt": P["tandan"][bid],
                                "pool": per_pool[pid]})
            pools_oracle.append({"tree": P["tree"], "gt": P["tandan"][bid],
                                 "pool": [d for k, d in enumerate(det)
                                          if d["bid"] == bid]})
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    hasil = {
        "penautan_di_atas_deteksi": {
            "presisi": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r + 1e-9), 4),
            "ari": round(float(np.mean(aris)), 4) if aris else None,
            "n_pool": n_pool_total,
            "n_pool_seluruhnya_positif_palsu": n_pool_palsu,
            "frac_pool_palsu": round(n_pool_palsu / max(n_pool_total, 1), 4)},
        "n_tandan_dievaluasi": len(pools_nyata),
        "PENAUT_NYATA": {a: EP.nilai(pools_nyata, a, skema, tau)
                         for a in ["R0", "R0cal", "R1", "R2", "R3", "R4"]},
        "TAUTAN_ORACLE": {a: EP.nilai(pools_oracle, a, skema, tau)
                          for a in ["R0", "R4"]},
    }
    multi_n = [q for q in pools_nyata if len(q["pool"]) >= 2]
    multi_o = [q for q in pools_oracle if len(q["pool"]) >= 2]
    hasil["HANYA_multi_tampak"] = {
        "n_nyata": len(multi_n), "n_oracle": len(multi_o),
        "nyata_R4": EP.nilai(multi_n, "R4", skema, tau),
        "nyata_R0cal": EP.nilai(multi_n, "R0cal", skema, tau),
        "oracle_R4": EP.nilai(multi_o, "R4", skema, tau)}
    hasil["bootstrap"] = {
        "nyata_R4_vs_R0": EP.bootstrap_pohon(pools_nyata, "R4", "R0", skema, tau),
        "nyata_R4_vs_R0cal_multi": EP.bootstrap_pohon(multi_n, "R4", "R0cal", skema, tau),
    }
    return hasil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_003_endtoend.json"))
    ap.add_argument("--sapu-ambang", nargs="*", type=float, default=None,
                    help="setel ambang penaut di val-DETEKSI, bukan warisan dari "
                         "klasterisasi kotak GT. Distribusi skor pasangan deteksi "
                         "berbeda (banyak positif palsu), jadi ambang GT terlalu "
                         "pelit: recall pasangan 0,12 dan hanya 29% tandan yang "
                         "punya pool >=2 tampak.")
    args = ap.parse_args()

    o = json.loads((SUB / "results" / "pt_e_001_oracle.json").read_text())
    conf, skema, tau = o["conf_dikunci"], o["skema_bobot_dikunci"], tuple(o["tau_R4_dikunci"])
    pen = json.loads((SUB / "results" / "pt_e_002_penaut.json").read_text())
    nama_var = pen.get("varian_dipakai_endtoend", pen["varian_terbaik_di_val"])
    V = pen["varian"][nama_var]
    ambang = V["ambang_dikunci_dari_val"]
    pakai_kelas = V.get("pakai_kelas_sama", True)
    pakai_prob = V.get("pakai_prob_prediksi", False)
    pakai_reid = V.get("pakai_reid", False)
    print(f"terkunci: conf={conf} skema={skema} tau={tau} | penaut={nama_var} "
          f"ambang={ambang} kelas_GT={pakai_kelas} prob_pred={pakai_prob} "
          f"reid={pakai_reid}")

    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    desk = PP.bangun_deskriptor(ids["train"] + ids["val"] + ids["test"],
                                SUB / "results" / "deskriptor_crop.npz")
    emb = None
    if pakai_reid:
        z_ = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
        emb = {k: z_[k] for k in z_.files}
    # WAJIB: konstanta pergeseran arah-putar hidup sebagai global di
    # penaut_pertandan dan hanya terisi saat skrip itu dijalankan langsung.
    # Tanpa baris ini, fitur arah TIDAK aktif di sini dan hasilnya diam-diam
    # kembali ke fitur lama — terlihat sebagai angka yang identik persis.
    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    print(f"konstanta arah-putar dipas di train: {len(PP.HARAP)} entri")
    print("melatih ulang penaut di pasangan kotak GT split train...")
    prob = PP.bangun_prob_prediksi(
        {k: ids[k] for k in ["train", "val", "test"]}) if pakai_prob else None
    Xtr, ytr = PP.pasangan(ids["train"], desk, True, emb, pakai_kelas, prob)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=PP.SEED).fit(Xtr, ytr)

    pohon = {s: [EP.muat_pohon(t) for t in ids[s]] for s in ["val", "test"]}
    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ["val", "test"]}

    reid_fn = None
    if pakai_reid:
        import torch
        m = RD.Reid().cuda().eval()
        m.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt"))

        def reid_fn(crops):
            out = []
            with torch.no_grad():
                for i in range(0, len(crops), 256):
                    x = RD.ke_tensor(crops[i:i + 256], False, "cuda")
                    out.append(m(x).float().cpu().numpy())
            return np.concatenate(out)

    if args.sapu_ambang:
        print("menyetel ambang penaut di val-DETEKSI...")
        sapuan, terbaik = {}, (None, -1.0)
        for a in args.sapu_ambang:
            r = jalankan("val", pohon["val"], z["val"], clf, a, conf, skema, tau,
                         reid_fn, pakai_reid, pakai_kelas, pakai_prob)
            # JANGAN pakai nama `m` di sini: `m` adalah model re-ID yang
            # ditangkap closure `reid_fn`. Menimpanya membuat closure memanggil
            # dict -> TypeError di tengah sapuan.
            hm = r["HANYA_multi_tampak"]
            akur = r["PENAUT_NYATA"]["R4"]["akurasi"]
            sapuan[f"{a:.2f}"] = {
                "R4_semua": akur, "n_multi": hm["n_nyata"],
                "R4_multi": hm["nyata_R4"]["akurasi"],
                "f1_pasangan": r["penautan_di_atas_deteksi"]["f1"],
                "frac_pool_palsu": r["penautan_di_atas_deteksi"]["frac_pool_palsu"]}
            print(f"  ambang {a:.2f}: R4 {akur:.4f} | multi {hm['n_nyata']} "
                  f"| F1 pasangan {r['penautan_di_atas_deteksi']['f1']:.4f}")
            if akur > terbaik[1]:
                terbaik = (a, akur)
        ambang = terbaik[0]
        print(f"  -> ambang dikunci dari val-deteksi: {ambang}")

    hasil = {"conf": conf, "skema": skema, "tau": list(tau), "penaut": nama_var,
             "ambang_penaut": ambang, "pakai_kelas_sama": pakai_kelas,
             "pakai_prob_prediksi": pakai_prob, "pakai_reid": pakai_reid,
             "sapuan_ambang_val_deteksi": (sapuan if args.sapu_ambang else None),
             "split": {}}
    for s in ["val", "test"]:
        print(f"menjalankan pipeline utuh di {s}...")
        hasil["split"][s] = jalankan(s, pohon[s], z[s], clf, ambang, conf, skema, tau,
                                     reid_fn, pakai_reid, pakai_kelas, pakai_prob)

    t = hasil["split"]["test"]
    selisih = t["PENAUT_NYATA"]["R4"]["akurasi"] - t["TAUTAN_ORACLE"]["R4"]["akurasi"]
    hasil["gerbang_G2"] = {
        "syarat": "akurasi R4 penaut nyata >= akurasi R4 oracle - 2,0 pp (test)",
        "nyata": t["PENAUT_NYATA"]["R4"]["akurasi"],
        "oracle": t["TAUTAN_ORACLE"]["R4"]["akurasi"],
        "selisih_pp": round(selisih * 100, 3),
        "putusan": "LOLOS" if selisih * 100 >= -2.0 else "GUGUR"}

    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["gerbang_G2"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
