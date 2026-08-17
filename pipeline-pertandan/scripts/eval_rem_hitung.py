"""PT-E-007 — Memakai penghitung Baseline-SawitMVC sebagai REM penggabungan.

Gagasan. Kelemahan terbesar penaut di PT-E-003 bukan salah gabung, melainkan
**terlalu pelit**: untuk 1.269 tandan asli ia menghasilkan 3.091 kelompok
(2,4x terlalu banyak), karena berhenti pada ambang skor yang ditebak sendiri.

Algoritma M01-M05 dari `ULM-SawitMVC/Baseline-SawitMVC` justru tahu jawaban yang
tidak diketahui penaut: **berapa banyak tandan di pohon ini**. Ia tidak pernah
mencocokkan kotak, jadi ia tidak bisa menggantikan penaut — tetapi angkanya bisa
dipakai sebagai TARGET: gabungkan terus, dari pasangan berskor tertinggi ke
bawah, sampai jumlah kelompok turun ke angka itu.

Tiga mode dibandingkan pada deteksi dan penaut yang persis sama:

  A  ambang tetap        seperti PT-E-003 — berhenti saat skor < ambang
  B  rem M01             berhenti saat jumlah kelompok <= taksiran M01
  C  rem oracle          berhenti saat jumlah kelompok <= jumlah tandan SEBENARNYA
                         -> plafon gagasan ini; menjawab "seberapa jauh gagasan
                            ini bisa membawa kita kalau cacahnya sempurna"

Kendala keras tetap berlaku di ketiga mode: satu kotak per sisi per tandan, dan
plafon ukuran kelompok (3 untuk 4-sisi, 6 untuk 8-sisi).

## Yang TIDAK diklaim skrip ini

Kalau jumlah kelompok dipaksa sama dengan taksiran M01, maka menghitung kelompok
otomatis mengembalikan angka M01 itu sendiri. Jadi **counting tidak diuji di
sini** — tidak ada kontribusi baru yang bisa diklaim untuk counting. Yang diuji
hanya satu: apakah KLASIFIKASI per tandan membaik.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/eval_rem_hitung.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402
import eval_endtoend as EE              # noqa: E402
import reid_pertandan as RD             # noqa: E402
import eval_counting_baseline as CB     # noqa: E402

SUB = PP.SUB
KELAS = PP.KELAS


def skor_pasangan(clf, nv, det, pakai_reid, pakai_kelas, pakai_prob):
    """Semua pasangan lintas-sisi + skornya, terurut menurun."""
    per_sisi = defaultdict(list)
    for k, d in enumerate(det):
        per_sisi[d["s"]].append(k)
    kandidat = []
    for a, b in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[a], per_sisi[b]
        if not A or not B:
            continue
        F = [EE.fitur_det(det[i], det[j], nv, pakai_reid, pakai_kelas, pakai_prob)
             for i in A for j in B]
        S = clf.predict_proba(np.array(F, float))[:, 1].reshape(len(A), len(B))
        for x, i in enumerate(A):
            for y, j in enumerate(B):
                kandidat.append((float(S[x, y]), i, j))
    kandidat.sort(reverse=True)
    return kandidat


def gabung(det, nv, kandidat, ambang=None, target=None):
    """Union-find serakah. Berhenti saat skor < ambang ATAU jumlah kelompok <= target.

    `ambang` dan `target` boleh dua-duanya diberikan; yang mana pun tercapai
    lebih dulu menghentikan penggabungan.
    """
    n = len(det)
    uf = PP.UF(n)
    ukuran = Counter({k: 1 for k in range(n)})
    sisi = {k: {d["s"]} for k, d in enumerate(det)}
    maks = 3 if nv == 4 else 6
    n_kelompok = n
    for s, i, j in kandidat:
        if ambang is not None and s < ambang:
            break
        if target is not None and n_kelompok <= target:
            break
        ri, rj = uf.cari(i), uf.cari(j)
        if ri == rj or (sisi[ri] & sisi[rj]) or ukuran[ri] + ukuran[rj] > maks:
            continue
        uf.gabung(ri, rj)
        rn = uf.cari(ri)
        ukuran[rn] = ukuran[ri] + ukuran[rj]
        sisi[rn] = sisi[ri] | sisi[rj]
        n_kelompok -= 1
    return [uf.cari(k) for k in range(n)]


def nilai_mode(pohon, z, clf, cfg, conf, skema, tau, reid_fn, mode, ambang,
               predict_m01):
    pools, tp, fp, fn = [], 0, 0, 0
    aris, n_kel, n_benar_kel = [], 0, 0
    for P in pohon:
        det = EE.deteksi_pohon(P, z, conf, reid_fn)
        if not det:
            continue
        kandidat = skor_pasangan(clf, P["n_sisi"], det, cfg["pakai_reid"],
                                 cfg["pakai_kelas"], cfg["pakai_prob"])
        if mode == "A_ambang_tetap":
            lab = gabung(det, P["n_sisi"], kandidat, ambang=ambang)
        elif mode == "B_rem_M01":
            d_in = [{"class": KELAS[int(np.argmax(d["p"]))], "x_norm": d["cx"],
                     "y_norm": d["cy"], "side_index": d["s"]} for d in det]
            t = sum(predict_m01(d_in).values())
            lab = gabung(det, P["n_sisi"], kandidat, target=max(t, 1))
        elif mode == "C_rem_oracle":
            lab = gabung(det, P["n_sisi"], kandidat, target=max(len(P["tandan"]), 1))
        else:
            raise ValueError(mode)

        n_kel += len(set(lab))
        n_benar_kel += len(P["tandan"])
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
        milik = defaultdict(Counter)
        for k, d in enumerate(det):
            if d["bid"] is not None:
                milik[d["bid"]][lab[k]] += 1
        for bid, c in milik.items():
            pools.append({"tree": P["tree"], "gt": P["tandan"][bid],
                          "pool": per_pool[c.most_common(1)[0][0]]})
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    multi = [q for q in pools if len(q["pool"]) >= 2]
    return {
        "penautan": {"presisi": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r + 1e-9), 4),
                     "ari": round(float(np.mean(aris)), 4) if aris else None},
        "n_kelompok": n_kel, "n_tandan_sebenarnya": n_benar_kel,
        "rasio_kelompok": round(n_kel / max(n_benar_kel, 1), 3),
        "n_tandan_dievaluasi": len(pools), "n_multi_tampak": len(multi),
        "frac_multi": round(len(multi) / max(len(pools), 1), 4),
        "aturan": {a: EP.nilai(pools, a, skema, tau) for a in ["R0", "R0cal", "R4"]},
        "multi_R4": EP.nilai(multi, "R4", skema, tau),
        "multi_R0cal": EP.nilai(multi, "R0cal", skema, tau),
        "boot_R4_vs_R0": EP.bootstrap_pohon(pools, "R4", "R0", skema, tau),
        "boot_R4_vs_R0cal_multi": EP.bootstrap_pohon(multi, "R4", "R0cal", skema, tau),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default=str(SUB / "results" / "pt_e_007_rem_hitung.json"))
    args = ap.parse_args()

    o = json.loads((SUB / "results" / "pt_e_001_oracle.json").read_text())
    conf, skema, tau = o["conf_dikunci"], o["skema_bobot_dikunci"], tuple(o["tau_R4_dikunci"])
    pen = json.loads((SUB / "results" / "pt_e_002_penaut.json").read_text())
    nama_var = pen.get("varian_dipakai_endtoend", pen["varian_terbaik_di_val"])
    V = pen["varian"][nama_var]
    cfg = {"pakai_kelas": V.get("pakai_kelas_sama", True),
           "pakai_prob": V.get("pakai_prob_prediksi", False),
           "pakai_reid": V.get("pakai_reid", False)}
    ambang = V["ambang_dikunci_dari_val"]
    print(f"penaut {nama_var} | ambang {ambang} | cfg {cfg} | conf {conf}")

    predict_m01 = CB.muat_algoritma()["M01_selector_b2b3"]
    man = PP.muat_manifest()
    ids = {s: [t for t, v in man.items() if v == s] for s in ["train", "val", "test"]}
    desk = PP.bangun_deskriptor(ids["train"] + ids["val"] + ids["test"],
                                SUB / "results" / "deskriptor_crop.npz")
    emb = None
    reid_fn = None
    if cfg["pakai_reid"]:
        zz = np.load(SUB / "results" / "reid_embedding.npz", allow_pickle=True)
        emb = {k: zz[k] for k in zz.files}
        import torch
        m = RD.Reid().cuda().eval()
        m.load_state_dict(torch.load(SUB / "runs" / "reid_resnet18" / "best.pt"))

        def reid_fn(crops):
            out = []
            with torch.no_grad():
                for i in range(0, len(crops), 256):
                    out.append(m(RD.ke_tensor(crops[i:i + 256], False, "cuda"))
                               .float().cpu().numpy())
            return np.concatenate(out)

    prob = PP.bangun_prob_prediksi({k: ids[k] for k in ["train", "val", "test"]}) \
        if cfg["pakai_prob"] else None
    # WAJIB: konstanta pergeseran arah-putar hidup sebagai global di
    # penaut_pertandan dan hanya terisi saat skrip itu dijalankan langsung.
    # Tanpa baris ini, fitur arah TIDAK aktif di sini dan hasilnya diam-diam
    # kembali ke fitur lama — terlihat sebagai angka yang identik persis.
    PP.HARAP = PP.hitung_harapan_geser(ids["train"])
    print(f"konstanta arah-putar dipas di train: {len(PP.HARAP)} entri")
    print("melatih penaut...")
    Xtr, ytr = PP.pasangan(ids["train"], desk, True, emb, cfg["pakai_kelas"], prob)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=PP.SEED).fit(Xtr, ytr)

    pohon = {s: [EP.muat_pohon(t) for t in ids[s]] for s in ["val", "test"]}
    z = {s: np.load(SUB / "results" / f"pred_skorpenuh_{s}.npz", allow_pickle=True)
         for s in ["val", "test"]}

    hasil = {"penaut": nama_var, "ambang_mode_A": ambang, "conf": conf,
             "skema": skema, "tau": list(tau), "split": {}}
    for s in ["val", "test"]:
        hasil["split"][s] = {}
        for mode in ["A_ambang_tetap", "B_rem_M01", "C_rem_oracle"]:
            print(f"  {s} / {mode} ...", flush=True)
            r = nilai_mode(pohon[s], z[s], clf, cfg, conf, skema, tau, reid_fn,
                           mode, ambang, predict_m01)
            hasil["split"][s][mode] = r
            print(f"    kelompok {r['n_kelompok']} vs {r['n_tandan_sebenarnya']} "
                  f"(rasio {r['rasio_kelompok']}) | multi {r['frac_multi']:.1%} | "
                  f"R0 {r['aturan']['R0']['akurasi']} R4 {r['aturan']['R4']['akurasi']} | "
                  f"F1 {r['penautan']['f1']}")

    t = hasil["split"]["test"]
    hasil["ringkas_test"] = {
        m: {"R0": t[m]["aturan"]["R0"]["akurasi"],
            "R4": t[m]["aturan"]["R4"]["akurasi"],
            "frac_multi": t[m]["frac_multi"],
            "f1_penautan": t[m]["penautan"]["f1"],
            "delta_R4_vs_R0_pp": t[m]["boot_R4_vs_R0"]["delta_pp"],
            "ci95": t[m]["boot_R4_vs_R0"]["ci95_pp"]}
        for m in t}
    hasil["pembanding_pipeline_lama_test"] = 0.7203
    Path(args.keluaran).write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(hasil["ringkas_test"], indent=1, ensure_ascii=False))
    print(f"-> {args.keluaran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
