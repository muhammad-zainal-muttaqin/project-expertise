"""PT-E-035 — Dynamic Ensemble Selection berparameter minimal untuk DAMIMAS.

PT-E-034 menunjukkan rata-rata berbobot mentok 0,7523 bahkan dengan bobot dipas
langsung di TEST, sementara oracle pilih-anggota 0,8739. Sisa 12,2 pp hanya bisa
diambil penggabung BERGANTUNG-MASUKAN (Cruz, Sabourin & Cavalcanti, "Dynamic
classifier selection: Recent advances and perspectives", Information Fusion 2018).

Kaveat yang mengikat rancangan ini: `moe_classifier` DAMIMAS sudah pernah mencoba
gating dan MEROSOT jadi "pilih klasik saja" (test 0,7234). Penyebab paling masuk
akal: gating dengan banyak parameter dilatih pada data seleksi kecil. Karena itu
seluruh varian di sini **nol atau satu parameter**, dan pemilihannya lewat CV
5-fold tingkat pohon di dalam VAL -- bukan fit VAL.

Varian:
  A  max-confidence DES  : pakai anggota paling yakin per tandan          0 param
  B  confidence-weighted : w_m ~ conf_m^T, rata-rata berbobot per tandan  1 param (T)
  C  agreement-gated     : kalau anggota sepakat pakai rata-rata; kalau   1 param
                           tidak, pakai anggota paling yakin
  D  rata-rata biasa     : pembanding (PT-E-029)                          0 param
"""
import json
import numpy as np
from pathlib import Path

R = Path("pipeline-pertandan/results"); K = 4; SEED = 0
A = {"convnext224": ("damimas_classifier_hibrida_convnext224_s42_pred.npz","bunch_prob"),
     "convnext128": ("damimas_classifier_hibrida_convnext_tiny_s42_pred.npz","bunch_prob"),
     "klasik":      ("damimas_classifier_klasik_pred.npz","bunch_prob"),
     "set_transformer": ("damimas_set_transformer_convnext_tiny_s42_pred.npz","prob"),
     "corn224":     ("damimas_classifier_corn_s42_pred.npz","bunch_prob")}

def load(f, k, s):
    z = np.load(R/f, allow_pickle=True); P = np.asarray(z[f"{s}_{k}"], float)
    return P/np.clip(P.sum(1, keepdims=True), 1e-9, None)

B = {s: np.stack([load(f, k, s) for f, k in A.values()]) for s in ("val", "test")}  # (M,N,K)
ref = np.load(R/A["convnext224"][0], allow_pickle=True)
Y = {s: np.asarray(ref[f"{s}_bunch_y"], int) for s in ("val", "test")}
tv = np.asarray(ref["val_bunch_tree"]); tt = np.asarray(ref["test_bunch_tree"])
nvt = np.asarray(ref["test_bunch_nview"], int)

def prediksi(P, varian, T=1.0):
    conf = P.max(2)                                   # (M,N)
    if varian == "D":
        return P.mean(0).argmax(1)
    if varian == "A":
        return P[conf.argmax(0), np.arange(P.shape[1])].argmax(1)
    if varian == "B":
        w = conf**T; w = w/np.clip(w.sum(0, keepdims=True), 1e-9, None)
        return (w[:, :, None]*P).sum(0).argmax(1)
    if varian == "C":
        rata = P.mean(0); setuju = (P.argmax(2) == P.argmax(2)[0]).mean(0)
        pilih = P[conf.argmax(0), np.arange(P.shape[1])]
        return np.where(setuju >= T, rata.argmax(1), pilih.argmax(1))
    raise ValueError(varian)

GRID = {"A": [1.0], "D": [1.0], "B": [0.5, 1, 2, 4, 8, 16], "C": [0.4, 0.6, 0.8, 1.01]}
pohon = np.unique(tv); rng = np.random.default_rng(SEED)
fmap = {t: i % 5 for i, t in enumerate(rng.permutation(pohon))}
fid = np.array([fmap[t] for t in tv])

cv = {}
for v, grid in GRID.items():
    for T in grid:
        ok = []
        for f in range(5):
            te = fid == f
            ok.append(prediksi(B["val"][:, te], v, T) == Y["val"][te])
        cv[(v, T)] = float(np.concatenate(ok).mean())
best = max(cv, key=cv.get)
print("CV 5-fold tingkat pohon di dalam VAL:")
for k in sorted(cv, key=lambda x: -cv[x]):
    print(f"  varian {k[0]} T={k[1]:<5} {cv[k]:.4f}" + ("   <- terpilih" if k == best else ""))

v, T = best
yhv = prediksi(B["val"], v, T); yh = prediksi(B["test"], v, T)
m1 = nvt == 1
p29 = np.load(R/"pt_e_029_ensemble_kelas_damimas_pred.npz", allow_pickle=True)
ba = (np.asarray(p29["test_yhat"], int) == Y["test"]).astype(float)
bb = (yh == Y["test"]).astype(float)
uniq = sorted(set(tt.tolist())); ip = {t: np.where(tt == t)[0] for t in uniq}
rr = np.random.default_rng(0); d = []
for _ in range(2000):
    s = rr.choice(len(uniq), len(uniq))
    jj = np.concatenate([ip[uniq[k]] for k in s])
    d.append(bb[jj].mean() - ba[jj].mean())
d = np.array(d)*100
hasil = {"pt_e": "035", "varian": v, "T": T,
         "cv_val": {f"{k[0]}_T{k[1]}": round(val, 4) for k, val in cv.items()},
         "val": round(float((yhv == Y["val"]).mean()), 4),
         "test": round(float(bb.mean()), 4),
         "test_1view": round(float((yh[m1] == Y["test"][m1]).mean()), 4),
         "test_multi": round(float((yh[~m1] == Y["test"][~m1]).mean()), 4),
         "vs_pt_e_029": {"delta_pp": round(float((bb.mean()-ba.mean())*100), 2),
                         "ci95": [round(float(np.percentile(d, 2.5)), 2),
                                  round(float(np.percentile(d, 97.5)), 2)],
                         "P(delta>0)": round(float((d > 0).mean()), 3)},
         "plafon_oracle": 0.8739, "target_IDEA": 0.80}
np.savez_compressed(R/"pt_e_035_des_pred.npz", test_yhat=yh, test_y=Y["test"], test_tree=tt)
(R/"pt_e_035_des.json").write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
print("\n=== TEST (dibuka sekali) ===")
print(json.dumps({k: hasil[k] for k in ("varian","T","val","test","test_1view","test_multi","vs_pt_e_029","target_IDEA")}, indent=1, ensure_ascii=False))
