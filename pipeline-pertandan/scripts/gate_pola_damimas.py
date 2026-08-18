"""PT-E-036 — Gerbang atas POLA PERSELISIHAN, bukan keyakinan. Nol GPU.

PT-E-035 mengukur keyakinan hampir buta soal siapa yang benar (korelasi +0,1185;
memilih anggota paling yakin 0,5711 lawan menebak acak 0,5435). Pertanyaan yang
belum dijawab: apakah sinyalnya ada di POLA -- yaitu identitas siapa-bilang-apa?

Kalau `corn224` bilang B2 sementara `convnext224` bilang B3, mungkin salah satu
punya bias sistematis yang bisa dipelajari, terlepas dari seberapa yakin mereka.

Fitur gerbang (semuanya dari keluaran anggota, nol inferensi baru):
  - kelas prediksi tiap anggota (one-hot)      M*K
  - keyakinan tiap anggota                     M
  - probabilitas rata-rata                     K
  - jumlah tampak                              1
  - berapa anggota sepakat dengan mayoritas    1

Target: kelas benar. Model: gradient boosting kecil.

**Evaluasi HANYA lewat CV 5-fold tingkat pohon di dalam VAL.** TEST tidak
disentuh kecuali CV menunjukkan gerbang mengalahkan rata-rata biasa. Ini penting:
`moe_classifier` DAMIMAS sudah pernah merosot pada tugas serupa, jadi beban
pembuktian ada pada gerbang, bukan pada baseline.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

R = Path("pipeline-pertandan/results"); K = 4; SEED = 0
A = {"convnext224": ("damimas_classifier_hibrida_convnext224_s42_pred.npz","bunch_prob"),
     "convnext128": ("damimas_classifier_hibrida_convnext_tiny_s42_pred.npz","bunch_prob"),
     "klasik":      ("damimas_classifier_klasik_pred.npz","bunch_prob"),
     "set_transformer": ("damimas_set_transformer_convnext_tiny_s42_pred.npz","prob"),
     "corn224":     ("damimas_classifier_corn_s42_pred.npz","bunch_prob")}
def L(f,k,s):
    z=np.load(R/f,allow_pickle=True); P=np.asarray(z[f"{s}_{k}"],float)
    return P/np.clip(P.sum(1,keepdims=True),1e-9,None)
P={s:np.stack([L(f,k,s) for f,k in A.values()]) for s in ("val","test")}
ref=np.load(R/A["convnext224"][0],allow_pickle=True)
Y={s:np.asarray(ref[f"{s}_bunch_y"],int) for s in ("val","test")}
tv=np.asarray(ref["val_bunch_tree"]); tt=np.asarray(ref["test_bunch_tree"])
nv={s:np.asarray(ref[f"{s}_bunch_nview"],int) for s in ("val","test")}

def fitur(s):
    Q=P[s]; M,N,_=Q.shape
    yh=Q.argmax(2)                                   # (M,N)
    oh=np.zeros((N,M*K),np.float32)
    for m in range(M): oh[np.arange(N), m*K+yh[m]]=1
    conf=Q.max(2).T                                  # (N,M)
    rata=Q.mean(0)                                   # (N,K)
    mayor=np.array([np.bincount(yh[:,i],minlength=K).max() for i in range(N)],float)
    return np.c_[oh, conf, rata, nv[s][:,None], mayor[:,None]]

Xv, Xt = fitur("val"), fitur("test")
pohon=np.unique(tv); rng=np.random.default_rng(SEED)
fmap={t:i%5 for i,t in enumerate(rng.permutation(pohon))}
fid=np.array([fmap[t] for t in tv])

ok_gate, ok_rata = [], []
for f in range(5):
    tr, te = fid!=f, fid==f
    g=HistGradientBoostingClassifier(max_iter=200,learning_rate=.08,
        max_leaf_nodes=15,random_state=SEED).fit(Xv[tr],Y["val"][tr])
    ok_gate.append(g.predict(Xv[te])==Y["val"][te])
    ok_rata.append(P["val"][:,te].mean(0).argmax(1)==Y["val"][te])
g_cv=float(np.concatenate(ok_gate).mean()); r_cv=float(np.concatenate(ok_rata).mean())
print(f"CV 5-fold tingkat pohon DI DALAM VAL:")
print(f"  rata-rata probabilitas : {r_cv:.4f}")
print(f"  gerbang pola           : {g_cv:.4f}   ({(g_cv-r_cv)*100:+.2f} pp)")

hasil={"pt_e":"036","cv_val_rata":round(r_cv,4),"cv_val_gerbang":round(g_cv,4),
       "delta_cv_pp":round((g_cv-r_cv)*100,2),"test_dibuka":False}
if g_cv > r_cv + 1e-9:
    print("\n  CV menunjukkan gerbang menang -> TEST dibuka sekali")
    g=HistGradientBoostingClassifier(max_iter=200,learning_rate=.08,
        max_leaf_nodes=15,random_state=SEED).fit(Xv,Y["val"])
    yh=g.predict(Xt); acc=float((yh==Y["test"]).mean())
    m1=nv["test"]==1
    hasil.update(test_dibuka=True,test=round(acc,4),
        test_1view=round(float((yh[m1]==Y["test"][m1]).mean()),4),
        test_multi=round(float((yh[~m1]==Y["test"][~m1]).mean()),4))
    np.savez_compressed(R/"pt_e_036_gate_pred.npz",test_yhat=yh,test_y=Y["test"],test_tree=tt)
    print(f"  TEST: {acc:.4f}")
else:
    print("\n  CV TIDAK menunjukkan keunggulan -> TEST TIDAK dibuka (menghindari peeking)")
hasil["target_IDEA"]=0.80
(R/"pt_e_036_gate.json").write_text(json.dumps(hasil,indent=1,ensure_ascii=False))
print("\n"+json.dumps(hasil,indent=1,ensure_ascii=False))
