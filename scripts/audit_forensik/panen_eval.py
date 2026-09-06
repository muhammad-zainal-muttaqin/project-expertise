"""Tahap 2 Pipeline Panen: penalaan ambang pada VALIDATION, lalu TEST satu kali."""
import json, pickle, itertools, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1001/-workspace/ebcbd941-6775-4113-b727-404085458263/scratchpad")
from panen_pipeline import evaluate, GT, split, RES, EMPAT_SISI

D = pickle.load(open(f"{RES}/dets.pkl", "rb"))
E = pickle.load(open(f"{RES}/edge_model.pkl", "rb"))
VAL = [t for t in D["val"] if t in EMPAT_SISI]
TEST = [t for t in D["test"] if t in EMPAT_SISI]
print(f"VAL {len(VAL)} pohon · TEST {len(TEST)} pohon", flush=True)

GRID = dict(
    det_conf=[0.20, 0.30, 0.40],
    link_thr=[0.30, 0.45, 0.60],
    max_size=[2, 3],
    single_thr=[0.30, 0.45, 0.60],
)
THR = dict(t_coarse=[1.30, 1.50, 1.70], t_b1b2=[0.45, 0.60, 0.75],
           t_b3b4=[2.30, 2.50, 2.70])


def objective(r):
    """Prioritas pengguna: cacah benar dan sedikit yang terlewat, kelas tepat."""
    return (1.5 * r["count_matang"]["within1"] + 1.0 * r["count_total"]["within1"]
            + 1.0 * r["physical_f1"] + 1.0 * r["class2_acc"] + 0.5 * r["class4_within1"])


best, log = None, []
for dc, lt, ms, st in itertools.product(GRID["det_conf"], GRID["link_thr"],
                                        GRID["max_size"], GRID["single_thr"]):
    r, _ = evaluate(D["val"], VAL, E, lt, ms, st, 1.5, 0.6, 2.5, dc)
    j = objective(r)
    log.append(dict(det_conf=dc, link_thr=lt, max_size=ms, single_thr=st, J=j,
                    f1=r["physical_f1"], mae=r["count_total"]["mae"],
                    w1=r["count_total"]["within1"], mw1=r["count_matang"]["within1"],
                    c2=r["class2_acc"]))
    if best is None or j > best[0]:
        best = (j, dc, lt, ms, st)
    print(f"  conf={dc} link={lt} size={ms} single={st} -> J={j:.4f} "
          f"F1={r['physical_f1']:.4f} MAE={r['count_total']['mae']:.3f} "
          f"±1={r['count_total']['within1']:.3f} matang±1={r['count_matang']['within1']:.3f}",
          flush=True)

_, DC, LT, MS, ST = best
print(f"\ntopologi terpilih: conf={DC} link={LT} max_size={MS} single={ST}", flush=True)

bt = None
for tc, t12, t34 in itertools.product(THR["t_coarse"], THR["t_b1b2"], THR["t_b3b4"]):
    r, _ = evaluate(D["val"], VAL, E, LT, MS, ST, tc, t12, t34, DC)
    j = objective(r)
    if bt is None or j > bt[0]:
        bt = (j, tc, t12, t34)
_, TC, T12, T34 = bt
print(f"ambang skor terpilih: kasar={TC} B1|B2={T12} B3|B4={T34}", flush=True)

rv, _ = evaluate(D["val"], VAL, E, LT, MS, ST, TC, T12, T34, DC)
rt, per_tree = evaluate(D["test"], TEST, E, LT, MS, ST, TC, T12, T34, DC)

profil = dict(det_conf=DC, link_thr=LT, max_size=MS, single_thr=ST,
              t_coarse=TC, t_b1b2=T12, t_b3b4=T34)
json.dump(dict(profil=profil, val=rv, test=rt, per_tree_test=per_tree, grid=log),
          open(f"{RES}/panen_results.json", "w"), indent=1)


def show(tag, r):
    print(f"\n=== {tag} ===")
    print(f"  F1 fisik      : {r['physical_f1']:.4f}  (P {r['precision']:.4f} / R {r['recall']:.4f})")
    for k, lab in [("count_total", "cacah total  "), ("count_matang", "cacah MATANG "),
                   ("count_belum", "cacah BELUM  ")]:
        c = r[k]
        print(f"  {lab} : MAE {c['mae']:.3f}  tepat {c['exact']:.3f}  ±1 {c['within1']:.3f}")
    print(f"  kelas 2       : akurasi {r['class2_acc']:.4f}  F1 {r['class2_f1']:.4f}")
    print(f"  kelas 4       : akurasi {r['class4_acc']:.4f}  ±1 {r['class4_within1']:.4f}  "
          f"makro-F1 {r['class4_macro_f1']:.4f}   (n={r['n_matched']})")


show("VALIDATION", rv)
show("TEST (dibuka satu kali)", rt)
print("\n--- pembanding hasil terkunci proyek pada test 953 ---")
print("  V2-E-045 : F1 0,8043 · MAE 1,393 · ±1 0,6148 · kelas4 0,7111")
print("  GSP      : F1 0,8387 · MAE 1,363 · ±1 0,6370 · kelas4 0,7442 · makro 0,6034")
print("PANEN EVAL DONE")
