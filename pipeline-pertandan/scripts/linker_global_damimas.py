"""Penaut DAMIMAS di ruang deteksi dengan fitur konteks dan asosiasi global.

Perbaikan terhadap PT-E-017 ada pada tiga tempat yang sebelumnya bercampur:

1. seluruh fitting hanya memakai pohon DAMIMAS;
2. fitur pasangan melihat ranking kandidat di dua sisi, sehingga kompetisi
   tidak baru muncul setelah skor independen telanjur dibuat;
3. perakit kandidat mencakup complete/average-link dan correlation clustering
   biner. Perakit global menghukum sebuah klaster bila pasangan internal lain
   menyangkal satu edge yang tampak kuat—kelemahan utama union-find serakah.

TRAIN dipakai memasang prior dan model sisi. Model/perakit/ambang dipilih pada
VAL. Cache dan label TEST baru dibangun setelah konfigurasi terkunci.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score


sys.path.insert(0, str(Path(__file__).parent))
import eval_endtoend as EE  # noqa: E402
import eval_pertandan as EP  # noqa: E402
import penaut_pertandan as PP  # noqa: E402
import reid_pertandan as RD  # noqa: E402


SUB = Path(__file__).resolve().parents[1]
GRID = (.05, .10, .15, .20, .25, .30, .35, .40, .50, .60, .70, .80, .90)


def prior_train(ids):
    dx, dy = defaultdict(list), defaultdict(list)
    for tree in ids:
        nv, box = PP.muat_pohon(tree)
        for a, b in itertools.combinations(box, 2):
            if a["s"] == b["s"] or a["bid"] is None or a["bid"] != b["bid"]:
                continue
            if a["s"] > b["s"]:
                a, b = b, a
            k = (nv, (b["s"] - a["s"]) % nv)
            dx[k].append(b["cx"] - a["cx"])
            dy[k].append(b["cy"] - a["cy"])
    out = {}
    for k in dx:
        x, y = np.asarray(dx[k]), np.asarray(dy[k])
        mx, my = float(np.median(x)), float(np.median(y))
        sx = max(float(1.4826 * np.median(np.abs(x - mx))), .035)
        sy = max(float(1.4826 * np.median(np.abs(y - my))), .025)
        out[k] = (mx, my, sx, sy, len(x))
    return out


def muat_reid(path: Path, device: str):
    model = RD.Reid().to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    @torch.inference_mode()
    def fn(crops):
        out = []
        for i in range(0, len(crops), 256):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                z = model(RD.ke_tensor(crops[i:i + 256], False, device))
            out.append(z.float().cpu().numpy())
        return np.concatenate(out)
    return fn


def konteks_simpul(det, nv):
    """Statistik relatif dalam satu sisi; tak bergantung GT."""
    C = np.zeros((len(det), 12), np.float32)
    per_sisi = defaultdict(list)
    for i, d in enumerate(det):
        per_sisi[d["s"]].append(i)
    for _s, idx in per_sisi.items():
        vals = np.asarray([[det[i]["cx"], det[i]["cy"],
                            math.log(max(det[i]["w"] * det[i]["h"], 1e-9))]
                           for i in idx])
        med = np.median(vals, 0)
        scale = np.maximum(1.4826 * np.median(np.abs(vals - med), 0), .03)
        for col in range(3):
            urut = np.argsort(np.argsort(vals[:, col], kind="stable"), kind="stable")
            rank = urut / max(len(idx) - 1, 1)
            for j, i in enumerate(idx):
                C[i, col] = rank[j]
                C[i, 3 + col] = (vals[j, col] - med[col]) / scale[col]
        ys = vals[:, 1]
        for j, i in enumerate(idx):
            jarak = np.sort(np.abs(ys - ys[j]))
            C[i, 6] = jarak[1] if len(jarak) > 1 else 1.
            C[i, 7] = len(idx) / 20.
            C[i, 8] = det[i]["conf"]
            C[i, 9] = det[i]["tepi"]
            C[i, 10] = det[i]["s"] / max(nv - 1, 1)
            C[i, 11] = nv / 8.
    return C


def fitur_rich(det, nv, prior):
    ctx = konteks_simpul(det, nv)
    V = []
    for d, c in zip(det, ctx):
        area = max(d["w"] * d["h"], 1e-9)
        V.append([d["cx"], d["cy"], d["w"], d["h"], math.log(area),
                  d["w"] / max(d["h"], 1e-9), d["conf"], d["tepi"],
                  *d["p"], *c])

    pairs, base, heuristic, y = [], [], [], []
    for i, j in itertools.combinations(range(len(det)), 2):
        if det[i]["s"] == det[j]["s"]:
            continue
        a, b = (det[i], det[j]) if det[i]["s"] < det[j]["s"] else (det[j], det[i])
        ia, ib = (i, j) if det[i]["s"] < det[j]["s"] else (j, i)
        d = (b["s"] - a["s"]) % nv
        mu_x, mu_y, sx, sy, _n = prior.get((nv, d), (0., 0., .20, .08, 0))
        rdx = (b["cx"] - a["cx"] - mu_x) / sx
        rdy = (b["cy"] - a["cy"] - mu_y) / sy
        pa, pb = a["p"], b["p"]
        js = .5 * (np.sum(pa * np.log(np.clip(pa / np.clip((pa + pb) / 2, 1e-8, 1),
                                                  1e-8, 1e8))) +
                   np.sum(pb * np.log(np.clip(pb / np.clip((pa + pb) / 2, 1e-8, 1),
                                                  1e-8, 1e8))))
        f = EE.fitur_det(a, b, nv, True, False, True)
        f += [rdx, abs(rdx), rdy, abs(rdy), rdx * rdx + rdy * rdy,
              abs(ctx[ia, 0] - ctx[ib, 0]), abs(ctx[ia, 1] - ctx[ib, 1]),
              abs(ctx[ia, 2] - ctx[ib, 2]),
              abs(ctx[ia, 3] - ctx[ib, 3]), abs(ctx[ia, 4] - ctx[ib, 4]),
              min(a["conf"], b["conf"]), max(a["conf"], b["conf"]),
              a["conf"] * b["conf"], float(np.abs(pa - pb).sum()), float(js),
              float(d == 1 or d == nv - 1), float(d == nv // 2)]
        pairs.append((i, j)); base.append(f)
        heuristic.append(abs(rdx) + abs(rdy) + .5 * abs(ctx[ia, 1] - ctx[ib, 1]))
        y.append(int(a["bid"] is not None and a["bid"] == b["bid"]))

    # Rank/margin kandidat dihitung per simpul DAN target sisi. Ini membuat
    # "a paling cocok dengan siapa?" menjadi fitur model, bukan keputusan hilir.
    comp = np.zeros((len(pairs), 7), np.float32)
    grup = defaultdict(list)
    for e, (i, j) in enumerate(pairs):
        grup[(i, det[j]["s"])].append(e)
        grup[(j, det[i]["s"])].append(e)
    per_edge = defaultdict(list)
    for edges in grup.values():
        vals = np.asarray([heuristic[e] for e in edges])
        order = np.argsort(vals)
        ranks = np.empty(len(edges), int); ranks[order] = np.arange(len(edges))
        best = vals[order[0]]
        second = vals[order[1]] if len(order) > 1 else best + 3.
        for q, e in enumerate(edges):
            margin = (second - vals[q]) if ranks[q] == 0 else (best - vals[q])
            per_edge[e].append((ranks[q] / max(len(edges) - 1, 1),
                                float(ranks[q] == 0), margin, len(edges) / 20.))
    for e in range(len(pairs)):
        a, b = per_edge[e]
        comp[e] = [a[0], b[0], a[1], b[1], a[2], b[2], min(a[3], b[3])]
    E = np.c_[np.asarray(base, np.float32), comp]
    return (np.asarray(V, np.float32), np.asarray(pairs, np.int32), E,
            np.asarray(y, np.int8))


def bangun_graf(split, ids, pred_path, conf, prior, reid_fn, cache):
    pred_path = Path(pred_path).resolve()
    st = pred_path.stat()
    signature = {"path": str(pred_path), "size": st.st_size,
                 "mtime_ns": st.st_mtime_ns, "conf": float(conf)}
    if cache.exists():
        obj = joblib.load(cache)
        if (obj.get("split") == split and obj.get("ids") == ids and
                obj.get("signature_prediksi") == signature):
            return obj["graf"]
        print(f"  abaikan cache stale: {cache}", flush=True)
    z = np.load(pred_path, allow_pickle=True)
    graphs = []
    for n, tree in enumerate(ids, 1):
        P = EP.muat_pohon(tree)
        det = EE.deteksi_pohon(P, z, conf, reid_fn)
        if det:
            V, pairs, E, y = fitur_rich(det, P["n_sisi"], prior)
            if len(pairs):
                graphs.append({"tree": tree, "nv": P["n_sisi"], "kotak": det,
                               "V": V, "pairs": pairs, "E": E, "y": y, "P": P})
        if n % 100 == 0:
            print(f"  cache {split}: {n}/{len(ids)}", flush=True)
    joblib.dump({"split": split, "ids": ids,
                 "signature_prediksi": signature,
                 "graf": graphs}, cache, compress=3)
    return graphs


def model_kandidat():
    return {
        "hist_l15": HistGradientBoostingClassifier(
            max_iter=350, learning_rate=.06, max_leaf_nodes=15,
            l2_regularization=5., class_weight="balanced", random_state=42),
        "hist_l31": HistGradientBoostingClassifier(
            max_iter=350, learning_rate=.05, max_leaf_nodes=31,
            l2_regularization=10., class_weight="balanced", random_state=43),
        "hist_l63": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.04, max_leaf_nodes=63,
            l2_regularization=15., class_weight="balanced", random_state=44),
        "extra_l3": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=3, max_features=.8,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "extra_l8": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=8, max_features=1.,
            class_weight="balanced", n_jobs=-1, random_state=43),
    }


def score_model(model, graphs):
    return [model.predict_proba(g["E"])[:, 1].astype(np.float32) for g in graphs]


def gabung_score(weights, scores):
    return [sum(w * scores[n][i] for n, w in weights.items())
            for i in range(len(next(iter(scores.values()))))]


def matriks_score(g, flat):
    n = len(g["kotak"])
    S = np.zeros((n, n), np.float32)
    for (i, j), s in zip(g["pairs"], flat):
        S[i, j] = S[j, i] = s
    return S


def labels_union(n, pasangan):
    uf = PP.UF(n)
    for i, j in pasangan:
        uf.gabung(int(i), int(j))
    roots = [uf.cari(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.asarray([remap[r] for r in roots], int)


def labels_dari_uf(uf, n):
    roots = [uf.cari(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.asarray([remap[r] for r in roots], int)


def rakit_hungarian_grid(g, flat, thresholds, max_mode):
    det, n = g["kotak"], len(g["kotak"])
    S = matriks_score(g, flat)
    ambang_min = min(thresholds)
    per_sisi = defaultdict(list)
    for i, d in enumerate(det):
        per_sisi[d["s"]].append(i)
    kandidat = []
    for sa, sb in itertools.combinations(sorted(per_sisi), 2):
        A, B = per_sisi[sa], per_sisi[sb]
        M = S[np.ix_(A, B)]
        for x, y in zip(*linear_sum_assignment(-M)):
            if M[x, y] >= ambang_min:
                kandidat.append((float(M[x, y]), A[x], B[y]))
    kandidat.sort(reverse=True)
    uf = PP.UF(n); sisi = {i: {d["s"]} for i, d in enumerate(det)}
    ukuran = {i: 1 for i in range(n)}
    maks = g["nv"] if max_mode == "penuh" else (3 if g["nv"] == 4 else 6)
    out, pos = {}, 0
    for ambang in sorted(thresholds, reverse=True):
        while pos < len(kandidat) and kandidat[pos][0] >= ambang:
            _s, i, j = kandidat[pos]; pos += 1
            a, b = uf.cari(i), uf.cari(j)
            if a == b or sisi[a] & sisi[b] or ukuran[a] + ukuran[b] > maks:
                continue
            sa, sb, ua, ub = sisi[a], sisi[b], ukuran[a], ukuran[b]
            uf.gabung(a, b); r = uf.cari(a)
            ukuran[r] = ua + ub; sisi[r] = sa | sb
        out[ambang] = labels_dari_uf(uf, n)
    return out


def rakit_hungarian(g, flat, ambang, max_mode):
    return rakit_hungarian_grid(g, flat, [ambang], max_mode)[ambang]


def rakit_aglom_grid(g, flat, thresholds, metode, max_mode):
    n, det = len(g["kotak"]), g["kotak"]
    S = matriks_score(g, flat)
    clusters = {i: [i] for i in range(n)}
    maks = g["nv"] if max_mode == "penuh" else (3 if g["nv"] == 4 else 6)
    best = None
    out = {}
    for ambang in sorted(thresholds, reverse=True):
        while True:
            if best is None:
                keys = list(clusters)
                for ai, a in enumerate(keys):
                    A = clusters[a]; sisi_a = {det[i]["s"] for i in A}
                    for b in keys[ai + 1:]:
                        B = clusters[b]
                        if (len(A) + len(B) > maks or
                                sisi_a & {det[j]["s"] for j in B}):
                            continue
                        q = S[np.ix_(A, B)].ravel()
                        if metode == "min":
                            s = float(q.min())
                        elif metode == "top2":
                            s = float(np.sort(q)[-min(2, len(q)):].mean())
                        else:
                            s = float(q.mean())
                        if best is None or s > best[0]:
                            best = (s, a, b)
            if best is None or best[0] < ambang:
                break
            _s, a, b = best
            clusters[a] += clusters.pop(b)
            best = None
        lab = np.empty(n, int)
        for k, anggota in enumerate(clusters.values()):
            lab[anggota] = k
        out[ambang] = lab
    return out


def rakit_aglom(g, flat, ambang, metode, max_mode):
    return rakit_aglom_grid(g, flat, [ambang], metode, max_mode)[ambang]


def ilp_komponen(nodes, S, sisi, ambang):
    m = len(nodes)
    if m <= 1:
        return []
    if m > 34:
        return None
    pairs = list(itertools.combinations(range(m), 2))
    pos = {p: i for i, p in enumerate(pairs)}
    ub = np.ones(len(pairs))
    coef = np.zeros(len(pairs))
    lt = math.log(ambang / (1 - ambang))
    for k, (i, j) in enumerate(pairs):
        if sisi[nodes[i]] == sisi[nodes[j]]:
            ub[k] = 0.
        p = float(np.clip(S[nodes[i], nodes[j]], 1e-5, 1 - 1e-5))
        coef[k] = math.log(p / (1 - p)) - lt
    rr, cc, vv, row = [], [], [], 0
    for i, j, k in itertools.combinations(range(m), 3):
        ij, ik, jk = pos[(i, j)], pos[(i, k)], pos[(j, k)]
        for a, b, c in ((ij, jk, ik), (ij, ik, jk), (ik, jk, ij)):
            rr += [row, row, row]; cc += [a, b, c]; vv += [1., 1., -1.]; row += 1
    A = coo_matrix((vv, (rr, cc)), shape=(row, len(pairs))).tocsr()
    res = milp(c=-coef, integrality=np.ones(len(pairs)),
               bounds=Bounds(np.zeros(len(pairs)), ub),
               constraints=LinearConstraint(A, -np.inf, np.ones(row)),
               options={"time_limit": 3., "mip_rel_gap": .001, "presolve": True})
    if res.x is None:
        return None
    return [(nodes[i], nodes[j]) for x, (i, j) in zip(res.x, pairs) if x > .5]


def rakit_ilp(g, flat, ambang, _metode, _max_mode):
    n, S = len(g["kotak"]), matriks_score(g, flat)
    # Komponen longgar memangkas ILP tanpa mengubah kandidat yang masuk akal.
    uf = PP.UF(n)
    cutoff = min(.08, ambang / 3)
    for (i, j), p in zip(g["pairs"], flat):
        if p >= cutoff:
            uf.gabung(int(i), int(j))
    comp = defaultdict(list)
    for i in range(n):
        comp[uf.cari(i)].append(i)
    pilih = []
    sisi = [d["s"] for d in g["kotak"]]
    for nodes in comp.values():
        q = ilp_komponen(nodes, S, sisi, ambang)
        if q is None:
            # Komponen luar biasa besar: fallback average-link tetap global
            sub = {"kotak": [g["kotak"][i] for i in nodes], "nv": g["nv"]}
            pairs, sf = [], []
            for a, b in itertools.combinations(range(len(nodes)), 2):
                if sisi[nodes[a]] != sisi[nodes[b]]:
                    pairs.append((a, b)); sf.append(S[nodes[a], nodes[b]])
            sub["pairs"] = np.asarray(pairs, int)
            lab = rakit_aglom(sub, np.asarray(sf), ambang, "mean", "penuh")
            for group in set(lab):
                mem = [nodes[i] for i in np.flatnonzero(lab == group)]
                pilih += list(itertools.combinations(mem, 2))
        else:
            pilih += q
    return labels_union(n, pilih)


def nilai_labels(graphs, labels):
    tp = fp = fn = 0
    ari, coverage_ok, coverage_n, all_ok, all_n = [], 0, 0, 0, 0
    pool_palsu = pool_multi = 0
    count_err = []
    for g, lab in zip(graphs, labels):
        det = g["kotak"]
        gt = [d["bid"] if d["bid"] is not None else -100000 - i
              for i, d in enumerate(det)]
        for i, j in itertools.combinations(range(len(det)), 2):
            if det[i]["s"] == det[j]["s"]:
                continue
            pred = lab[i] == lab[j]
            benar = det[i]["bid"] is not None and det[i]["bid"] == det[j]["bid"]
            tp += pred and benar; fp += pred and not benar; fn += benar and not pred
        ari.append(adjusted_rand_score(gt, lab))
        pools = defaultdict(list)
        for i, l in enumerate(lab):
            pools[int(l)].append(i)
        for mem in pools.values():
            if len(mem) >= 2:
                pool_multi += 1
                pool_palsu += all(det[i]["bid"] is None for i in mem)
        per_bid = defaultdict(list)
        for i, d in enumerate(det):
            if d["bid"] is not None:
                per_bid[d["bid"]].append(i)
        for mem in per_bid.values():
            if len(mem) >= 2:
                coverage_n += 1
                coverage_ok += max(Counter(lab[i] for i in mem).values()) >= 2
        multigt = {b: c for b, c in Counter(
            a["bid"] for side in g["P"]["sisi"] for a in side["gt"]
            if a["bid"] is not None).items() if c >= 2}
        all_n += len(multigt)
        for bid in multigt:
            mem = per_bid.get(bid, [])
            all_ok += (len(mem) >= 2 and
                       max(Counter(lab[i] for i in mem).values(), default=0) >= 2)
        count_err.append(abs(len(pools) - len(g["P"]["tandan"])))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    out = {"presisi": p, "recall": r, "f1": f1,
           "ari": float(np.mean(ari)),
           "cakupan_atas_terdeteksi": coverage_ok / max(coverage_n, 1),
           "cakupan_atas_semua": all_ok / max(all_n, 1),
           "frac_pool_palsu": pool_palsu / max(pool_multi, 1),
           "mae_jumlah_pool": float(np.mean(count_err)),
           "n_bisa": coverage_n, "n_semua": all_n}
    out["utility"] = (.35 * f1 + .25 * out["ari"] +
                      .20 * out["cakupan_atas_terdeteksi"] + .10 * p +
                      .10 * (1 - out["frac_pool_palsu"]))
    return out


def nilai(graphs, scores, assembler, ambang, metode="mean", max_mode="observasi"):
    labels = []
    for g, s in zip(graphs, scores):
        if assembler == "hungarian":
            lab = rakit_hungarian(g, s, ambang, max_mode)
        elif assembler == "ilp":
            lab = rakit_ilp(g, s, ambang, metode, max_mode)
        else:
            lab = rakit_aglom(g, s, ambang, metode, max_mode)
        labels.append(lab)
    return nilai_labels(graphs, labels)


def nilai_grid(graphs, scores, assembler, thresholds, metode, max_mode):
    """Nilai semua ambang dari satu lintasan merge yang persis ekuivalen."""
    labels = {t: [] for t in thresholds}
    for g, s in zip(graphs, scores):
        if assembler == "hungarian":
            q = rakit_hungarian_grid(g, s, thresholds, max_mode)
        else:
            q = rakit_aglom_grid(g, s, thresholds, metode, max_mode)
        for t in thresholds:
            labels[t].append(q[t])
    return {t: nilai_labels(graphs, labels[t]) for t in thresholds}


def serial(x):
    if isinstance(x, dict):
        return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-train", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_train.npz")
    ap.add_argument("--pred-val", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_val.npz")
    ap.add_argument("--pred-test", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_test.npz")
    ap.add_argument("--reid", type=Path,
                    default=SUB / "runs" / "reid_resnet18_damimas" / "best.pt")
    ap.add_argument("--conf", type=float, default=.10)
    ap.add_argument("--tanpa-ilp", action="store_true")
    ap.add_argument("--output", type=Path,
                    default=SUB / "results" / "damimas_linker_global.json")
    args = ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    man = PP.muat_manifest()
    ids = {s: sorted(t for t, v in man.items()
                     if v == s and t.startswith("DAMIMAS_"))
           for s in ("train", "val", "test")}
    prior = prior_train(ids["train"])
    PP.HARAP = {k: v[0] for k, v in prior.items()}
    reid_fn = muat_reid(args.reid, device)
    cache_tag = args.reid.parent.name.replace("reid_resnet18_", "")
    graphs = {}
    for split, pred in (("train", args.pred_train), ("val", args.pred_val)):
        pred_tag = pred.stem.replace("pred_", "")
        cache = SUB / "results" / (
            f"cache_linker_damimas_{cache_tag}_{pred_tag}_{split}.joblib")
        graphs[split] = bangun_graf(split, ids[split], pred, args.conf,
                                    prior, reid_fn, cache)
        print(f"{split}: {len(graphs[split])} graf", flush=True)

    X = np.concatenate([g["E"] for g in graphs["train"]])
    y = np.concatenate([g["y"] for g in graphs["train"]])
    print(f"latih sisi: {len(y)} pasangan, {int(y.sum())} positif ({y.mean():.3%}), "
          f"dim={X.shape[1]}", flush=True)
    models, scores_val, auc = {}, {}, {}
    yv = np.concatenate([g["y"] for g in graphs["val"]])
    for nama, spec in model_kandidat().items():
        print(f"fit {nama}", flush=True)
        models[nama] = copy.deepcopy(spec).fit(X, y)
        scores_val[nama] = score_model(models[nama], graphs["val"])
        auc[nama] = float(roc_auc_score(yv, np.concatenate(scores_val[nama])))
        print(f"  AUC val {auc[nama]:.5f}", flush=True)

    top = sorted(auc, key=auc.get, reverse=True)[:3]
    sumber = {n: {n: 1.} for n in top}
    for a, b in itertools.combinations(top, 2):
        sumber[f"{a}+{b}"] = {a: .5, b: .5}
    sumber["mean_top3"] = {n: 1 / len(top) for n in top}
    score_sumber = {n: gabung_score(w, scores_val) for n, w in sumber.items()}

    ranking = []
    methods = [("hungarian", "mean"), ("aglom", "mean"),
               ("aglom", "min"), ("aglom", "top2")]
    for nama, sv in score_sumber.items():
        for assembler, metode in methods:
            for max_mode in ("observasi", "penuh"):
                semua_metrik = nilai_grid(graphs["val"], sv, assembler,
                                           GRID, metode, max_mode)
                for threshold, m in semua_metrik.items():
                    ranking.append({"sumber": nama, "bobot_skor": sumber[nama],
                                    "assembler": assembler, "metode": metode,
                                    "max_mode": max_mode, "ambang": threshold,
                                    "metrik": m})
        print(f"sapu {nama} selesai", flush=True)
    ranking.sort(key=lambda q: q["metrik"]["utility"], reverse=True)

    # ILP mahal; jalankan hanya untuk tiga score terbaik dan grid ringkas.
    if not args.tanpa_ilp:
        for nama in [q["sumber"] for q in ranking[:3]]:
            if any(r["sumber"] == nama and r["assembler"] == "ilp" for r in ranking):
                continue
            for threshold in (.10, .20, .30, .40, .50, .60):
                m = nilai(graphs["val"], score_sumber[nama], "ilp", threshold)
                ranking.append({"sumber": nama, "bobot_skor": sumber[nama],
                                "assembler": "ilp", "metode": "global",
                                "max_mode": "penuh", "ambang": threshold,
                                "metrik": m})
            print(f"ILP {nama} selesai", flush=True)
        ranking.sort(key=lambda q: q["metrik"]["utility"], reverse=True)

    terbaik = ranking[0]
    print("TERKUNCI DI VAL:", json.dumps(serial(terbaik), indent=2), flush=True)
    # TEST baru dibangun dan diberi skor setelah konfigurasi final terkunci.
    graphs["test"] = bangun_graf(
        "test", ids["test"], args.pred_test, args.conf, prior, reid_fn,
        SUB / "results" / (
            f"cache_linker_damimas_{cache_tag}_"
            f"{args.pred_test.stem.replace('pred_', '')}_test.joblib"))
    need = terbaik["bobot_skor"]
    st = {n: score_model(models[n], graphs["test"]) for n in need}
    score_test = gabung_score(need, st)
    mt = nilai(graphs["test"], score_test, terbaik["assembler"],
               terbaik["ambang"], terbaik["metode"], terbaik["max_mode"])
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "prior/model TRAIN; model+assembler+ambang VAL; TEST sekali",
        "detektor_linker": {"train": str(args.pred_train), "val": str(args.pred_val),
                            "test": str(args.pred_test), "conf": args.conf},
        "reid": str(args.reid),
        "n": {s: {"pohon": len(ids[s]), "graf": len(graphs[s])} for s in graphs},
        "prior": {f"{k[0]}|{k[1]}": list(v) for k, v in sorted(prior.items())},
        "pasangan_train": {"n": len(y), "positif": int(y.sum()),
                            "dim_fitur": X.shape[1]},
        "auc_val": auc,
        "terpilih_di_val": terbaik,
        "test": mt,
        "ranking_val": ranking[:30],
        "terbaik_per_metrik_val": {
            k: max(ranking, key=lambda q: (-q["metrik"][k]
                                           if k in ("frac_pool_palsu", "mae_jumlah_pool")
                                           else q["metrik"][k]))
            for k in ("f1", "ari", "cakupan_atas_terdeteksi",
                      "frac_pool_palsu", "mae_jumlah_pool")},
        "detik": time.time() - t0,
    }
    args.output.write_text(json.dumps(serial(hasil), indent=2, ensure_ascii=False))
    run = SUB / "runs" / "linker_global_damimas"
    run.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "prior": prior, "config": terbaik,
                 "fitur_dim": X.shape[1]}, run / "model.joblib", compress=3)
    print(json.dumps({"val": terbaik["metrik"], "test": mt},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
