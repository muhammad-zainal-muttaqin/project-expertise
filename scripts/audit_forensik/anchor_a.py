"""Apakah cacat UF memengaruhi jangkar Hungarian A (953, max_size=4)?

Anchor A adalah satu-satunya profil test-locked yang memakai `max_size=4`,
yaitu wilayah tempat cacat `sweep_remote_pipeline.UF` berpotensi aktif.
Jalur terkuncinya melewati `evaluate_remote_class_head.evaluate_payload`, yang
memanggil `sweep.clusters(...)`.

Profil Anchor A (GSP_LINKER.md): proposal_min=0,125 · pair_mode=adjacent ·
link=0,15 · singleton=0,15 · max_size=4 · rank=score.
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import sweep_remote_pipeline as swp
import eval_remote_pipeline_postprocess as base

PROPOSAL, PAIR, LINK, SINGLE, MAXSZ = 0.125, "adjacent", 0.15, 0.15, 4


class UF_buggy:
    def __init__(self, sides, max_size):
        n = len(sides)
        self.parent = list(range(n)); self.size = [1] * n
        self.sides = [{i} for i in range(n)]      # cacat: indeks proposal
        self.max_size = max_size
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if (a == b or self.sides[a] & self.sides[b]
                or self.size[a] + self.size[b] > self.max_size):
            return False
        self.parent[b] = a; self.size[a] += self.size[b]
        self.sides[a] |= self.sides[b]; return True


def group(dets, edges, klass):
    uf = klass([d["side"] for d in dets], MAXSZ)
    for score, i, j in edges:
        if score < LINK:
            break
        uf.union(i, j)
    g = defaultdict(list)
    for i, d in enumerate(dets):
        g[uf.find(i)].append(i)
    return [sorted(v) for v in g.values()]


cfg = base.CONFIGS["SawitMVC-YOLO"]
prior = base.build_rotation_prior(base.load_records(cfg, "train"))
vote = swp.load_vote(Path("results/remote_eval_2026-08-27/fused_combined1716/"
                          "SawitMVC_YOLO__wbf_softvote.npz"))

for split in ["test", "val"]:
    recs = {t: r for t, r in base.load_records(cfg, split).items() if r["n_sides"] == 4}
    tot = {"buggy": [0, 0], "fixed": [0, 0]}
    beda_pohon = 0
    for t, rec in recs.items():
        dets = swp.make_detections(rec, vote, PROPOSAL)
        if not dets:
            continue
        edges = swp.build_edges(dets, rec["n_sides"], prior, PAIR)
        gb = group(dets, edges, UF_buggy)
        gf = group(dets, edges, swp.UF)
        for name, gs in [("buggy", gb), ("fixed", gf)]:
            tot[name][0] += len(gs)
            tot[name][1] += sum(1 for g in gs
                                if len({dets[i]["side"] for i in g}) != len(g))
        if sorted(gb) != sorted(gf):
            beda_pohon += 1
    b, f = tot["buggy"], tot["fixed"]
    print(f"\n[{split}] {len(recs)} pohon empat sisi · profil Anchor A "
          f"(proposal {PROPOSAL}, {PAIR}, link {LINK}, singleton {SINGLE}, max_size {MAXSZ})")
    print(f"  versi cacat : {b[0]:5d} klaster · {b[1]} melanggar kendala sisi")
    print(f"  versi benar : {f[0]:5d} klaster · {f[1]} melanggar kendala sisi")
    print(f"  pohon dengan partisi berbeda: {beda_pohon} dari {len(recs)}")
    print("  -> " + ("IDENTIK: angka terkunci Anchor A tidak terpengaruh"
                     if beda_pohon == 0 else
                     f"BERBEDA: {beda_pohon} pohon berubah, angka terkunci perlu dihitung ulang"))
