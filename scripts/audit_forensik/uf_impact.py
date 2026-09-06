"""Seberapa besar dampak cacat UF pada jalur sweep yang sebenarnya?

AF-E-010 mengukur pelanggaran pada daftar tepi geometri sederhana tanpa
penugasan Hungarian. Jalur sweep yang asli memakai `linear_sum_assignment` per
pasangan sisi, yang sudah memaksa satu-lawan-satu di dalam tiap pasangan sisi.
Skrip ini mengukur pelanggaran pada daftar tepi yang SEBENARNYA dipakai sweep.
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import sweep_remote_pipeline as swp
import eval_remote_pipeline_postprocess as base


class UF_buggy:
    def __init__(self, sides, max_size):
        n = len(sides)
        self.parent = list(range(n)); self.size = [1] * n
        self.sides = [{i} for i in range(n)]          # cacat: indeks proposal
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


def group(dets, edges, link_thr, max_size, klass):
    uf = klass([d["side"] for d in dets], max_size) if klass is swp.UF \
        else klass([d["side"] for d in dets], max_size)
    for score, i, j in edges:
        if score < link_thr:
            break
        uf.union(i, j)
    g = defaultdict(list)
    for i, d in enumerate(dets):
        g[uf.find(i)].append(d["side"])
    return list(g.values())


cfg = base.CONFIGS["SawitMVC-YOLO"]
recs = {t: r for t, r in base.load_records(cfg, "test").items() if r["n_sides"] == 4}
prior = base.build_rotation_prior(base.load_records(cfg, "train"))
vote = swp.load_vote(Path("results/remote_eval_2026-08-27/fused_combined1716/"
                          "SawitMVC_YOLO__wbf_softvote.npz"))
print(f"pohon empat sisi: {len(recs)}")

for pair_mode in ["all", "adjacent"]:
    for proposal_min, link_thr, max_size in [(0.10, 0.20, 3), (0.16, 0.05, 2),
                                             (0.125, 0.30, 3), (0.10, 0.20, 4)]:
        tot = {"buggy": [0, 0], "fixed": [0, 0]}
        for t, rec in recs.items():
            dets = swp.make_detections(rec, vote, proposal_min)
            if not dets:
                continue
            edges = swp.build_edges(dets, rec["n_sides"], prior, pair_mode)
            for name, K in [("buggy", UF_buggy), ("fixed", swp.UF)]:
                gs = group(dets, edges, link_thr, max_size, K)
                tot[name][0] += len(gs)
                tot[name][1] += sum(1 for g in gs if len(g) != len(set(g)))
        b, f = tot["buggy"], tot["fixed"]
        print(f"  pair_mode={pair_mode:8} proposal={proposal_min} link={link_thr} "
              f"max_size={max_size}")
        print(f"     cacat : {b[0]:5d} klaster · {b[1]:4d} melanggar sisi "
              f"({100*b[1]/max(b[0],1):5.2f}%)")
        print(f"     benar : {f[0]:5d} klaster · {f[1]:4d} melanggar sisi "
              f"({100*f[1]/max(f[0],1):5.2f}%)  Δklaster={f[0]-b[0]:+d}")
