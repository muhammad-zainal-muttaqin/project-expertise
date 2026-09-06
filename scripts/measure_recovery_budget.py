"""Ukur peluang pemulihan pipeline Panen pada VALIDATION saja.

GT hanya digunakan untuk diagnosis batas bersyarat, bukan inferensi kandidat.
Tidak ada pelatihan, pemilihan ambang, atau pembacaan hasil TEST.
"""
from __future__ import annotations

import ast
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/workspace/SawitMVC-YOLO')
CACHE = Path('/workspace/results_panen')


def load_state():
    with (DATA / 'split_manifest.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.reader(handle))
    split = {r[0]: r[-1] for r in rows[1:] if r}
    records, gt = {}, {}
    for tree, sp in split.items():
        if sp != 'val':
            continue
        rec = json.loads((DATA / 'json' / f'{tree}.json').read_text())
        if len(rec['images']) != 4:
            continue
        records[tree] = rec
        ann = {(side, a['box_index']): a['bbox_yolo'] for side, view in rec['images'].items()
               for a in view.get('annotations', [])}
        gt[tree] = []
        for b in rec.get('bunches') or []:
            c = {'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3}.get(b['class'], -1)
            apps = [(int(a['side'].split('_')[1]), ann[a['side'], a['box_index']])
                    for a in b['appearances'] if (a['side'], a['box_index']) in ann]
            if c >= 0 and apps:
                gt[tree].append({'c': c, 'app': apps})
    with (CACHE / 'dets.pkl').open('rb') as handle:
        # Arsip berisi seluruh split; hanya anggota VAL dipakai di bawah.
        bank = pickle.load(handle)['val']
    with (CACHE / 'edge_model_v2.pkl').open('rb') as handle:
        model = pickle.load(handle)
    profile = json.loads((CACHE / 'panen_final.json').read_text())['profil']
    # Impor fungsi murni saja, sehingga konstruksi model GPU tidak dijalankan.
    source = ROOT / 'scripts/audit_forensik/panen_pipeline.py'
    names = {'iou1', 'yolo_to_xyxy', 'pair_feats', 'gt_bunch_of',
             'build_pairs', 'UF', 'cluster_tree'}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = {'np': np, 'defaultdict': defaultdict, 'GT': gt,
          'NSIDE': 4, 'linear_sum_assignment': linear_sum_assignment}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return records, gt, bank, model, profile, ns


def overlap(dets, truth, ns):
    result = np.zeros((len(dets), len(truth)))
    for i, d in enumerate(dets):
        for j, b in enumerate(truth):
            result[i, j] = max((ns['iou1'](d['box'], ns['yolo_to_xyxy'](box))
                               for side, box in b['app'] if side == d['side']), default=0.)
    return result


def coverage(matrix, threshold=.5):
    return set(np.flatnonzero((matrix >= threshold).any(0)).tolist())


def class_of(score, p):
    if score < p['t_coarse']:
        return 0 if score < p['t_b1b2'] else 1
    return 2 if score < p['t_b3b4'] else 3


def main():
    records, gt, bank, model, profile, ns = load_state()
    total = Counter(); per_class = defaultdict(Counter); per_tree = []
    with threadpool_limits(limits=2):
        for tree in records:
            raw, truth = bank.get(tree, []), gt[tree]
            dets = [d for d in raw if d['conf'] >= profile['det_conf']]
            raw_iou, selected_iou = overlap(raw, truth, ns), overlap(dets, truth, ns)
            cov_raw, cov_select = coverage(raw_iou), coverage(selected_iou)
            X, _, keys = ns['build_pairs']({tree: dets}, [tree], labelled=False)
            prob = model.predict_proba(X)[:, 1] if len(X) else []
            edges = {(min(i, j), max(i, j)): float(p) for (_, i, j), p in zip(keys, prob)}
            groups = ns['cluster_tree'](dets, edges, profile['link_thr'], profile['max_size'])
            kept = [g for g in groups if len(g) > 1 or dets[g[0]]['conf'] >= profile['single_thr']]
            kept_members = [i for g in kept for i in g]
            cov_kept = coverage(selected_iou[kept_members])
            group_conf = [np.mean([dets[i]['conf'] for i in g]) for g in kept]
            group_iou = np.asarray([selected_iou[g].max(0) for g in kept]).reshape(len(kept), len(truth)) if truth else np.zeros((len(kept), 0))
            matched = set()
            correct = 0
            for gidx in np.argsort(-np.asarray(group_conf), kind='stable'):
                avail = [j for j in range(len(truth)) if j not in matched]
                j = max(avail, key=lambda j: group_iou[gidx, j]) if avail else -1
                if j >= 0 and group_iou[gidx, j] > .5:
                    matched.add(j)
                    g = kept[gidx]
                    s = np.average([dets[i]['score'] for i in g], weights=[dets[i]['conf'] for i in g])
                    correct += class_of(s, profile) == truth[j]['c']
            stats = Counter(gt=len(truth), raw_proposals=len(raw), kept_clusters=len(kept),
                            raw_coverage=len(cov_raw), threshold_coverage=len(cov_select),
                            retained_member_coverage=len(cov_kept), matched=len(matched),
                            correct_class_matched=correct)
            stats.update(missing_raw=len(truth) - len(cov_raw),
                         lost_at_confidence=len(cov_raw - cov_select),
                         lost_at_singleton=len(cov_select - cov_kept),
                         lost_at_grouping_or_assignment=len(cov_kept - matched))
            assert stats['gt'] == sum(stats[k] for k in ['missing_raw', 'lost_at_confidence',
                        'lost_at_singleton', 'lost_at_grouping_or_assignment', 'matched'])
            for j, b in enumerate(truth):
                pc = per_class[f"B{b['c']+1}"]; pc['gt'] += 1
                for key, cov in [('raw_coverage', cov_raw), ('threshold_coverage', cov_select),
                                 ('retained_member_coverage', cov_kept), ('matched', matched)]:
                    pc[key] += j in cov
                if j not in cov_raw:
                    continue
                ids = np.flatnonzero(raw_iou[:, j] >= .5)
                labels = [class_of(raw[i]['score'], profile) for i in ids]
                mean_score = np.average([raw[i]['score'] for i in ids], weights=[raw[i]['conf'] for i in ids])
                stats['oracle_identity_class_mean_correct'] += class_of(mean_score, profile) == b['c']
                stats['oracle_identity_any_view_class_correct'] += b['c'] in labels
                stats['oracle_identity_n_views'] += len(set(raw[i]['side'] for i in ids))
            total.update(stats)
            per_tree.append({'tree': tree, **stats})
    summary = {**total, 'trees': len(records),
               'trees_with_complete_raw_coverage': sum(r['raw_coverage'] == r['gt'] for r in per_tree),
               'trees_with_at_most_one_raw_missing': sum(r['gt'] - r['raw_coverage'] <= 1 for r in per_tree),
               'raw_coverage_fraction': total['raw_coverage'] / total['gt'],
               'pipeline_recall': total['matched'] / total['gt'],
               'oracle_identity_class_mean_accuracy': total['oracle_identity_class_mean_correct'] / total['raw_coverage'],
               'oracle_identity_best_view_accuracy': total['oracle_identity_any_view_class_correct'] / total['raw_coverage']}
    result = {'protocol': 'VAL only, frozen Panen final; GT-assisted diagnostics are NOT deployable performance',
              'coverage_definition': 'any same-side proposal IoU>=0.5; optimistic when one box overlaps multiple identities',
              'profile': profile, 'summary': summary, 'per_class': dict(per_class), 'per_tree': per_tree}
    path = ROOT / 'results/audit_2026-09-06/recovery_budget_val.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_tree'}, indent=2))


if __name__ == '__main__':
    main()
