"""Audit artefak AF dan Panen dengan konfigurasi tetap, tanpa inferensi GPU.

Evaluasi ini bersifat diagnostik. Tidak memilih parameter dari partisi test,
tidak mengubah hasil historis, dan tidak melatih model baru.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from build_combined_rgb_dataset import collect_source

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/workspace/SawitMVC-YOLO')
PANEN = Path('/workspace/results_panen')
SOURCE = ROOT / 'scripts/audit_forensik/panen_pipeline.py'


def metrics(pred, true):
    error = np.abs(np.asarray(pred) - np.asarray(true))
    return {'n': int(error.size), 'mae': float(error.mean()),
            'exact': float((error == 0).mean()), 'within1': float((error <= 1).mean())}


def add_end_to_end_cm(parsed):
    """Tambahkan pencatatan CM pada evaluator tanpa mengubah keputusannya.

    Instrumentasi AST menghindari import yang memuat model GPU. Pencocokan,
    klasifikasi, dan keluaran historis tetap berasal dari fungsi asli.
    """
    evaluate = next(n for n in parsed if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    evaluate.body.insert(0, ast.parse('audit_cm = np.zeros((5, 5), int)').body[0])
    tree_loop = next(n for n in evaluate.body if isinstance(n, ast.For)
                     and isinstance(n.target, ast.Name) and n.target.id == 't')
    position = next(i for i, n in enumerate(tree_loop.body) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Tuple)
                    and [x.id for x in n.target.elts] == ['c', 'gi'])
    extra = ast.parse('''
for audit_i, audit_cluster in enumerate(clus):
    audit_true = gt[match[audit_i]]['c'] if audit_i in match else 4
    audit_cm[fine(audit_cluster['score']), audit_true] += 1
for audit_i, audit_bunch in enumerate(gt):
    if audit_i not in used:
        audit_cm[4, audit_bunch['c']] += 1
''').body
    tree_loop.body[position:position] = extra
    returned = next(n for n in evaluate.body if isinstance(n, ast.Return))
    result_dict = returned.value.elts[0]
    result_dict.keywords.append(ast.keyword(arg='audit_cm_e2e', value=ast.parse('audit_cm.tolist()', mode='eval').body))
    result_dict.keywords.append(ast.keyword(arg='audit_macro_f1_e2e', value=ast.parse(
        'float(np.mean([2 * audit_cm[k,k] / max(audit_cm[k,:].sum() + audit_cm[:,k].sum(), 1) for k in range(4)]))',
        mode='eval').body))
    return ast.fix_missing_locations(ast.Module(body=parsed, type_ignores=[]))


def main():
    with (DATA / 'split_manifest.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.reader(handle))
    split = {r[0]: r[-1] for r in rows[1:] if r}
    records = {p.stem: json.loads(p.read_text(encoding='utf-8-sig'))
               for p in sorted((DATA / 'json').glob('*.json'))}
    gt = {}
    for tree, rec in records.items():
        ann = {(side, a['box_index']): a['bbox_yolo']
               for side, view in rec.get('images', {}).items()
               for a in view.get('annotations', [])}
        bunches = []
        for b in rec.get('bunches') or []:
            c = {'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3}.get(b.get('class'), -1)
            apps = [(int(x['side'].split('_')[1]), ann[x['side'], x['box_index']])
                    for x in b.get('appearances', []) if (x['side'], x['box_index']) in ann]
            if c >= 0 and apps:
                bunches.append({'c': c, 'app': apps})
        gt[tree] = bunches

    train = sorted(t for t in records if split[t] == 'train')
    test = sorted(t for t in records if split[t] == 'test')
    constants = {}
    for target, classes in [('B1', {0}), ('B1_B2', {0, 1}), ('total', {0, 1, 2, 3})]:
        ytr = np.array([sum(b['c'] in classes for b in gt[t]) for t in train])
        yte = np.array([sum(b['c'] in classes for b in gt[t]) for t in test])
        # Select the constant using TRAIN only; TEST is descriptive evaluation.
        candidates = np.arange(int(ytr.max()) + 1)
        selected = int(max(candidates, key=lambda c: (np.mean(np.abs(ytr - c) <= 1),
                                                       -np.mean(np.abs(ytr - c)), -c)))
        constants[target] = {'train_selected_constant': selected,
                             'test_distribution': dict(sorted(Counter(yte.tolist()).items())),
                             'test_mean': float(yte.mean()),
                             'constant_test': metrics(np.full(len(yte), selected), yte),
                             'always_zero_test': metrics(np.zeros(len(yte)), yte)}

    _, source_953 = collect_source(DATA, '953', {
        'train': ('images/train', 'labels/train'),
        'valid': ('images/val', 'labels/val'), 'test': ('images/test', 'labels/test')})
    _, source_depth = collect_source(Path('/workspace/SawitMVC-Depth-YOLO'), 'depth', {
        sp: (f'{sp}/images', f'{sp}/labels') for sp in ['train', 'valid', 'test']})
    shared = sorted(set(source_953) & set(source_depth))
    transition = Counter(f'{source_depth[t]}->{source_953[t]}' for t in shared)
    test_to_train = [t for t in shared if source_depth[t] == 'test' and source_953[t] == 'train']

    # The cache was generated by the user in this workspace. Do not load
    # arbitrary downloaded pickle files through this diagnostic.
    with (PANEN / 'dets.pkl').open('rb') as handle:
        bank = pickle.load(handle)
    with (PANEN / 'edge_model.pkl').open('rb') as handle:
        edge = pickle.load(handle)
    result_path = PANEN / 'panen_results.json'
    historical = json.loads(result_path.read_text())
    source_text = SOURCE.read_text()
    wanted = {'iou1', 'yolo_to_xyxy', 'pair_feats', 'gt_bunch_of', 'build_pairs',
              'UF', 'cluster_tree', 'evaluate'}
    parsed = ast.parse(source_text)
    nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in wanted]
    assert len(nodes) == len(wanted)
    namespace = {'np': np, 'defaultdict': defaultdict, 'GT': gt, 'NSIDE': 4,
                 'linear_sum_assignment': linear_sum_assignment}
    exec(compile(add_end_to_end_cm(nodes), str(SOURCE), 'exec'), namespace)
    legacy_eligible = {t for t, b in gt.items() if b and max(s for x in b for s, _ in x['app']) <= 4}
    canonical_eligible = {t for t, r in records.items() if len(r.get('images', {})) == 4}
    original_test = sorted(t for t in bank['test'] if t in legacy_eligible)
    canonical_test = sorted(t for t in test if t in canonical_eligible)
    with threadpool_limits(limits=2):
        original, _ = namespace['evaluate'](bank['test'], original_test, edge,
                                           **historical['profil'])
        canonical, per_tree = namespace['evaluate'](bank['test'], canonical_test, edge,
                                                    **historical['profil'])
    for key in ['physical_f1', 'class4_acc', 'class4_macro_f1']:
        assert np.isclose(original[key], historical['test'][key]), (key, original[key], historical['test'][key])
    excluded = sorted(set(canonical_test) - set(original_test))
    exclusion_reasons = [{'tree': t, 'n_images': len(records[t]['images']),
                          'gt_bunches': len(gt[t]), 'present_in_detection_cache': t in bank['test'],
                          'legacy_eligible': t in legacy_eligible} for t in excluded]

    final_path = PANEN / 'panen_final.json'
    final = json.loads(final_path.read_text())
    with (PANEN / 'edge_model_v2.pkl').open('rb') as handle:
        edge_v2 = pickle.load(handle)
    with threadpool_limits(limits=2):
        final_original, _ = namespace['evaluate'](bank['test'], original_test, edge_v2, **final['profil'])
        final_canonical, _ = namespace['evaluate'](bank['test'], canonical_test, edge_v2, **final['profil'])
    for key in ['physical_f1', 'class4_acc', 'class4_macro_f1']:
        assert np.isclose(final_original[key], final['test'][key])

    # Reconstruct full CORN posterior from cached cumulative probabilities.
    # This assesses information loss only, with no threshold search on TEST.
    corn = {}
    for sp in ['val', 'test']:
        with np.load(f'/workspace/crops953/corn_{sp}.npz') as archive:
            cum, y, score = archive['cum'], archive['y'], archive['score']
            prob = np.c_[1 - cum[:, 0], cum[:, 0] - cum[:, 1],
                         cum[:, 1] - cum[:, 2], cum[:, 2]]
            coarse_p = cum[:, 1]  # P(B3 or B4), exact for this taxonomy.
            corn[sp] = {'n': len(y), 'posterior_min': float(prob.min()),
                        'posterior_argmax_accuracy': float((prob.argmax(1) == y).mean()),
                        'rounded_expectation_accuracy': float((np.rint(score) == y).mean()),
                        'coarse_posterior_accuracy_at_half': float(((coarse_p >= .5) == (y > 1)).mean()),
                        'coarse_expectation_accuracy_at_1_5': float(((score >= 1.5) == (y > 1)).mean()),
                        'coarse_rule_disagreement': int(((coarse_p >= .5) != (score >= 1.5)).sum())}
    result = {'schema_version': 1, 'mode': 'read-only diagnostic, frozen configuration, no GPU inference',
              'constant_baselines': constants,
              'split_provenance': {'shared_tree_ids': len(shared),
                  'depth_source_to_combined_split_for_shared_trees': dict(sorted(transition.items())),
                  'depth_test_trees_moved_to_combined_train_by_builder': test_to_train,
                  'scope': 'builder policy applied to current source manifests; historical checkpoint exposure requires its own manifest'},
              'panen': {
                  'original_test_n': len(original_test), 'canonical_test_n': len(canonical_test),
                  'excluded_trees': exclusion_reasons,
                  'noncanonical_in_original': sorted(set(original_test) - set(canonical_test)),
                  'reproduced_original': original, 'canonical_frozen_profile': canonical,
              'canonical_per_tree': per_tree},
              'panen_final': {'reproduced_original': final_original,
                              'canonical_frozen_profile': final_canonical,
                              'counting_source': final['counting'],
                              'counting_reproduced': False},
              'corn_cached_diagnostic': corn,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [SOURCE, result_path, final_path, Path(__file__), DATA / 'split_manifest.csv']}}
    output = ROOT / 'results/audit_2026-09-06/latest_artifacts_review.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    compact = {**result, 'panen': {k: v for k, v in result['panen'].items() if k != 'canonical_per_tree'}}
    compact.pop('source_sha256')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
