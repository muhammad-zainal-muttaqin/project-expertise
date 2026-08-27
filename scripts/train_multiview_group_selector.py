"""Train a visual multi-view selector on the actual linked detector groups.

The ordinary proposal head predicts one class per crop and averages the
probabilities.  This experiment gives a small set transformer the complete
linked group: one pretrained RGB embedding per view, all available crop-head
probabilities, detector probabilities, score and geometry.  It is deliberately
post-cluster; boxes, linking and count reconciliation remain fixed.

Training labels are obtained only from TRAIN groups matched to ground truth.
Validation is used for checkpoint/profile selection.  The TEST path is
optional and is only evaluated after a validation candidate is selected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_remote_pipeline_postprocess as base  # noqa: E402
import evaluate_remote_class_head as post  # noqa: E402
import evaluate_remote_count_reconciled as count  # noqa: E402
import sweep_remote_pipeline as sweep  # noqa: E402
import train_proposal_crop_head as crop  # noqa: E402


K = len(base.NAMES)
ROOT = Path("/workspace/model_artifacts/project-expertise")
EVAL_ROOT = ROOT / "eval_2026-08-27"
MAX_GROUP = 3


def load_vote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], np.float32)
                for key in archive.files}


def vote_path(name: str, split: str, fused_root: Path = EVAL_ROOT) -> Path:
    safe = "SawitMVC_YOLO"
    if name == "baseline":
        return fused_root / f"fused_combined1716_{split}" / f"{safe}__wbf_softvote.npz"
    return ROOT / name / f"fused_{split}__wbf_softvote.npz"


def norm(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, np.float32), 1e-8)
    return p / max(float(p.sum()), 1e-8)


def norm_rows(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, np.float32), 1e-8)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)


def build_split(cfg: dict, split: str, vote: dict[str, np.ndarray], prior: dict,
                proposal_min: float, link_threshold: float,
                singleton_min: float, max_size: int, pair_mode: str,
                target_counts: dict[str, int] | None = None):
    records = count.four_side(base.load_records(cfg, split))
    payload = []
    for rec in records.values():
        dets = post.make_detections(rec, vote, vote, proposal_min)
        edges = sweep.build_edges(dets, rec["n_sides"], prior, pair_mode)
        raw = sweep.clusters(dets, edges, link_threshold, singleton_min, max_size)
        selected = None
        if target_counts is not None:
            selected = count.selected_clusters(
                dets, edges, link_threshold, singleton_min, max_size,
                target_counts[rec["tree_id"]], "max_member")
        payload.append({"rec": rec, "dets": dets, "edges": edges,
                        "raw": raw, "groups": selected})
    return records, payload


def fit_count_model(cfg: dict, vote: dict[str, np.ndarray], proposal_min: float):
    records = count.four_side(base.load_records(cfg, "train"))
    x = np.stack([count.feature_vector(r, vote, proposal_min)
                  for r in records.values()])
    y = np.asarray([count.target_count(r) for r in records.values()], float)
    alpha, cv = count.choose_alpha(x, y)
    return count.fit_ridge(x, y, alpha), cv


def predicted_targets(cfg: dict, vote: dict[str, np.ndarray], split: str,
                      proposal_min: float, model: dict) -> dict[str, int]:
    records = count.four_side(base.load_records(cfg, split))
    x = np.stack([count.feature_vector(r, vote, proposal_min)
                  for r in records.values()])
    return {tree: int(value) for tree, value in
            zip(records, count.predict_count(model, x))}


def key_of(member: dict) -> tuple[str, int]:
    return str(member["stem"]), int(member["row_index"])


def group_signature(group: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(key_of(x) for x in group["members"]))


def extract_visual_maps(cfg: dict, split: str, vote: dict[str, np.ndarray],
                        groups: list[dict], checkpoint: Path, img: int,
                        batch: int, workers: int):
    samples, _ = crop.build_samples(cfg, "953", split, vote, True)
    sample_index = {(s.stem, int(s.row_index)): i
                    for i, s in enumerate(samples)}
    keys = sorted({key_of(member) for group in groups
                   for member in group["members"]})
    missing = [key for key in keys if key not in sample_index]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} proposal keys missing from crop samples")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ck_args = ckpt["args"]
    visual = crop.ProposalModel(
        ck_args["backbone"], ck_args.get("channels", 3),
        ck_args.get("freeze_backbone", False)).cuda().eval()
    visual.load_state_dict(ckpt["model"])
    selected = [sample_index[key] for key in keys]
    ds = Subset(crop.ProposalDS(samples, img, False, "rgb"), selected)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                     num_workers=min(workers, 8), pin_memory=True,
                     persistent_workers=min(workers, 8) > 0)
    embeddings, head_probs = [], []
    with torch.inference_mode():
        for x, _y, _idx in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = visual.bb(x.cuda(non_blocking=True))
                logits = visual.fc(emb)
            embeddings.append(emb.float().cpu().numpy())
            head_probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    if not embeddings:
        return {}, {}, len(visual.bb.num_features) if hasattr(visual.bb, "num_features") else 0
    emb = np.concatenate(embeddings, 0).astype(np.float32)
    hp = np.concatenate(head_probs, 0).astype(np.float32)
    # L2 normalization makes the group model insensitive to backbone norm.
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
    return ({key: emb[i] for i, key in enumerate(keys)},
            {key: norm(hp[i]) for i, key in enumerate(keys)}, emb.shape[1])


def proposal_probability(vote: dict[str, np.ndarray], member: dict) -> np.ndarray:
    rows = vote.get(str(member["stem"]))
    if rows is None or int(member["row_index"]) >= len(rows):
        return norm(member["p"])
    return norm(rows[int(member["row_index"]), 5:5 + K])


def make_group_arrays(payload: list[dict], vote_bank: dict[str, dict[str, np.ndarray]],
                      emb_map: dict[tuple[str, int], np.ndarray],
                      visual_map: dict[tuple[str, int], np.ndarray],
                      emb_dim: int, max_group: int = MAX_GROUP):
    """Return padded token arrays, group auxiliary features and labels."""
    head_names = [x for x in vote_bank if x != "baseline"]
    # token: embedding, baseline p, each head p, score/geometry/side metadata.
    token_dim = emb_dim + K * (1 + len(head_names)) + 8
    tokens, masks, aux, labels, trees, signatures, matched = [], [], [], [], [], [], []
    for item in payload:
        rec, groups = item["rec"], item["groups"]
        if groups is None:
            continue
        # Match the complete tree globally once.  Matching each group against
        # a one-element list would allow several predicted groups to claim the
        # same GT bunch and would silently corrupt the supervised labels.
        global_matches = count.tree_matches(rec, groups)
        match_by_group = {int(i): int(j) for i, j in global_matches}
        for gi, group in enumerate(groups):
            members = sorted(group["members"], key=lambda x: (x["side"], key_of(x)))
            members = members[:max_group]
            tok = np.zeros((max_group, token_dim), np.float32)
            mask = np.ones(max_group, bool)
            per_head = []
            for mi, member in enumerate(members):
                key = key_of(member)
                base_p = norm(member["p"])
                parts = [emb_map[key], base_p]
                for name in head_names:
                    parts.append(proposal_probability(vote_bank[name], member))
                # score, normalized geometry, side and group-relative side flag.
                parts.append(np.asarray([
                    float(member["score"]), float(member["cx"]),
                    float(member["cy"]), float(member["w"]),
                    float(member["h"]), float(member["side"] / 3.),
                    float(len(members) / max_group),
                    float(mi / max(len(members) - 1, 1)),
                ], np.float32))
                tok[mi] = np.concatenate(parts)
                mask[mi] = False
                per_head.append([base_p] + [proposal_probability(vote_bank[n], member)
                                             for n in head_names])
            # Group-level low-dimensional context is explicitly supplied to
            # prevent the network from having to infer all summary statistics.
            valid = tok[~mask]
            ph = np.asarray(per_head, np.float32)
            # The score is the last eight token values, first one of that block.
            scores = valid[:, -(8)]
            aux_row = np.concatenate([
                np.average(ph, axis=0, weights=scores).ravel(),
                ph.mean(0).ravel(), ph.std(0).ravel(),
                np.asarray([len(members), scores.mean(), scores.max(), scores.min(),
                            scores.std(), float(group["score"])], np.float32),
            ]).astype(np.float32)
            tokens.append(tok); masks.append(mask); aux.append(aux_row)
            trees.append(rec["tree_id"]); signatures.append(group_signature(group))
            if gi in match_by_group:
                labels.append(int(rec["bunches"][match_by_group[gi]]["cls"]))
                matched.append(len(labels) - 1)
            else:
                labels.append(-1)
    if not tokens:
        return (np.zeros((0, max_group, token_dim), np.float32),
                np.zeros((0, max_group), bool), np.zeros((0, 1), np.float32),
                np.zeros(0, np.int64), np.asarray([], str), [], np.zeros(0, bool))
    return (np.asarray(tokens, np.float32), np.asarray(masks, bool),
            np.asarray(aux, np.float32), np.asarray(labels, np.int64),
            np.asarray(trees, str), signatures,
            np.asarray([x >= 0 for x in labels], bool))


class GroupDataset(Dataset):
    def __init__(self, tokens, masks, aux, labels, indices):
        self.tokens = torch.from_numpy(tokens[indices])
        self.masks = torch.from_numpy(masks[indices])
        self.aux = torch.from_numpy(aux[indices])
        self.labels = torch.from_numpy(labels[indices]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.tokens[index], self.masks[index], self.aux[index], self.labels[index]


class GroupSelector(nn.Module):
    def __init__(self, token_dim: int, aux_dim: int, d_model: int = 256,
                 layers: int = 2, heads: int = 8, dropout: float = .15):
        super().__init__()
        self.token = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, d_model),
                                   nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers,
                                              enable_nested_tensor=False)
        self.attn = nn.Linear(d_model, 1)
        self.aux = nn.Sequential(nn.LayerNorm(aux_dim), nn.Linear(aux_dim, d_model // 2),
                                 nn.GELU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(nn.LayerNorm(d_model + d_model // 2),
                                nn.Linear(d_model + d_model // 2, d_model),
                                nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_model, K))

    def forward_with_aux(self, tokens, mask, aux):
        x = self.token(tokens)
        x = self.encoder(x, src_key_padding_mask=mask)
        a = self.attn(x).squeeze(-1).masked_fill(mask, -1e4)
        a = torch.softmax(a, 1)
        pooled = (x * a.unsqueeze(-1)).sum(1)
        return self.fc(torch.cat([pooled, self.aux(aux)], 1))


def cls_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    f1 = []
    for c in range(K):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(np.mean(f1)),
            "f1_per_class": dict(zip(base.NAMES, f1))}


@torch.inference_mode()
def infer_model(model: GroupSelector, tokens: np.ndarray, masks: np.ndarray,
                aux: np.ndarray, batch: int) -> np.ndarray:
    out = []
    for start in range(0, len(tokens), batch):
        sl = slice(start, start + batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = torch.softmax(model.forward_with_aux(
                torch.from_numpy(tokens[sl]).cuda(),
                torch.from_numpy(masks[sl]).cuda(),
                torch.from_numpy(aux[sl]).cuda()).float(), 1)
        out.append(q.cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, K), np.float32)


def e2e_from_group_probs(payload: list[dict], probs: np.ndarray,
                         signatures: list[tuple], prior_exp: float,
                         class_prior: np.ndarray) -> dict:
    by_sig = {sig: probs[i] for i, sig in enumerate(signatures)}
    cm = np.zeros((K + 1, K + 1), int)
    total_tp = total_gt = total_pred = abs_count = exact = pm1 = 0
    class_correct = matched = 0
    for item in payload:
        rec, groups = item["rec"], item["groups"]
        if groups is None:
            continue
        chosen = []
        for group in groups:
            p = by_sig[group_signature(group)]
            if prior_exp:
                p = norm(p * np.power(np.maximum(class_prior, 1e-8), prior_exp))
            g = dict(group); g["cls"] = int(np.argmax(p)); chosen.append(g)
        matches = count.tree_matches(rec, chosen)
        total_tp += len(matches); total_gt += len(rec["bunches"]); total_pred += len(chosen)
        delta = len(chosen) - len(rec["bunches"])
        abs_count += abs(delta); exact += int(delta == 0); pm1 += int(abs(delta) <= 1)
        matched_pred = {i for i, _ in matches}; matched_gt = {j for _, j in matches}
        for i, j in matches:
            pc, gc = chosen[i]["cls"], rec["bunches"][j]["cls"]
            if 0 <= pc < K and 0 <= gc < K:
                cm[pc, gc] += 1; class_correct += int(pc == gc); matched += 1
        for i, g in enumerate(chosen):
            if i not in matched_pred: cm[g["cls"], K] += 1
        for j, b in enumerate(rec["bunches"]):
            if j not in matched_gt and 0 <= b["cls"] < K: cm[K, b["cls"]] += 1
    precision = total_tp / max(total_pred, 1); recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f1s = []
    for c in range(K):
        tp = cm[c, c]; fp = int(cm[c].sum() - tp); fn = int(cm[:, c].sum() - tp)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    n = len(payload)
    return {"physical_f1": f1, "precision": precision, "recall": recall,
            "tp": total_tp, "pred_clusters": total_pred, "gt_bunches": total_gt,
            "count_mae": abs_count / max(n, 1),
            "count_exact": exact / max(n, 1), "count_pm1": pm1 / max(n, 1),
            "matched_class_accuracy": class_correct / max(matched, 1),
            "matched": matched, "macro_f1": float(np.mean(f1s)),
            "per_class_f1": dict(zip(base.NAMES, f1s)),
            "confusion": cm.tolist()}


def train_one(model, train_ds, val_tokens, val_masks, val_aux, val_y,
              epochs: int, batch: int, workers: int, lr: float,
              class_weight: np.ndarray, patience: int,
              sample_weights: np.ndarray | None = None):
    weights = (class_weight[train_ds.labels.numpy()] if sample_weights is None
               else np.asarray(sample_weights, float))
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                    len(weights), replacement=True)
    loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                        num_workers=min(workers, 8), pin_memory=True,
                        persistent_workers=min(workers, 8) > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 30.)
    cw = torch.tensor(class_weight, device="cuda", dtype=torch.float32)
    best_state, best_score, stale, history = None, -math.inf, 0, []
    for ep in range(1, epochs + 1):
        model.train(); total = 0.; nb = 0
        for tok, mask, aux, y in loader:
            tok = tok.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True)
            aux = aux.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_with_aux(tok, mask, aux)
                p = torch.softmax(logits.float(), 1)
                loss = F.cross_entropy(logits.float(), y, weight=cw)
                loss = loss + .10 * F.smooth_l1_loss(
                    p @ torch.arange(K, device="cuda", dtype=torch.float32), y.float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); total += float(loss.detach()); nb += 1
        sch.step()
        model.eval()
        val_p = infer_model(model, val_tokens, val_masks, val_aux, batch * 2)
        met = cls_metrics(val_y, val_p)
        row = {"epoch": ep, "loss": total / max(nb, 1), **met}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = met["accuracy"] + .15 * met["macro_f1"]
        if score > best_score + 1e-8:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return history, best_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--fused-root", type=Path, default=EVAL_ROOT)
    ap.add_argument("--proposal-min", type=float, default=.125)
    ap.add_argument("--link-threshold", type=float, default=.30)
    ap.add_argument("--singleton-min", type=float, default=.15)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--pair-mode", choices=("all", "adjacent"), default="adjacent")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-test", action="store_true",
                    help="jalankan test hanya setelah candidate validation tersedia")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA diperlukan")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = base.CONFIGS["SawitMVC-YOLO"]
    names = ["baseline", "proposal_head_953_rgb_natural",
             "proposal_head_953_rgb_ordinal224", "crop_tta_reg_none",
             "proposal_head_953_multimodal", "proposal_head_953"]
    votes = {}
    for name in names:
        for split in ("train", "val"):
            path = vote_path(name, split, args.fused_root)
            if not path.exists(): raise FileNotFoundError(path)
        votes[name] = {s: load_vote(vote_path(name, s, args.fused_root))
                       for s in ("train", "val")}
    prior = base.build_rotation_prior(base.load_records(cfg, "train"))
    count_model, count_cv = fit_count_model(cfg, votes["baseline"]["train"], args.proposal_min)
    targets = {s: predicted_targets(cfg, votes["baseline"][s], s,
                                    args.proposal_min, count_model)
               for s in ("train", "val")}
    payloads, records = {}, {}
    for split in ("train", "val"):
        records[split], payloads[split] = build_split(
            cfg, split, votes["baseline"][split], prior,
            args.proposal_min, args.link_threshold, args.singleton_min,
            args.max_size, args.pair_mode, targets[split])
        print(json.dumps({"split": split, "trees": len(records[split]),
                          "groups": sum(len(x["groups"] or []) for x in payloads[split])},
                         ensure_ascii=False), flush=True)
    group_lists = {s: [g for item in payloads[s] for g in (item["groups"] or [])]
                   for s in ("train", "val")}
    maps, emb_dim = {}, 0
    for split in ("train", "val"):
        em, hp, emb_dim = extract_visual_maps(
            cfg, split, votes["baseline"][split], group_lists[split],
            args.checkpoint, args.img, args.batch, args.workers)
        maps[split] = (em, hp)
        print(json.dumps({"split": split, "embedding_dim": emb_dim,
                          "visual_keys": len(em)}, ensure_ascii=False), flush=True)
    # The visual checkpoint itself is one candidate; the extra vote banks are
    # fixed, already trained heads and do not alter geometry.
    bank_train = {name: votes[name]["train"] for name in names}
    bank_val = {name: votes[name]["val"] for name in names}
    arrays = {}
    for split, bank in (("train", bank_train), ("val", bank_val)):
        arrays[split] = make_group_arrays(
            payloads[split], bank, maps[split][0], maps[split][1], emb_dim, args.max_size)
        print(json.dumps({"split": split, "groups": int(len(arrays[split][0])),
                          "matched": int(arrays[split][6].sum()),
                          "token_dim": int(arrays[split][0].shape[-1]),
                          "aux_dim": int(arrays[split][2].shape[-1])}, ensure_ascii=False), flush=True)
    tr_tok, tr_mask, tr_aux, tr_lab, tr_tree, tr_sig, tr_ok = arrays["train"]
    va_tok, va_mask, va_aux, va_lab, va_tree, va_sig, va_ok = arrays["val"]
    tr_idx = np.flatnonzero(tr_ok); va_idx = np.flatnonzero(va_ok)
    if len(tr_idx) < 100 or len(va_idx) < 50: raise RuntimeError("matched group data terlalu kecil")
    # Equalize trees before class balancing so a high-density tree cannot
    # dominate the selector.
    unique, counts = np.unique(tr_tree[tr_idx], return_counts=True)
    tree_weight = {t: 1. / max(int(c), 1) for t, c in zip(unique, counts)}
    freq = np.bincount(tr_lab[tr_idx], minlength=K).astype(float)
    class_weight = np.sqrt(freq.max() / np.maximum(freq, 1.))
    # WeightedRandomSampler receives per-example weights; class_weight is
    # multiplied by the inverse group count per tree.
    ds = GroupDataset(tr_tok, tr_mask, tr_aux, tr_lab, tr_idx)
    ds_sample_weights = class_weight[ds.labels.numpy()] * np.asarray(
        [tree_weight[t] for t in tr_tree[tr_idx]], float)
    # Store the weights for train_one without changing its simple interface.
    # The normalization is irrelevant to sampling.
    model = GroupSelector(tr_tok.shape[-1], tr_aux.shape[-1]).cuda()
    history, best_score = train_one(
        model, ds, va_tok[va_idx], va_mask[va_idx], va_aux[va_idx], va_lab[va_idx],
        args.epochs, args.batch, args.workers, args.lr, class_weight, args.patience,
        ds_sample_weights)
    # Evaluate all selected validation groups, not only matched groups, so the
    # E2E score includes the unchanged physical/counting path.
    model.eval()
    val_group_p = infer_model(model, va_tok, va_mask, va_aux, args.batch * 2)
    train_group_p = infer_model(model, tr_tok, tr_mask, tr_aux, args.batch * 2)
    class_prior = freq / max(freq.sum(), 1.)
    e2e_candidates = {}
    for exponent in (0., -.25):
        e2e_candidates[f"group_prior_{exponent:g}"] = e2e_from_group_probs(
            payloads["val"], val_group_p, va_sig, exponent, class_prior)
    # Existing fixed heads form useful diagnostic references and residual
    # blends.  Their group probabilities are constructed from exactly the same
    # selected groups, so comparison is apples-to-apples.
    def group_vote_probs(split: str, name: str):
        out = []
        bank = {n: votes[n][split] for n in names}
        for item in payloads[split]:
            for group in item["groups"] or []:
                vals, ws = [], []
                for member in group["members"]:
                    vals.append(proposal_probability(bank[name], member)); ws.append(max(float(member["score"]), 1e-6))
                out.append(norm(np.average(np.stack(vals), axis=0, weights=ws)))
        return np.asarray(out, np.float32)
    refs = {name: group_vote_probs("val", name) for name in names}
    for ref_name in ("baseline", "proposal_head_953_rgb_ordinal224", "crop_tta_reg_none", "proposal_head_953_multimodal"):
        for w in (0., .25, .5, .75, 1.):
            mix = norm_rows((1. - w) * refs[ref_name] + w * val_group_p)
            e2e_candidates[f"blend_{ref_name}_{w:g}"] = e2e_from_group_probs(
                payloads["val"], mix, va_sig, 0., class_prior)
    best = max(e2e_candidates.items(), key=lambda kv: (
        kv[1]["matched_class_accuracy"], kv[1]["macro_f1"]))
    output = {
        "dataset": "SawitMVC-YOLO", "split": "validation",
        "protocol": "train matched linked groups; visual embedding + multi-head probabilities; test after validation lock",
        "checkpoint": str(args.checkpoint), "proposal_min": args.proposal_min,
        "link_threshold": args.link_threshold, "singleton_min": args.singleton_min,
        "max_size": args.max_size, "pair_mode": args.pair_mode,
        "head_bank": names, "embedding_dim": emb_dim,
        "token_dim": int(tr_tok.shape[-1]), "aux_dim": int(tr_aux.shape[-1]),
        "train_matched": int(len(tr_idx)), "val_matched": int(len(va_idx)),
        "count_model_alpha": count_model["alpha"], "count_cv": count_cv,
        "class_prior": class_prior.tolist(), "training_history": history,
        "e2e_validation": e2e_candidates,
        "best_validation": {"name": best[0], "metrics": best[1]},
    }
    args.output_root.joinpath("results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args),
                "embedding_dim": emb_dim, "token_dim": int(tr_tok.shape[-1]),
                "aux_dim": int(tr_aux.shape[-1]), "best_validation": best},
               args.output_root / "group_selector.pt")
    np.savez_compressed(args.output_root / "group_features.npz",
                        val_tokens=va_tok, val_masks=va_mask, val_aux=va_aux,
                        val_labels=va_lab, val_tree=va_tree,
                        train_tokens=tr_tok, train_masks=tr_mask, train_aux=tr_aux,
                        train_labels=tr_lab, train_tree=tr_tree)
    print(json.dumps({"best_validation": best, "output": str(args.output_root / "results.json")},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
