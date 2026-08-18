"""Terapkan classifier detected-space setelah seluruh aturan dikunci di VAL.

Keluaran class-aware boleh memancarkan beberapa hipotesis kelas untuk satu
proposal karena AP COCO dihitung per kategori. Keluaran fisik selalu satu baris
per proposal dan dipakai oleh linker/counting. Seluruh konfigurasi scoring,
router per kelas, rescore background, dan threshold operasi dipilih di VAL.
Prediksi, crop, label, serta cache TEST baru dibuka setelah lock tercetak.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "pipeline-pertandan"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SUB / "scripts"))
import eval_dump_damimas as ED  # noqa: E402
import fusi_detektor_damimas as FD  # noqa: E402
import fusi_proposal_damimas as FP  # noqa: E402
import classifier_deteksi_damimas as CD  # noqa: E402


def muat_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    mu = np.asarray(ck["aux_mean"], np.float32)
    sd = np.asarray(ck["aux_std"], np.float32)
    model = CD.HibridaDeteksi(len(mu), args["backbone"]).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    return model, mu, sd, args, ck


def infer_cache(model, cache, mu, sd, ukuran, batch, workers):
    ds = CD.DataProposal(cache["img"], cache["meta"])
    ds.aux = ((ds.aux - mu) / sd).astype(np.float32)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    return CD.infer(model, loader, ukuran)


def bank_prob(pred_path: Path, meta_path: Path, prob: np.ndarray):
    zm = np.load(meta_path, allow_pickle=False)
    stems = zm["stem"].astype(str)
    rows = zm["row_idx"].astype(int)
    zm.close()
    zp = np.load(pred_path, allow_pickle=True)
    det = {s: np.asarray(zp[s], np.float32) for s in zp.files}
    zp.close()
    out = {s: np.full((len(D), 5), np.nan, np.float32) for s, D in det.items()}
    for i, (s, j) in enumerate(zip(stems, rows)):
        if s not in out or j < 0 or j >= len(out[s]):
            raise RuntimeError(f"Metadata proposal tidak cocok: {s}[{j}]")
        out[s][j] = prob[i]
    for s, p in out.items():
        if len(p) and not np.isfinite(p).all():
            raise RuntimeError(f"Probabilitas tidak lengkap: {s}")
    return det, out


def nama(cfg):
    if cfg["mode"] == "asli":
        return "proposal_asli"
    return (f"{cfg['blend']}_a{cfg['alpha']:g}_T{cfg['temperature']:g}_"
            f"k{cfg['topk']}_lp{cfg['loc_power']:g}_"
            f"ep{cfg['exist_power']:g}_g{cfg['gamma']:g}")


def gabung_prob(D: np.ndarray, P: np.ndarray, cfg: dict):
    base = CD.prob_empat(D)
    model = np.clip(P[:, :4], 1e-9, None)
    model /= np.maximum(model.sum(1, keepdims=True), 1e-9)
    model = model ** (1 / cfg["temperature"])
    model /= np.maximum(model.sum(1, keepdims=True), 1e-9)
    a = cfg["alpha"]
    if cfg["blend"] == "geom":
        q = np.clip(base, 1e-9, None) ** (1 - a) * model ** a
    else:
        q = (1 - a) * base + a * model
    return q / np.maximum(q.sum(1, keepdims=True), 1e-9)


def prediksi(det, prob, cfg):
    if cfg["mode"] == "asli":
        return {s: D[:, :6].copy() for s, D in det.items()}
    out = {}
    for s, D in det.items():
        if not len(D):
            out[s] = np.zeros((0, 6), np.float32)
            continue
        P = prob[s]
        q = gabung_prob(D, P, cfg)
        exist = np.clip(1 - P[:, 4], 1e-8, 1)
        obj = np.clip(D[:, 4], 1e-8, 1)
        rows = []
        for i in range(len(D)):
            for k in np.argsort(-q[i])[:cfg["topk"]]:
                score = (obj[i] ** cfg["loc_power"] *
                         exist[i] ** cfg["exist_power"] *
                         max(float(q[i, k]), 1e-8) ** cfg["gamma"])
                rows.append(np.r_[D[i, :4], score, float(k)])
        out[s] = np.asarray(rows, np.float32).reshape(-1, 6)
    return out


def route_per_kelas(det, prob, configs):
    cache = {}
    out = {s: [] for s in det}
    for k, cfg in enumerate(configs):
        key = nama(cfg)
        if key not in cache:
            cache[key] = prediksi(det, prob, cfg)
        for s, D in cache[key].items():
            if len(D):
                out[s].append(D[D[:, 5].astype(int) == k])
    return {s: (np.concatenate(v).astype(np.float32)
                if v else np.zeros((0, 6), np.float32)) for s, v in out.items()}


def obj_kelas(m, kelas):
    return (.65 * m["AP50_per_kelas"][kelas] +
            .35 * m["AP50_95_per_kelas"][kelas])


def configs_awal():
    out = [{"mode": "asli", "blend": "arit", "alpha": 0.,
            "temperature": 1., "topk": 1, "loc_power": 1.,
            "exist_power": 0., "gamma": 0.}]
    for blend in ("arit", "geom"):
        for alpha in (.50, .75, 1.):
            for topk in (1, 2):
                for lp in (.75, 1.):
                    for ep in (1., 1.5):
                        for gamma in (.75, 1.):
                            out.append({"mode": "detected", "blend": blend,
                                        "alpha": alpha, "temperature": 1.,
                                        "topk": topk, "loc_power": lp,
                                        "exist_power": ep, "gamma": gamma})
    return out


RUANG = {
    "blend": ("arit", "geom"),
    "alpha": (0., .10, .25, .40, .50, .60, .75, .90, 1.),
    "temperature": (.60, .75, .90, 1., 1.10, 1.25, 1.50),
    "topk": (1, 2, 3, 4),
    "loc_power": (.25, .50, .75, 1., 1.25, 1.50),
    "exist_power": (.25, .50, .75, 1., 1.25, 1.50, 2., 2.50),
    "gamma": (.25, .50, .75, 1., 1.25, 1.50, 2.),
}


def cari_classaware(coco, paths, det, prob):
    ranking, pred_cache = [], {}

    def nilai(cfg):
        key = nama(cfg)
        if key not in pred_cache:
            pred_cache[key] = prediksi(det, prob, cfg)
        m = FD.coco_detail(coco, paths, pred_cache[key])
        return {"config": dict(cfg), "metrik": m,
                "objective": FD.objektif(m)}

    for i, cfg in enumerate(configs_awal(), 1):
        row = nilai(cfg); ranking.append(row)
        if i % 24 == 0:
            print(f"class-aware {i}/{len(configs_awal())} "
                  f"best={max(x['objective'] for x in ranking):.6f}", flush=True)
    global_best = max(ranking, key=lambda x: x["objective"])
    cfg = dict(global_best["config"])
    if cfg["mode"] != "asli":
        for field, values in RUANG.items():
            pilih = global_best
            for value in values:
                q = dict(cfg); q[field] = value
                row = nilai(q); ranking.append(row)
                if row["objective"] > pilih["objective"] + 1e-12:
                    pilih = row
            global_best = pilih; cfg = dict(pilih["config"])
            print(f"refine global {field}: {pilih['objective']:.6f} {nama(cfg)}",
                  flush=True)

    # AP kategori separabel. Tiap kelas mulai dari seluruh kandidat yang sudah
    # benar-benar dievaluasi, lalu direfinement tanpa membuka TEST.
    per_kelas, jejak = [], {}
    for kelas in ED.NAMA:
        awal = max(ranking, key=lambda x: obj_kelas(x["metrik"], kelas))
        cfg = dict(awal["config"]); best = awal
        trail = [{"tahap": "seed", "objective": obj_kelas(best["metrik"], kelas),
                  "config": dict(cfg)}]
        if cfg["mode"] != "asli":
            for field, values in RUANG.items():
                pilih = best
                skor = obj_kelas(best["metrik"], kelas)
                for value in values:
                    q = dict(cfg); q[field] = value
                    row = nilai(q); ranking.append(row)
                    s = obj_kelas(row["metrik"], kelas)
                    if s > skor + 1e-12:
                        pilih, skor = row, s
                best, cfg = pilih, dict(pilih["config"])
                trail.append({"tahap": field, "objective": skor,
                              "config": dict(cfg)})
        per_kelas.append(cfg); jejak[kelas] = trail
        print(f"route {kelas}: {obj_kelas(best['metrik'], kelas):.6f} {nama(cfg)}",
              flush=True)
    routed = route_per_kelas(det, prob, per_kelas)
    met = FD.coco_detail(coco, paths, routed)
    if FD.objektif(met) + 1e-12 < global_best["objective"]:
        raise RuntimeError("Router per kelas merusak objective VAL")
    return global_best, per_kelas, met, routed, ranking, jejak


def prediksi_fisik(det, prob, cfg_kelas, cfg_obj):
    out = {}
    for s, D in det.items():
        if not len(D):
            out[s] = np.zeros((0, 11), np.float32)
            continue
        if cfg_kelas["mode"] == "asli":
            q = CD.prob_empat(D)
        else:
            q = gabung_prob(D, prob[s], cfg_kelas)
        exist = np.clip(1 - prob[s][:, 4], 1e-8, 1)
        obj = (np.clip(D[:, 4], 1e-8, 1) ** cfg_obj["loc_power"] *
               exist ** cfg_obj["exist_power"])
        kelas = q.argmax(1).astype(np.float32)
        anchor = np.arange(len(D), dtype=np.float32)
        out[s] = np.c_[D[:, :4], obj, kelas, q * obj[:, None], anchor].astype(
            np.float32)
    return out


def cari_fisik(coco, paths, gt, det, prob, cfg_kelas):
    cv = FP.coco_agnostik(coco)
    ranking = []
    for lp in (.25, .50, .75, 1., 1.25, 1.50):
        for ep in (0., .25, .50, .75, 1., 1.50, 2., 3.):
            cfg = {"loc_power": lp, "exist_power": ep}
            p = prediksi_fisik(det, prob, cfg_kelas, cfg)
            m = FP.nilai_coco(cv, paths, p)
            ranking.append({"config": cfg, "metrik": m,
                            "objective": .65 * m["AP50"] + .35 * m["AP50_95"]})
    best = max(ranking, key=lambda x: x["objective"])
    pred = prediksi_fisik(det, prob, cfg_kelas, best["config"])
    score, precision, recall, f1, n_gt = FP.kurva_objek(gt, pred)
    j = int(np.argmax(f1))
    operasi = {"threshold": float(score[j]), "precision": float(precision[j]),
               "recall": float(recall[j]), "f1": float(f1[j]), "n_gt": n_gt}
    return best, pred, operasi, ranking


def nilai_operasi_fisik(gt, pred, threshold):
    score, precision, recall, f1, n_gt = FP.kurva_objek(gt, pred)
    keep = np.flatnonzero(score >= threshold)
    if not len(keep):
        return {"threshold": threshold, "precision": 0., "recall": 0.,
                "f1": 0., "n_gt": n_gt}
    j = int(keep[-1])
    return {"threshold": threshold, "precision": float(precision[j]),
            "recall": float(recall[j]), "f1": float(f1[j]), "n_gt": n_gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pred-val", type=Path, required=True)
    ap.add_argument("--pred-test", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=CD.DS)
    ap.add_argument("--tag", default="detected_final")
    ap.add_argument("--cache-tag")
    ap.add_argument("--cache-dir", type=Path, default=SUB / "results")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    model, mu, sd, tr_args, ck = muat_model(args.checkpoint)
    cache_tag = args.cache_tag or tr_args["cache_tag"]
    common = dict(dataset=args.dataset, cache_dir=args.cache_dir, tag=cache_tag,
                  ukuran=int(tr_args["crop_size"]), pad=float(tr_args["pad"]),
                  pos_iou=float(tr_args["pos_iou"]),
                  neg_iou=float(tr_args["neg_iou"]),
                  rebuild=args.rebuild_cache)

    # FASE VAL: tidak ada operasi pada path/cache/label TEST di atas ini.
    cache_v = CD.bangun_cache(split="val", pred_path=args.pred_val, **common)
    pv = infer_cache(model, cache_v, mu, sd, int(tr_args["ukuran"]),
                     args.batch, args.workers)
    det_v, prob_v = bank_prob(args.pred_val, cache_v["meta"], pv)
    coco_v, paths_v, gt_v = ED.bangun_gt(args.dataset, "val")
    global_best, per_kelas, met_v, pred_v, ranking, jejak = cari_classaware(
        coco_v, paths_v, det_v, prob_v)
    amb_info = ED.pilih_ambang(gt_v, pred_v)
    ambang = {n: amb_info[n]["ambang"] for n in ED.NAMA}
    fisik_best, fisik_v, fisik_op_v, fisik_rank = cari_fisik(
        coco_v, paths_v, gt_v, det_v, prob_v, global_best["config"])
    lock = {
        "checkpoint": CD.fingerprint(args.checkpoint),
        "class_aware": {"global_best": global_best,
                        "per_kelas": dict(zip(ED.NAMA, per_kelas)),
                        "metrik": met_v, "objective": FD.objektif(met_v),
                        "ambang_operasi": amb_info, "jejak": jejak},
        "fisik": {"config": fisik_best["config"],
                   "metrik": fisik_best["metrik"],
                   "objective": fisik_best["objective"],
                   "operasi": fisik_op_v},
    }
    print("TERKUNCI DI VAL", json.dumps(lock, indent=2,
                                        ensure_ascii=False), flush=True)

    # FASE TEST: baru sekarang prediksi, citra, crop, label dan GT TEST dibuka.
    cache_t = CD.bangun_cache(split="test", pred_path=args.pred_test, **common)
    pt = infer_cache(model, cache_t, mu, sd, int(tr_args["ukuran"]),
                     args.batch, args.workers)
    det_t, prob_t = bank_prob(args.pred_test, cache_t["meta"], pt)
    coco_t, paths_t, gt_t = ED.bangun_gt(args.dataset, "test")
    pred_t = route_per_kelas(det_t, prob_t, per_kelas)
    met_t = FD.coco_detail(coco_t, paths_t, pred_t)
    operasi_t = ED.nilai_ambang(gt_t, pred_t, ambang)
    fisik_t = prediksi_fisik(det_t, prob_t, global_best["config"],
                             fisik_best["config"])
    met_fisik_t = FP.nilai_coco(FP.coco_agnostik(coco_t), paths_t, fisik_t)
    op_fisik_t = nilai_operasi_fisik(gt_t, fisik_t, fisik_op_v["threshold"])

    pred_path = {}; fisik_path = {}
    for split, p, f in (("val", pred_v, fisik_v), ("test", pred_t, fisik_t)):
        pp = ROOT / "results" / f"pred_damimas_{args.tag}_{split}.npz"
        fp = ROOT / "results" / f"pred_damimas_{args.tag}_physical_{split}.npz"
        np.savez_compressed(pp, **p); np.savez_compressed(fp, **f)
        pred_path[split], fisik_path[split] = str(pp), str(fp)
    prob_test_path = SUB / "results" / f"damimas_classifier_deteksi_{args.tag}_test.npz"
    zmt = np.load(cache_t["meta"], allow_pickle=False)
    np.savez_compressed(prob_test_path, prob=pt, stem=zmt["stem"],
                        row_idx=zmt["row_idx"])
    zmt.close()
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": ("classifier/checkpoint sudah dipilih VAL saat training; "
                     "scoring+router+threshold dipilih VAL; TEST dibuka setelah lock"),
        "terkunci_di_val": lock,
        "test": {"class_aware": met_t, "titik_operasi": operasi_t,
                 "fisik": met_fisik_t, "operasi_fisik": op_fisik_t},
        "prediksi_class_aware": pred_path,
        "prediksi_fisik": fisik_path,
        "probabilitas_test": str(prob_test_path),
        "ranking_val": sorted(ranking, key=lambda x: x["objective"],
                              reverse=True)[:30],
        "ranking_fisik_val": sorted(fisik_rank, key=lambda x: x["objective"],
                                    reverse=True)[:20],
    }
    output = args.output or ROOT / "results" / f"damimas_{args.tag}.json"
    output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"val": {"class_aware": met_v,
                              "fisik": fisik_best["metrik"]},
                      "test": hasil["test"]}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"-> {output}")


if __name__ == "__main__":
    main()
