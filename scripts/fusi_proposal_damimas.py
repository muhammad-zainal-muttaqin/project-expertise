"""Fusi class-agnostic untuk bank proposal fisik DAMIMAS.

Kepala mAP class-aware dan kepala identitas fisik tidak harus memakai keluaran
yang sama. Skrip ini melipat hipotesis kelas menjadi satu proposal per objek,
lalu memilih fusi lintas-detektor dari AP lokalisasi VAL. Vektor empat kelas
tetap dibawa sebagai fitur bagi classifier/linker, tetapi tidak memengaruhi
evaluasi lokalisasi.

Format anggota:

    --anggota nama=val.npz=test.npz
    --anggota nama=train.npz=val.npz=test.npz

Nama yang diberikan lewat ``--agnostik`` dianggap detektor satu-kelas: skornya
dipakai untuk objectness dan tidak disalahartikan sebagai bukti kelas B1.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_dump_damimas as ED  # noqa: E402


def parse_anggota(items):
    out = {}
    for item in items:
        p = item.split("=")
        if len(p) == 3:
            nama, val, test = p
            paths = {"val": val, "test": test}
        elif len(p) == 4:
            nama, train, val, test = p
            paths = {"train": train, "val": val, "test": test}
        else:
            raise ValueError("--anggota harus NAMA=VAL=TEST atau NAMA=TRAIN=VAL=TEST")
        if nama in out:
            raise ValueError(f"Nama anggota duplikat: {nama}")
        out[nama] = {s: str(Path(q).resolve()) for s, q in paths.items()}
    if not out:
        raise ValueError("Sedikitnya satu --anggota diperlukan")
    return out


def iou(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-12)


def kelompok_greedy(rows, ambang):
    """Kelompokkan indeks secara class-agnostic, tertinggi lebih dahulu."""
    if not len(rows):
        return []
    urut = list(np.argsort(-rows[:, 4]))
    grup = []
    while urut:
        i = int(urut.pop(0)); g = [i]
        if urut:
            ov = iou(rows[i, :4], rows[np.asarray(urut), :4])
            g += [j for j, q in zip(urut, ov) if q >= ambang]
            urut = [j for j, q in zip(urut, ov) if q < ambang]
        grup.append(g)
    return grup


def lipat_sumber(rows, ambang, agnostik, mode_box="wavg"):
    rows = np.asarray(rows, np.float32).reshape(-1, rows.shape[-1] if rows.ndim == 2 else 6)
    out = []
    for g in kelompok_greedy(rows, ambang):
        q = rows[g]
        terbaik = q[int(np.argmax(q[:, 4]))]
        w = np.maximum(q[:, 4].astype(float), 1e-7)
        box = (terbaik[:4].copy() if mode_box == "max" else
               np.average(q[:, :4], axis=0, weights=w))
        skor_k = np.zeros(4, np.float32)
        if not agnostik:
            for r in q:
                k = int(r[5])
                if 0 <= k < 4:
                    skor_k[k] = max(skor_k[k], float(r[4]))
        out.append({"box": box.astype(np.float32),
                    "conf": float(terbaik[4]), "kelas": skor_k})
    return out


def fusi_stem(blok, cfg, agnostik):
    lokal = []
    for mid, (nama, bobot) in enumerate(zip(cfg["anggota"], cfg["bobot"])):
        for d in lipat_sumber(blok[nama], cfg["iou_intra"], nama in agnostik,
                              cfg.get("box_intra", "wavg")):
            lokal.append((d, float(bobot), mid))
    lokal.sort(key=lambda x: -x[0]["conf"] * x[1])
    grup, pusat = [], []
    for d, w, mid in lokal:
        if pusat:
            ov = iou(d["box"], np.stack(pusat)); j = int(np.argmax(ov))
        else:
            ov, j = np.zeros(0), -1
        # Satu sumber hanya boleh menyumbang satu proposal ke satu cluster.
        if len(ov) and ov[j] >= cfg["iou_inter"] and all(x[2] != mid for x in grup[j]):
            grup[j].append((d, w, mid))
            ww = [max(x[0]["conf"] * x[1], 1e-8) for x in grup[j]]
            if cfg.get("box_inter", "wavg") == "max":
                pusat[j] = grup[j][int(np.argmax(ww))][0]["box"].copy()
            else:
                pusat[j] = np.average(np.stack([x[0]["box"] for x in grup[j]]),
                                      axis=0, weights=ww)
        else:
            grup.append([(d, w, mid)]); pusat.append(d["box"].copy())

    out = []
    total_bobot = float(sum(cfg["bobot"]))
    for anchor, (box, g) in enumerate(zip(pusat, grup)):
        conf = np.asarray([x[0]["conf"] for x in g], float)
        bw = np.asarray([x[1] for x in g], float)
        if cfg["skor"] == "max":
            obj = float(conf.max())
        elif cfg["skor"] == "avg_all":
            obj = float(np.sum(conf * bw) / max(total_bobot, 1e-9))
        elif cfg["skor"] == "noisy_or":
            obj = float(1 - np.prod(1 - np.clip(conf, 0, 1)))
        else:
            obj = float(np.average(conf, weights=bw))
        bukti = [(x[0]["kelas"], x[1]) for x in g if x[0]["kelas"].sum() > 0]
        if bukti:
            p = np.average(np.stack([x[0] for x in bukti]), axis=0,
                           weights=[x[1] for x in bukti]).astype(np.float32)
            # Objectness dari detektor agnostik boleh lebih tinggi, tetapi
            # distribusi kelas tetap berasal dari detektor class-aware.
            pn = p / max(float(p.max()), 1e-9) * obj
        else:
            pn = np.full(4, obj / 4, np.float32)
        kelas = int(np.argmax(pn))
        out.append(np.r_[box, obj, kelas, pn, float(anchor)])
    return np.asarray(out, np.float32).reshape(-1, 11)


def buat_pred(cfg, bank, agnostik):
    stems = next(iter(bank.values())).keys()
    return {s: fusi_stem({n: bank[n][s] for n in cfg["anggota"]}, cfg, agnostik)
            for s in stems}


def coco_agnostik(coco):
    d = copy.deepcopy(coco.dataset)
    d["categories"] = [{"id": 1, "name": "tandan"}]
    for a in d["annotations"]:
        a["category_id"] = 1
    c = COCO(); c.dataset = d; c.createIndex()
    return c


def nilai_coco(coco, paths, pred):
    det = []
    for image_id, path in enumerate(paths, 1):
        for r in pred[path.stem]:
            x1, y1, x2, y2, conf = r[:5]
            det.append({"image_id": image_id, "category_id": 1,
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(conf)})
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(det), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP50": float(ev.stats[1]), "AP50_95": float(ev.stats[0])}


def kurva_objek(gt, pred):
    cand, n_gt = [], sum(len(x) for x in gt.values())
    G = {s: q[:, 1:5] for s, q in gt.items()}
    for s, d in pred.items():
        cand += [(float(r[4]), s, r[:4]) for r in d]
    cand.sort(reverse=True, key=lambda x: x[0])
    used = {s: np.zeros(len(g), bool) for s, g in G.items()}
    tp, fp, score = [], [], []
    for conf, s, box in cand:
        ov = iou(box, G[s]); avail = np.flatnonzero(~used[s])
        ok = False
        if len(avail):
            j = avail[int(np.argmax(ov[avail]))]
            ok = bool(ov[j] >= .5)
            if ok:
                used[s][j] = True
        tp.append(ok); fp.append(not ok); score.append(conf)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    p = tp / np.maximum(tp + fp, 1); r = tp / max(n_gt, 1)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return np.asarray(score), p, r, f, n_gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anggota", action="append", required=True)
    ap.add_argument("--agnostik", action="append", default=[])
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "damimas_fusi_proposal.json")
    ap.add_argument("--tag", default="damimas_proposal")
    args = ap.parse_args()
    sumber = parse_anggota(args.anggota)
    agnostik = set(args.agnostik)
    if not agnostik <= set(sumber):
        raise ValueError(f"--agnostik tidak dikenal: {sorted(agnostik-set(sumber))}")

    coco_v, paths_v, gt_v = ED.bangun_gt(args.dataset, "val")
    bank_v = {n: ED.muat_prediksi(Path(q["val"]), set(gt_v))
              for n, q in sumber.items()}
    nama = list(sumber)
    subsets = [(n,) for n in nama]
    subsets += list(itertools.combinations(nama, 2))
    if len(nama) >= 3:
        subsets += [tuple(nama)]
    configs = []
    for sub in subsets:
        for intra in (.60, .70, .80):
            for box_intra in ("max", "wavg"):
                for inter in ((.55, .65, .75) if len(sub) > 1 else (.65,)):
                    for box_inter in (("max", "wavg") if len(sub) > 1 else ("max",)):
                        for skor in (("max", "avg_present", "avg_all", "noisy_or")
                                     if len(sub) > 1 else ("max",)):
                            configs.append({"anggota": list(sub),
                                            "bobot": [1.] * len(sub),
                                            "iou_intra": intra,
                                            "box_intra": box_intra,
                                            "iou_inter": inter,
                                            "box_inter": box_inter,
                                            "skor": skor})
    cv = coco_agnostik(coco_v)
    ranking = []
    for i, cfg in enumerate(configs, 1):
        p = buat_pred(cfg, bank_v, agnostik); m = nilai_coco(cv, paths_v, p)
        obj = .65 * m["AP50"] + .35 * m["AP50_95"]
        ranking.append({"config": cfg, "metrik": m, "objective": obj})
        if i % 25 == 0 or i == len(configs):
            print(f"{i}/{len(configs)} best={max(x['objective'] for x in ranking):.5f}",
                  flush=True)
    terbaik = max(ranking, key=lambda x: x["objective"])
    pred_val = buat_pred(terbaik["config"], bank_v, agnostik)
    score, pv, rv, fv, ng = kurva_objek(gt_v, pred_val)
    j = int(np.argmax(fv)); threshold = float(score[j])
    operasi_val = {"threshold": threshold, "precision": float(pv[j]),
                   "recall": float(rv[j]), "f1": float(fv[j]), "n_gt": ng}
    print("TERKUNCI DI VAL", json.dumps({**terbaik, "operasi": operasi_val},
                                         indent=2), flush=True)

    # TEST baru dibuka sesudah fusi dan threshold terkunci.
    coco_t, paths_t, gt_t = ED.bangun_gt(args.dataset, "test")
    bank_t = {n: ED.muat_prediksi(Path(q["test"]), set(gt_t))
              for n, q in sumber.items()}
    pred_test = buat_pred(terbaik["config"], bank_t, agnostik)
    mt = nilai_coco(coco_agnostik(coco_t), paths_t, pred_test)
    st, pt, rt, ft, nt = kurva_objek(gt_t, pred_test)
    keep = np.flatnonzero(st >= threshold)
    if len(keep):
        k = int(keep[-1]); operasi_test = {
            "threshold": threshold, "precision": float(pt[k]),
            "recall": float(rt[k]), "f1": float(ft[k]), "n_gt": nt}
    else:
        operasi_test = {"threshold": threshold, "precision": 0., "recall": 0.,
                        "f1": 0., "n_gt": nt}

    out_pred = {}
    semua_train = all("train" in q for q in sumber.values())
    if semua_train:
        stems = {p.stem for p in (args.dataset / "images" / "train").glob("*.jpg")}
        bank_tr = {n: ED.muat_prediksi(Path(q["train"]), stems)
                   for n, q in sumber.items()}
        out_pred["train"] = buat_pred(terbaik["config"], bank_tr, agnostik)
    out_pred.update({"val": pred_val, "test": pred_test})
    pred_paths = {}
    for split, pred in out_pred.items():
        path = ROOT / "results" / f"pred_{args.tag}_{split}.npz"
        np.savez_compressed(path, **pred); pred_paths[split] = str(path)

    hasil = {"dataset": "SawitMVC-YOLO-Damimas",
             "protokol": "fusi+threshold dipilih VAL; TEST sekali",
             "anggota": sumber, "agnostik": sorted(agnostik),
             "terpilih_di_val": terbaik, "operasi_val": operasi_val,
             "test": mt, "operasi_test": operasi_test,
             "ranking_val": sorted(ranking, key=lambda x: x["objective"],
                                   reverse=True)[:30],
             "prediksi_fisik": pred_paths}
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"val": terbaik["metrik"], "test": mt,
                      "operasi_test": operasi_test}, indent=2), flush=True)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
