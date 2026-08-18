"""Relabel kotak deteksi dengan classifier crop DAMIMAS-only.

Lokalisasi dari kepala fusion dipertahankan. Untuk setiap kotak, probabilitas
C1 dicari dari deteksi YOLO terdekat, crop dinilai ConvNeXt residual, lalu aturan
relabel konservatif/blend dipilih per kelas di VAL berdasarkan AP. TEST baru
dipotong dan diinferensi setelah seluruh routing terkunci.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "pipeline-pertandan"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SUB / "scripts"))
import eval_dump_damimas as ED  # noqa: E402
import fusi_detektor_damimas as FD  # noqa: E402
import classifier_hibrida_damimas as CH  # noqa: E402
import penaut_pertandan as PP  # noqa: E402
import reid_pertandan as RD  # noqa: E402


NAMA = ("B1", "B2", "B3", "B4")


def iou_satu(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def dedup_full(D):
    if not len(D):
        return np.zeros((0, 11), np.float32)
    _, idx = np.unique(D[:, 10], return_index=True)
    return D[np.sort(idx)]


def p_c1_untuk(box, kelas, D):
    if len(D):
        ov = iou_satu(box, D[:, :4])
        j = int(np.argmax(ov))
        if ov[j] >= .15:
            p = np.clip(D[j, 6:10].astype(float), 1e-8, None)
            return (p / p.sum()).astype(np.float32)
    p = np.full(4, .05, np.float32); p[int(kelas)] = .85
    return p


def crop_box(img, box, ukuran=128, pad=.10):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    a1, b1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    a2, b2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    return (cv2.resize(img[b1:b2, a1:a2], (ukuran, ukuran), interpolation=cv2.INTER_AREA)
            if a2 - a1 > 3 and b2 - b1 > 3
            else np.zeros((ukuran, ukuran, 3), np.uint8))


def fitur_aux(img, box, p, side, nv):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    area = bw * bh
    desc = PP.deskriptor(img, box)
    eks = float(p @ np.arange(4)); ent = float(-(p * np.log(np.clip(p, 1e-9, 1))).sum())
    q = np.sort(p)
    geo = [side / max(nv - 1, 1), nv / 8., cx, cy, bw, bh, area,
           bw / max(bh, 1e-6), min(cx, 1 - cx, cy, 1 - cy)]
    return np.r_[desc, p, eks, ent, p.max(), q[-1] - q[-2], geo].astype(np.float32)


def muat_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    mu, sd = np.asarray(ck["aux_mean"], np.float32), np.asarray(ck["aux_std"], np.float32)
    model = CH.Hibrida(len(mu), args.get("backbone", "convnext_tiny"),
                       args.get("mode_c1", "residual")).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, mu, sd, int(args.get("ukuran", 160))


@torch.inference_mode()
def infer_model(model, crops, aux, pc1, ukuran, device):
    out = np.zeros((len(crops), 4), np.float32)
    for i in range(0, len(crops), 128):
        im = torch.from_numpy(crops[i:i + 128].copy()).permute(0, 3, 1, 2)
        x = CH.olah_img(im, False, ukuran)
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            q = torch.softmax(model(x, torch.from_numpy(aux[i:i + 128]).to(device),
                                    torch.from_numpy(pc1[i:i + 128]).to(device)).float(), 1)
        out[i:i + 128] = q.cpu().numpy()
    return out


def bangun_bank(paths, pred_fusi, zbase, model, mu, sd, ukuran, device):
    n_sisi = {}
    for p in paths:
        tree, _sep, _side = p.stem.rpartition("_")
        n_sisi[tree] = n_sisi.get(tree, 0) + 1
    crops, aux, pc1, ref = [], [], [], []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        tree, _sep, side_s = path.stem.rpartition("_")
        side, nv = int(side_s) - 1, n_sisi[tree]
        Df = pred_fusi.get(path.stem, np.zeros((0, 6), np.float32))
        Db = dedup_full(zbase[path.stem] if path.stem in zbase.files
                        else np.zeros((0, 11), np.float32))
        for row in Df:
            p = p_c1_untuk(row[:4], int(row[5]), Db)
            crops.append(crop_box(img, row[:4]))
            pc1.append(p); aux.append(fitur_aux(img, row[:4], p, side, nv))
            ref.append((path.stem, row.astype(np.float32)))
    if not ref:
        return {p.stem: [] for p in paths}
    aux = ((np.stack(aux) - mu) / sd).astype(np.float32)
    pc1 = np.stack(pc1).astype(np.float32)
    ph = infer_model(model, np.stack(crops), aux, pc1, ukuran, device)
    out = {p.stem: [] for p in paths}
    for (stem, row), p0, p1 in zip(ref, pc1, ph):
        out[stem].append({"row": row, "pbase": p0, "ph": p1})
    return out


def daftar_config():
    out = [{"mode": "asli", "nama": "asli"}]
    for source in ("base", "hibrida", "final"):
        for alpha in (.50, .65, .80, 1.0):
            for gamma in (0., .5):
                out.append({"mode": "blend", "source": source, "alpha": alpha,
                            "gamma": gamma,
                            "nama": f"blend_{source}_a{alpha:g}_g{gamma:g}"})
        for theta in (.45, .55, .65, .75):
            for margin in (0., .15):
                for gamma in (0., .5):
                    out.append({"mode": "gate", "source": source, "theta": theta,
                                "margin": margin, "gamma": gamma,
                                "nama": f"gate_{source}_t{theta:g}_m{margin:g}_g{gamma:g}"})
    return out


def sumber_prob(d, nama):
    if nama == "base":
        return d["pbase"]
    if nama == "hibrida":
        return d["ph"]
    return .8 * d["pbase"] + .2 * d["ph"]


def terapkan_satu(d, cfg):
    """Terapkan satu aturan tanpa mengubah identitas kotak sumber."""
    r = d["row"].copy(); asal = int(r[5]); conf = float(r[4])
    if cfg["mode"] == "asli":
        return r
    src = sumber_prob(d, cfg["source"])
    if cfg["mode"] == "blend":
        one = np.zeros(4); one[asal] = 1.
        p = (1 - cfg["alpha"]) * one + cfg["alpha"] * src
        kelas = int(np.argmax(p))
    else:
        urut = np.sort(src)
        boleh = (src.max() >= cfg["theta"] and
                 urut[-1] - urut[-2] >= cfg["margin"])
        kelas = int(np.argmax(src)) if boleh else asal
        p = src if boleh else np.eye(4)[asal]
    r[5] = kelas
    r[4] = conf * float(max(p)) ** cfg["gamma"]
    return r


def baris_asli(det):
    return (np.stack([d["row"] for d in det]).astype(np.float32)
            if det else np.zeros((0, 6), np.float32))


def prediksi(bank, cfg):
    out = {}
    for stem, det in bank.items():
        # Kontrol harus bit-for-bit mempertahankan input fusion. Menjalankan
        # NMS lagi pada kontrol akan membuat selisih yang bukan akibat relabel.
        if cfg["mode"] == "asli":
            out[stem] = baris_asli(det)
        else:
            D = np.asarray([terapkan_satu(d, cfg) for d in det],
                           np.float32).reshape(-1, 6)
            out[stem] = FD.nms_numpy(D, .60)
    return out


def route_kelas_asal(bank, cfgs):
    """Satu aturan per kelas *asal*, sehingga satu kotak tak bisa terduplikasi.

    Routing per kelas keluaran tidak sah untuk relabel: aturan B2 dan B3 dapat
    sama-sama mengambil kotak sumber yang sama. Di sini setiap kotak memilih
    tepat satu aturan berdasarkan label fusion awalnya, lalu NMS membersihkan
    tabrakan yang memang muncul sesudah perubahan label.
    """
    if all(c["mode"] == "asli" for c in cfgs):
        return {stem: baris_asli(det) for stem, det in bank.items()}
    out = {}
    for stem, det in bank.items():
        rows = [terapkan_satu(d, cfgs[int(d["row"][5])]) for d in det]
        D = np.asarray(rows, np.float32).reshape(-1, 6)
        out[stem] = FD.nms_numpy(D, .60)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-val", type=Path,
                    default=ROOT / "results" / "pred_damimas_fusi_yolo_awal_val.npz")
    ap.add_argument("--pred-test", type=Path,
                    default=ROOT / "results" / "pred_damimas_fusi_yolo_awal_test.npz")
    ap.add_argument("--base-val", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_val.npz")
    ap.add_argument("--base-test", type=Path,
                    default=SUB / "results" / "pred_skorpenuh_test.npz")
    ap.add_argument("--checkpoint", type=Path, default=SUB / "runs" /
                    "classifier_hibrida_damimas_convnext_tiny_s42" / "best.pt")
    ap.add_argument("--dataset", type=Path, default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "damimas_relabel_classifier.json")
    ap.add_argument("--pred-val-out", type=Path,
                    default=ROOT / "results" / "pred_damimas_relabel_val.npz")
    ap.add_argument("--pred-test-out", type=Path,
                    default=ROOT / "results" / "pred_damimas_relabel_test.npz")
    ap.add_argument("--route-passes", type=int, default=2,
                    help="jumlah lintasan coordinate-ascent kelas asal di VAL")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mu, sd, ukuran = muat_model(args.checkpoint, device)

    coco_v, paths_v, gt_v = ED.bangun_gt(args.dataset, "val")
    pfv = ED.muat_prediksi(args.pred_val, set(gt_v))
    zbv = np.load(args.base_val, allow_pickle=True)
    print("infer crop VAL...", flush=True)
    bank_v = bangun_bank(paths_v, pfv, zbv, model, mu, sd, ukuran, device)
    cfgs = daftar_config(); preds_v = {}; ranking = []
    for i, cfg in enumerate(cfgs, 1):
        pred = prediksi(bank_v, cfg); preds_v[cfg["nama"]] = pred
        m = FD.coco_detail(coco_v, paths_v, pred)
        ranking.append({"config": cfg, "metrik": m,
                        "objective": FD.objektif(m)})
        print(f"{i:03d}/{len(cfgs)} {cfg['nama']} mAP50={m['mAP50']:.4f} "
              f"mAP50-95={m['mAP50_95']:.4f}", flush=True)
    # Mulai dari aturan global terbaik, lalu coordinate-ascent per kelas asal.
    # Setiap kandidat dinilai sebagai pipeline utuh, bukan memilih AP kelas
    # secara terpisah yang dapat menghasilkan konflik antarkelas.
    global_best = max(ranking, key=lambda q: q["objective"])
    per_kelas = [global_best["config"]] * 4
    pred_val = route_kelas_asal(bank_v, per_kelas)
    mv = FD.coco_detail(coco_v, paths_v, pred_val)
    route_obj = FD.objektif(mv)
    jejak_route = []
    for lintasan in range(max(args.route_passes, 0)):
        berubah = False
        for k, nama in enumerate(NAMA):
            terbaik_k = (route_obj, per_kelas[k], mv, pred_val)
            for cfg in cfgs:
                coba = list(per_kelas); coba[k] = cfg
                pv = route_kelas_asal(bank_v, coba)
                met = FD.coco_detail(coco_v, paths_v, pv)
                q = (FD.objektif(met), cfg, met, pv)
                if q[0] > terbaik_k[0] + 1e-12:
                    terbaik_k = q
            obj_baru, cfg_baru, met_baru, pv_baru = terbaik_k
            if obj_baru > route_obj + 1e-12:
                berubah |= cfg_baru["nama"] != per_kelas[k]["nama"]
                per_kelas[k] = cfg_baru
                route_obj, mv, pred_val = obj_baru, met_baru, pv_baru
            jejak_route.append({"lintasan": lintasan + 1, "kelas_asal": nama,
                                "config": per_kelas[k], "objective": route_obj})
            print(f"route pass={lintasan + 1} asal={nama} "
                  f"{per_kelas[k]['nama']} obj={route_obj:.6f}", flush=True)
        if not berubah:
            break
    original = next(q for q in ranking if q["config"]["mode"] == "asli")
    if route_obj + 1e-12 < original["objective"]:
        per_kelas = [original["config"]] * 4
        pred_val, mv, route_obj = preds_v["asli"], original["metrik"], original["objective"]
    terkunci = {"routing": "kelas_asal", "per_kelas_asal": per_kelas,
                "global_awal": global_best["config"], "metrik": mv,
                "objective": route_obj, "jejak": jejak_route}
    print("TERKUNCI:", json.dumps(terkunci, indent=2), flush=True)

    # Baru sekarang memuat citra, label COCO, dan prediksi TEST.
    coco_t, paths_t, gt_t = ED.bangun_gt(args.dataset, "test")
    pft = ED.muat_prediksi(args.pred_test, set(gt_t))
    zbt = np.load(args.base_test, allow_pickle=True)
    print("infer crop TEST...", flush=True)
    bank_t = bangun_bank(paths_t, pft, zbt, model, mu, sd, ukuran, device)
    pred_test = route_kelas_asal(bank_t, per_kelas)
    mt = FD.coco_detail(coco_t, paths_t, pred_test)
    ambang_info = ED.pilih_ambang(gt_v, pred_val)
    ambang = {n: ambang_info[n]["ambang"] for n in NAMA}
    operasi = {"val": ED.nilai_ambang(gt_v, pred_val, ambang),
               "test": ED.nilai_ambang(gt_t, pred_test, ambang)}
    np.savez_compressed(args.pred_val_out, **pred_val)
    np.savez_compressed(args.pred_test_out, **pred_test)
    hasil = {"dataset": "SawitMVC-YOLO-Damimas",
             "protokol": "classifier train-only; seluruh relabel/routing di VAL; TEST sekali",
             "checkpoint": str(args.checkpoint), "input_fusion": {
                 "val": str(args.pred_val), "test": str(args.pred_test)},
             "terkunci_di_val": terkunci, "test": mt,
             "ambang_dipilih_di_val": ambang_info, "titik_operasi": operasi,
             "ranking_val": sorted(ranking, key=lambda q: q["objective"], reverse=True)[:20],
             "prediksi": {"val": str(args.pred_val_out), "test": str(args.pred_test_out)}}
    args.output.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"val": mv, "test": mt, "operasi_test": operasi["test"]},
                     indent=2, ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
