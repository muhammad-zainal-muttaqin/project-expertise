"""Seleksi greedy dan fusi per-kelas untuk bank detektor DAMIMAS.

Semua kandidat dibentuk dan dipilih memakai VAL. Hanya konfigurasi final yang
kemudian diterapkan ke TEST. Fusi sengaja boleh berbeda per kelas karena
arsitektur terbaik untuk B1 tidak harus terbaik untuk B4.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
NAMA = ("B1", "B2", "B3", "B4")
sys.path.insert(0, str(Path(__file__).parent))
import eval_dump_damimas as ED  # noqa: E402


def parse_anggota(items):
    out = {}
    for x in items:
        bagian = x.split("=")
        if len(bagian) == 3:
            nama, val, test = bagian
            paths = {"val_path": val, "test_path": test}
        elif len(bagian) == 4:
            nama, train, val, test = bagian
            paths = {"train_path": train, "val_path": val,
                     "test_path": test}
        else:
            raise SystemExit(
                "--anggota harus NAMA=VAL.npz=TEST.npz atau "
                "NAMA=TRAIN.npz=VAL.npz=TEST.npz")
        if nama in out:
            raise SystemExit(f"Nama anggota duplikat: {nama}")
        out[nama] = {k: str(Path(v).resolve()) for k, v in paths.items()}
    if len(out) < 2:
        raise SystemExit("Fusi membutuhkan sedikitnya dua anggota")
    return out


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-12)


def nms_numpy(D: np.ndarray, ambang: float) -> np.ndarray:
    if not len(D):
        return D
    keep = []
    for k in range(4):
        idx = np.flatnonzero(D[:, 5].astype(int) == k)
        urut = idx[np.argsort(-D[idx, 4])]
        while len(urut):
            i = int(urut[0]); keep.append(i)
            urut = urut[1:]
            if len(urut):
                urut = urut[iou(D[i, :4], D[urut, :4]) < ambang]
    return D[np.asarray(keep, int)] if keep else np.zeros((0, 6), np.float32)


def wbf_satu(rows: list[tuple[np.ndarray, float, int]], ambang: float,
             aturan_skor: str, total_bobot: float) -> np.ndarray:
    """WBF satu kelas. Tuple berisi (baris, bobot_model, id_model)."""
    if not rows:
        return np.zeros((0, 6), np.float32)
    rows.sort(key=lambda x: float(x[0][4]) * x[1], reverse=True)
    grup: list[list[tuple[np.ndarray, float, int]]] = []
    pusat: list[np.ndarray] = []
    for row, wm, mid in rows:
        if pusat:
            ov = iou(row[:4], np.stack(pusat))
            j = int(np.argmax(ov))
        else:
            ov, j = np.zeros(0), -1
        if len(ov) and ov[j] >= ambang:
            grup[j].append((row, wm, mid))
        else:
            grup.append([(row, wm, mid)]); pusat.append(row[:4].copy())
            continue
        g = grup[j]
        ww = np.asarray([float(r[0][4]) * r[1] for r in g])
        pusat[j] = np.average(np.stack([r[0][:4] for r in g]), axis=0,
                              weights=np.maximum(ww, 1e-9))

    out = []
    for c, g in zip(pusat, grup):
        skor = np.asarray([float(r[0][4]) for r in g])
        bobot = np.asarray([r[1] for r in g])
        if aturan_skor == "max":
            s = float(skor.max())
        elif aturan_skor == "noisy_or":
            # Bukti independen dari beberapa arsitektur menaikkan confidence.
            # Bobot bekerja sebagai eksponen evidence; seluruh kandidat bobot
            # tetap dipilih di VAL sehingga perbedaan kalibrasi tidak ditebak.
            s = float(1. - np.prod(np.power(
                np.clip(1. - skor, 1e-9, 1.), bobot)))
        elif aturan_skor == "avg_all":
            # Satu model boleh menyumbang paling banyak sekali ke penyebut.
            terbaik = {}
            for sc, bw, (_, _, mid) in zip(skor, bobot, g):
                terbaik[mid] = max(terbaik.get(mid, 0.), float(sc * bw))
            s = sum(terbaik.values()) / max(total_bobot, 1e-9)
        else:
            s = float(np.average(skor, weights=np.maximum(bobot, 1e-9)))
        kelas = float(g[0][0][5])
        out.append(np.r_[c, np.clip(s, 0, 1), kelas])
    return np.asarray(out, np.float32)


def fusi_stem(blok: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    nama = cfg["anggota"]
    bobot = cfg["bobot"]
    if cfg["jenis"] == "nms":
        rr = []
        maks = max(bobot)
        for n, w in zip(nama, bobot):
            d = blok[n].copy()
            if len(d):
                d[:, 4] *= w / maks
                rr.append(d)
        D = np.concatenate(rr) if rr else np.zeros((0, 6), np.float32)
        return nms_numpy(D, cfg["iou"])

    semua = []
    total = float(sum(bobot))
    for k in range(4):
        rows = []
        for mid, (n, w) in enumerate(zip(nama, bobot)):
            d = blok[n]
            rows.extend((r.copy(), float(w), mid)
                        for r in d[d[:, 5].astype(int) == k])
        q = wbf_satu(rows, cfg["iou"], cfg["skor"], total)
        if len(q):
            semua.append(q)
    return np.concatenate(semua) if semua else np.zeros((0, 6), np.float32)


def buat_pred(cfg: dict, bank: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if cfg["jenis"] == "individual":
        return bank[cfg["anggota"]]
    if cfg["jenis"] == "route":
        sumber = [buat_pred(q, bank) for q in cfg["per_kelas"]]
        stems = bank[next(iter(bank))].keys()
        out = {}
        for stem in stems:
            rr = [sumber[k][stem][sumber[k][stem][:, 5].astype(int) == k]
                  for k in range(4)]
            out[stem] = (np.concatenate([x for x in rr if len(x)])
                         if any(len(x) for x in rr) else np.zeros((0, 6), np.float32))
        return out
    stems = bank[next(iter(bank))].keys()
    return {stem: fusi_stem({n: bank[n][stem] for n in cfg["anggota"]}, cfg)
            for stem in stems}


def coco_detail(coco, paths, pred):
    det = []
    for image_id, path in enumerate(paths, 1):
        for row in pred[path.stem]:
            x1, y1, x2, y2, conf, kelas = row[:6]
            det.append({"image_id": image_id, "category_id": int(kelas) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(conf)})
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(det), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    p = ev.eval["precision"]
    ap50, ap95 = {}, {}
    for k, nama in enumerate(NAMA):
        x50, x = p[0, :, k, 0, 2], p[:, :, k, 0, 2]
        ap50[nama] = float(x50[x50 > -1].mean()) if (x50 > -1).any() else 0.
        ap95[nama] = float(x[x > -1].mean()) if (x > -1).any() else 0.
    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0]),
            "AP50_per_kelas": ap50, "AP50_95_per_kelas": ap95}


def objektif(m):
    return .65 * m["mAP50"] + .35 * m["mAP50_95"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anggota", action="append", required=True,
                    help=("ulang: NAMA=VAL.npz=TEST.npz atau "
                          "NAMA=TRAIN.npz=VAL.npz=TEST.npz"))
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-YOLO-Damimas"))
    ap.add_argument("--keluaran", type=Path,
                    default=ROOT / "results" / "damimas_fusi_detektor.json")
    ap.add_argument("--pred-val-out", type=Path,
                    default=ROOT / "results" / "pred_damimas_fusi_val.npz")
    ap.add_argument("--pred-test-out", type=Path,
                    default=ROOT / "results" / "pred_damimas_fusi_test.npz")
    ap.add_argument("--pred-train-out", type=Path,
                    default=ROOT / "results" / "pred_damimas_fusi_train.npz")
    args = ap.parse_args()
    sumber = parse_anggota(args.anggota)

    # Fase seleksi hanya membuka VAL. Selain mencegah pemakaian label test,
    # pemisahan akses ini membuat urutan lock dapat diaudit dari eksekusi:
    # dump dan anotasi TEST bahkan belum dimuat saat konfigurasi dibandingkan.
    coco_v, paths_v, gt_v = ED.bangun_gt(args.dataset, "val")
    bank_v = {
        nama: ED.muat_prediksi(Path(info["val_path"]), set(gt_v))
        for nama, info in sumber.items()
    }

    configs = [{"jenis": "individual", "anggota": n} for n in sumber]
    names = list(sumber)
    # Pasangan dan seluruh bank memberi keragaman; tiga rumus skor menangani
    # perbedaan kalibrasi confidence antar-arsitektur.
    subsets = [sub for ukuran in range(2, len(names) + 1)
               for sub in combinations(names, ukuran)]
    for sub in subsets:
        # Confidence antar-arsitektur tidak terkalibrasi sama. Equal-weight
        # tetap kandidat, lalu setiap anggota diberi kesempatan menjadi sumber
        # dominan. Skala absolut tidak penting; rasio bobotlah yang diuji.
        weight_grid = [[1.] * len(sub)]
        for j in range(len(sub)):
            w = [1.] * len(sub); w[j] = 2.
            weight_grid.append(w)
        for ambang in (.45, .55, .65, .75):
            for bobot in weight_grid:
                configs.append({"jenis": "nms", "anggota": list(sub),
                                "bobot": bobot, "iou": ambang})
                for skor in ("avg_present", "avg_all", "max", "noisy_or"):
                    configs.append({"jenis": "wbf", "anggota": list(sub),
                                    "bobot": bobot, "iou": ambang,
                                    "skor": skor})

    dinilai = []
    for i, cfg in enumerate(configs, 1):
        pred = buat_pred(cfg, bank_v)
        m = coco_detail(coco_v, paths_v, pred)
        dinilai.append({"config": cfg, "metrik": m, "objective": objektif(m)})
        print(f"{i:03d}/{len(configs)} obj={objektif(m):.4f} "
              f"mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f}", flush=True)

    # Routing per kelas memilih kandidat dengan skor kelasnya sendiri.
    per_kelas = []
    for k, nama in enumerate(NAMA):
        q = max(dinilai, key=lambda r: .65 * r["metrik"]["AP50_per_kelas"][nama]
                + .35 * r["metrik"]["AP50_95_per_kelas"][nama])
        per_kelas.append(q["config"])
    route = {"jenis": "route", "per_kelas": per_kelas}
    pred_route = buat_pred(route, bank_v)
    mr = coco_detail(coco_v, paths_v, pred_route)
    dinilai.append({"config": route, "metrik": mr, "objective": objektif(mr)})
    terbaik = max(dinilai, key=lambda r: r["objective"])

    # Konfigurasi kini terkunci; baru buka anotasi dan dump prediksi TEST.
    print("TERKUNCI DI VAL", json.dumps(terbaik, indent=2,
                                        ensure_ascii=False), flush=True)
    pred_val = buat_pred(terbaik["config"], bank_v)

    # TRAIN tidak ikut seleksi. Bila semua anggota menyediakannya, terapkan
    # lock yang sama agar proposal/linker/counting tidak mengalami domain
    # shift base-vs-fusion antara train dan val/test.
    pred_train = None
    if all("train_path" in info for info in sumber.values()):
        stems_tr = {p.stem for p in (args.dataset / "images" / "train").glob("*.jpg")}
        bank_tr = {
            nama: ED.muat_prediksi(Path(info["train_path"]), stems_tr)
            for nama, info in sumber.items()
        }
        pred_train = buat_pred(terbaik["config"], bank_tr)

    coco_t, paths_t, gt_t = ED.bangun_gt(args.dataset, "test")
    bank_t = {
        nama: ED.muat_prediksi(Path(info["test_path"]), set(gt_t))
        for nama, info in sumber.items()
    }
    pred_test = buat_pred(terbaik["config"], bank_t)
    mt = coco_detail(coco_t, paths_t, pred_test)
    ambang_info = ED.pilih_ambang(gt_v, pred_val)
    ambang = {n: ambang_info[n]["ambang"] for n in NAMA}
    operasi = {"val": ED.nilai_ambang(gt_v, pred_val, ambang),
               "test": ED.nilai_ambang(gt_t, pred_test, ambang)}

    args.pred_val_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.pred_val_out, **pred_val)
    np.savez_compressed(args.pred_test_out, **pred_test)
    pred_paths = {"val": str(args.pred_val_out), "test": str(args.pred_test_out)}
    if pred_train is not None:
        np.savez_compressed(args.pred_train_out, **pred_train)
        pred_paths["train"] = str(args.pred_train_out)
    hasil = {
        "dataset": "SawitMVC-YOLO-Damimas",
        "protokol": "semua seleksi/routing di VAL; TEST sekali setelah terkunci",
        "anggota": sumber,
        "terpilih": terbaik,
        "test": mt,
        "ambang_dipilih_di_val": ambang_info,
        "titik_operasi": operasi,
        "ranking_val": sorted(dinilai, key=lambda r: r["objective"], reverse=True)[:20],
        "prediksi": pred_paths,
    }
    args.keluaran.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(json.dumps({"terpilih": terbaik, "test": mt,
                      "operasi_test": operasi["test"]}, indent=2, ensure_ascii=False))
    print(f"-> {args.keluaran}")


if __name__ == "__main__":
    main()
