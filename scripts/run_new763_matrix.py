"""Runner sequence untuk matriks baseline 763 pohon.

Runner bersifat idempotent: job yang sudah memiliki ``DONE.json`` dilewati,
job yang gagal dicatat dan tidak menyamarkan error, sedangkan job berikutnya
tetap bisa diteruskan. Training berjalan satu per satu agar VRAM tidak
terfragmentasi dan setiap run memiliki artefak sendiri.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DATA = Path("/workspace/SawitMVC-Depth-YOLO-RGB")
PROJECT = ROOT / "runs_new763"
RESULTS = ROOT / "results" / "new763"
PRED = RESULTS / "predictions"
ARCH_KIND = {"yolo26l": "yolo", "rtdetr_l": "rtdetr", "rfdetr_l": "rfdetr"}


def jobs(seeds: list[int]) -> list[dict]:
    # Urutan sengaja: anggota yang sudah terbukti pada dataset lama lebih dulu,
    # lalu RF-DETR yang membutuhkan stack RF-DETR dan checkpoint khusus.
    out = []
    for seed in seeds:
        for arch in ("yolo26l", "rtdetr_l", "rfdetr_l"):
            out.append({
                "arch": arch, "seed": seed, "imgsz": 1280,
                "epochs": 60 if arch != "rfdetr_l" else 20,
                "patience": 15 if arch != "rfdetr_l" else 5,
                "batch": 4, "grad_accum": 4,
                "name": f"{arch}_rgb_s{seed}_i1280",
            })
    return out


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=ROOT, check=False)
    if proc.returncode:
        raise RuntimeError(f"command gagal ({proc.returncode}); lihat {log}")


def copy_history(run_dir: Path, name: str) -> None:
    history = ROOT / "results" / "riwayat_epoch_new763"
    history.mkdir(parents=True, exist_ok=True)
    candidates = ["results.csv", "args.yaml", "training_config.json",
                  "metrics.csv", "baseline_args.json"]
    for filename in candidates:
        src = run_dir / filename
        if src.is_file():
            shutil.copy2(src, history / f"{name}__{filename}")


def best_weights(job: dict) -> Path:
    run_dir = PROJECT / job["name"]
    if job["arch"] == "rfdetr_l":
        candidates = [run_dir / "checkpoint_best_ema.pth",
                      run_dir / "checkpoint_best_total.pth",
                      run_dir / "checkpoint_last.pth"]
    else:
        candidates = [run_dir / "weights" / "best.pt",
                      run_dir / "weights" / "last.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"bobot tidak ditemukan: {candidates}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    ap.add_argument("--only", nargs="*", default=None,
                    help="nama job; default semua")
    ap.add_argument("--start-at", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"venv belum siap: {PYTHON}")
    DATA.joinpath("data.yaml").is_file() or raise_error("data.yaml hilang")
    PROJECT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dataset": str(DATA.resolve()), "protocol": "top3_rgb_baseline",
        "jobs": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for index, job in enumerate(jobs(args.seeds)):
        if index < args.start_at or (args.only and job["name"] not in args.only):
            continue
        name = job["name"]
        state = manifest["jobs"].get(name, {})
        if state.get("status") == "done":
            print(f"SKIP {name}: done", flush=True)
            continue
        log = RESULTS / "logs" / f"{name}.log"
        manifest["jobs"][name] = {**job, "status": "training", "log": str(log)}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n===== TRAIN {name} =====", flush=True)
        try:
            train_cmd = [str(PYTHON), str(ROOT / "scripts" / "train_baseline_new763.py"),
                         "--arch", job["arch"], "--data", str(DATA),
                         "--project", str(PROJECT), "--name", name,
                         "--epochs", str(job["epochs"]), "--patience", str(job["patience"]),
                         "--imgsz", str(job["imgsz"]), "--batch", str(job["batch"]),
                         "--grad-accum", str(job["grad_accum"]), "--seed", str(job["seed"])]
            run(train_cmd, log)
            run_dir = PROJECT / name
            copy_history(run_dir, name)
            weight = best_weights(job)
            state = {**job, "status": "trained", "weights": str(weight.resolve()),
                     "log": str(log)}
            manifest["jobs"][name] = state
            manifest_path.write_text(json.dumps(manifest, indent=2))
            if not args.no_eval:
                print(f"===== EVAL {name} =====", flush=True)
                eval_json = RESULTS / f"{name}.json"
                eval_cmd = [str(PYTHON), str(ROOT / "scripts" / "eval_new763_pycoco.py"),
                            "--kind", ARCH_KIND[job["arch"]], "--weights", str(weight),
                            "--dataset", str(DATA), "--run-name", name,
                            "--imgsz", str(job["imgsz"]), "--out-json", str(eval_json),
                            "--pred-dir", str(PRED)]
                run(eval_cmd, log)
                state.update({"status": "done", "result": str(eval_json.resolve())})
                manifest["jobs"][name] = state
                manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"SELESAI {name}", flush=True)
        except Exception as exc:
            manifest["jobs"][name] = {**job, "status": "failed",
                                       "error": str(exc), "log": str(log)}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"GAGAL {name}: {exc}", flush=True)
    print(f"Manifest: {manifest_path}")
    return 0


def raise_error(message: str) -> None:
    raise FileNotFoundError(message)


if __name__ == "__main__":
    raise SystemExit(main())
