"""Run the three seed-42 detectors on the combined 1716-record RGB corpus.

The script waits for the currently active seed-42 RF-DETR run to release the
GPU, then trains at most two combined-corpus jobs at once. RF-DETR may pair
with either Ultralytics model; YOLO26-L and RT-DETR-L are never paired.
Evaluation is serialized after training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DATA = Path("/workspace/SawitMVC-Combined-1716-RGB")
PROJECT = ROOT / "runs_combined1716"
RESULTS = ROOT / "results" / "combined1716"
LOGS = RESULTS / "logs"
PRED = RESULTS / "predictions"
OLD_ROOT = Path("/workspace/project-expertise")
OLD_RESULTS = OLD_ROOT / "results" / "new763"
OLD_DATA = Path("/workspace/SawitMVC-Depth-YOLO")
MAX_ACTIVE = 2
HEADROOM_MIB = 1024
BUDGET_MIB = {"yolo26l": 12_500, "rtdetr_l": 11_500, "rfdetr_l": 11_000}

JOBS = [
    {
        "arch": "yolo26l", "epochs": 60, "patience": 15,
        "batch": 4, "grad_accum": 4, "imgsz": 1280, "seed": 42,
        "name": "combined1716_yolo26l_rgb_s42_i1280",
    },
    {
        "arch": "rtdetr_l", "epochs": 60, "patience": 15,
        "batch": 4, "grad_accum": 4, "imgsz": 1280, "seed": 42,
        "name": "combined1716_rtdetr_l_rgb_s42_i1280",
    },
    {
        "arch": "rfdetr_l", "epochs": 60, "patience": 15,
        "batch": 4, "grad_accum": 4, "imgsz": 1280, "seed": 42,
        "name": "combined1716_rfdetr_l_rgb_s42_i1280",
    },
]


def gpu_memory() -> tuple[int, int]:
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True,
        ).strip().splitlines()[0]
        used, total = (int(v.strip()) for v in line.split(",", 1))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0, 24_564


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def write_manifest(manifest: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS / "campaign_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(RESULTS / "campaign_manifest.json")


def train_command(job: dict) -> list[str]:
    return [
        str(PYTHON), str(ROOT / "scripts" / "train_baseline_new763.py"),
        "--arch", job["arch"], "--data", str(DATA),
        "--project", str(PROJECT), "--name", job["name"],
        "--epochs", str(job["epochs"]), "--patience", str(job["patience"]),
        "--imgsz", str(job["imgsz"]), "--batch", str(job["batch"]),
        "--grad-accum", str(job["grad_accum"]), "--seed", str(job["seed"]),
    ]


def weight_path(job: dict) -> Path:
    run_dir = PROJECT / job["name"]
    if job["arch"] == "rfdetr_l":
        candidates = [run_dir / "checkpoint_best_ema.pth",
                      run_dir / "checkpoint_best_total.pth",
                      run_dir / "checkpoint_last.pth"]
    else:
        candidates = [run_dir / "weights" / "best.pt",
                      run_dir / "weights" / "last.pt"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no checkpoint for {job['name']}")


def eval_command(job: dict, weight: Path, output: Path) -> list[str]:
    kind = {"yolo26l": "yolo", "rtdetr_l": "rtdetr", "rfdetr_l": "rfdetr"}[job["arch"]]
    return [
        str(PYTHON), str(ROOT / "scripts" / "eval_new763_pycoco.py"),
        "--kind", kind, "--weights", str(weight), "--dataset", str(DATA),
        "--run-name", job["name"], "--imgsz", str(job["imgsz"]),
        "--out-json", str(output), "--pred-dir", str(PRED),
    ]


def finalize_previous_rf() -> None:
    """Finish the seed-42 RF-DETR run whose original runner was replaced."""
    name = "rfdetr_l_rgb_s42_i1280"
    run_dir = OLD_ROOT / "runs_new763" / name
    weight = run_dir / "checkpoint_best_ema.pth"
    output = OLD_RESULTS / f"{name}.json"
    log = OLD_RESULTS / "logs" / f"{name}.log"
    if output.is_file():
        return
    if not weight.is_file():
        print(f"previous RF-DETR has no final checkpoint: {weight}", flush=True)
        return
    cmd = [
        str(PYTHON), str(ROOT / "scripts" / "eval_new763_pycoco.py"),
        "--kind", "rfdetr", "--weights", str(weight),
        "--dataset", str(OLD_DATA), "--run-name", name, "--imgsz", "1280",
        "--out-json", str(output), "--pred-dir", str(OLD_RESULTS / "predictions"),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        code = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    manifest_path = OLD_RESULTS / "matrix_manifest.json"
    if manifest_path.is_file() and code == 0:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = manifest.get("jobs", {}).get(name, {})
        state.update({"status": "done", "weights": str(weight.resolve()),
                      "log": str(log.resolve()), "result": str(output.resolve())})
        manifest["jobs"][name] = state
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
        history = OLD_ROOT / "results" / "riwayat_epoch_new763"
        history.mkdir(parents=True, exist_ok=True)
        for filename in ("metrics.csv", "args.yaml", "baseline_args.json"):
            src = run_dir / filename
            if src.is_file():
                shutil.copy2(src, history / f"{name}__{filename}")
    print(f"previous RF-DETR evaluation exit={code}", flush=True)


def can_start(job: dict, active: dict[str, dict]) -> bool:
    if len(active) >= MAX_ACTIVE:
        return False
    arches = {item["job"]["arch"] for item in active.values()}
    if job["arch"] in arches:
        return False
    if arches | {job["arch"]} == {"yolo26l", "rtdetr_l"}:
        return False
    used, total = gpu_memory()
    return used + BUDGET_MIB[job["arch"]] + HEADROOM_MIB <= total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0,
                    help="active RF-DETR PID; new jobs wait for it to exit")
    ap.add_argument("--adopt-name", default="",
                    help="job name (from JOBS) already running outside this runner")
    ap.add_argument("--adopt-pid", type=int, default=0,
                    help="PID of the externally started job to adopt")
    args = ap.parse_args()
    if not (DATA / "data.yaml").is_file():
        raise FileNotFoundError(DATA / "data.yaml")
    PROJECT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    by_name = {job["name"]: job for job in JOBS}
    manifest = {"dataset": str(DATA), "protocol": "combined_1716_group_safe",
                "jobs": {job["name"]: {**job, "status": "pending"} for job in JOBS}}

    active: dict[str, dict] = {}
    if args.adopt_name:
        if args.adopt_name not in by_name:
            raise ValueError(f"adopt name not in matrix: {args.adopt_name}")
        if not pid_alive(args.adopt_pid):
            raise ValueError(f"adopt pid {args.adopt_pid} is not alive")
        log = LOGS / f"{args.adopt_name}.log"
        active[args.adopt_name] = {
            "job": by_name[args.adopt_name], "pid": args.adopt_pid,
            "external": True, "log": log,
        }
        manifest["jobs"][args.adopt_name]["status"] = "training"
        manifest["jobs"][args.adopt_name]["log"] = str(log.resolve())
        print(f"ADOPT {args.adopt_name} pid={args.adopt_pid}", flush=True)
    write_manifest(manifest)

    if args.wait_pid:
        print(f"WAIT pid={args.wait_pid} for current RF-DETR seed 42", flush=True)
        while pid_alive(args.wait_pid):
            time.sleep(30)
        # Wait for CUDA contexts and dataloader children to release memory.
        for _ in range(60):
            used, _ = gpu_memory()
            if used < 1000:
                break
            time.sleep(5)
        print(f"GPU released: {gpu_memory()[0]} MiB", flush=True)
        finalize_previous_rf()

    completed: set[str] = set()
    while len(completed) < len(JOBS):
        for name, item in list(active.items()):
            job = item["job"]
            log = item["log"]
            if item.get("external"):
                if pid_alive(item["pid"]):
                    continue
                code = 0
            else:
                proc: subprocess.Popen = item["proc"]
                code = proc.poll()
                if code is None:
                    continue
            if code == 0 and "Reducing to batch" not in log.read_text(
                encoding="utf-8", errors="replace"
            ):
                manifest["jobs"][name]["status"] = "trained"
                manifest["jobs"][name]["checkpoint"] = str(weight_path(job).resolve())
            else:
                manifest["jobs"][name]["status"] = "failed"
                manifest["jobs"][name]["error"] = (
                    f"training exit={code}; automatic batch reduction detected"
                )
            write_manifest(manifest)
            completed.add(name)
            del active[name]
            print(f"TRAIN DONE {name} exit={code}", flush=True)

        for job in JOBS:
            name = job["name"]
            if name in completed or name in active:
                continue
            if not can_start(job, active):
                continue
            log = LOGS / f"{name}.log"
            cmd = train_command(job)
            with log.open("a", encoding="utf-8") as fh:
                fh.write("\n$ " + " ".join(cmd) + "\n")
            out = log.open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
            active[name] = {"job": job, "proc": proc, "log": log, "stream": out}
            manifest["jobs"][name]["status"] = "training"
            manifest["jobs"][name]["log"] = str(log.resolve())
            write_manifest(manifest)
            print(f"START {name} active={len(active)}/2 vram={gpu_memory()[0]}/{gpu_memory()[1]}", flush=True)

        if len(completed) < len(JOBS):
            time.sleep(10)

    # Safe, serialized evaluation after all training has released the GPU.
    for job in JOBS:
        name = job["name"]
        if manifest["jobs"][name]["status"] != "trained":
            continue
        weight = weight_path(job)
        output = RESULTS / f"{name}.json"
        log = LOGS / f"{name}.log"
        cmd = eval_command(job, weight, output)
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n$ " + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            manifest["jobs"][name].update({"status": "done", "result": str(output.resolve())})
        else:
            manifest["jobs"][name].update({"status": "failed", "error": f"evaluation exit={proc.returncode}"})
        write_manifest(manifest)
        print(f"EVAL DONE {name} exit={proc.returncode}", flush=True)
    print(f"Manifest: {RESULTS / 'campaign_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
