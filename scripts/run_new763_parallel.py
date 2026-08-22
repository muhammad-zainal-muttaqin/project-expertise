"""Memory-aware parallel runner for the new763 RGB baseline matrix.

This runner is deliberately conservative for a single 24 GB GPU:

* at most two training processes are active;
* YOLO26-L and RT-DETR-L are never paired;
* a 1 GiB headroom is required before a second process starts;
* a run that triggers Ultralytics' automatic batch reduction is invalidated;
* evaluation is serialized after all training, so it cannot compete with a
  training process for VRAM.

It can take over an already-running RF-DETR process through ``--external-pid``.
The external process is treated as one occupied GPU slot and is never killed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from run_new763_matrix import (
    ARCH_KIND,
    DATA,
    PRED,
    PROJECT,
    RESULTS,
    best_weights,
    copy_history,
    jobs,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MANIFEST = RESULTS / "matrix_manifest.json"

# Measured peaks from completed 1280px runs were approximately 12.3G (YOLO),
# 11.3G (RT-DETR), and 10.6 GiB currently resident for RF-DETR. The budgets
# below include a small cushion around those observations.
BUDGET_MIB = {"yolo26l": 12_500, "rtdetr_l": 11_500, "rfdetr_l": 11_000}
HEADROOM_MIB = 1_024
MAX_ACTIVE = 2
POLL_SECONDS = 10


def gpu_memory() -> tuple[int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(",", 1))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0, 24_564


def write_manifest(manifest: dict) -> None:
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)


def append_log(log: Path, cmd: list[str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")


def train_cmd(job: dict) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "train_baseline_new763.py"),
        "--arch",
        job["arch"],
        "--data",
        str(DATA),
        "--project",
        str(PROJECT),
        "--name",
        job["name"],
        "--epochs",
        str(job["epochs"]),
        "--patience",
        str(job["patience"]),
        "--imgsz",
        str(job["imgsz"]),
        "--batch",
        str(job["batch"]),
        "--grad-accum",
        str(job["grad_accum"]),
        "--seed",
        str(job["seed"]),
    ]


def eval_cmd(job: dict, weight: Path, out_json: Path) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "eval_new763_pycoco.py"),
        "--kind",
        ARCH_KIND[job["arch"]],
        "--weights",
        str(weight),
        "--dataset",
        str(DATA),
        "--run-name",
        job["name"],
        "--imgsz",
        str(job["imgsz"]),
        "--out-json",
        str(out_json),
        "--pred-dir",
        str(PRED),
    ]


def active_arches(active: dict[str, dict]) -> list[str]:
    return [item["job"]["arch"] for item in active.values()]


def can_start(job: dict, active: dict[str, dict]) -> tuple[bool, str]:
    if len(active) >= MAX_ACTIVE:
        return False, "two slots occupied"
    arches = active_arches(active)
    if job["arch"] in arches:
        return False, "same architecture already active"
    # YOLO+RT is intentionally disallowed even though the nominal sum is near
    # the card limit; it leaves too little room for allocator fragmentation.
    if {job["arch"], *arches} >= {"yolo26l", "rtdetr_l"} and {
        job["arch"], *arches
    } == {"yolo26l", "rtdetr_l"}:
        return False, "YOLO+RT-DETR pair is too close to the VRAM ceiling"
    used, total = gpu_memory()
    needed = BUDGET_MIB[job["arch"]] + HEADROOM_MIB
    if used + needed > total:
        return False, f"VRAM guard: {used}+{needed}>{total} MiB"
    return True, "ok"


def mark_failed(manifest: dict, job: dict, error: str) -> None:
    manifest["jobs"][job["name"]] = {
        **job,
        "status": "failed",
        "error": error,
        "log": str((RESULTS / "logs" / f"{job['name']}.log").resolve()),
    }
    write_manifest(manifest)


def mark_trained(manifest: dict, job: dict) -> bool:
    log = RESULTS / "logs" / f"{job['name']}.log"
    if log.is_file() and "Reducing to batch" in log.read_text(
        encoding="utf-8", errors="replace"
    ):
        mark_failed(manifest, job, "Ultralytics reduced the batch automatically; run invalidated")
        return False
    try:
        weight = best_weights(job)
    except FileNotFoundError as exc:
        mark_failed(manifest, job, f"training exited but no checkpoint found: {exc}")
        return False
    run_dir = PROJECT / job["name"]
    copy_history(run_dir, job["name"])
    manifest["jobs"][job["name"]] = {
        **job,
        "status": "trained",
        "weights": str(weight.resolve()),
        "log": str(log.resolve()),
    }
    write_manifest(manifest)
    return True


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run_serial_evaluations(manifest: dict, all_jobs: list[dict]) -> None:
    for job in all_jobs:
        state = manifest["jobs"].get(job["name"], {})
        if state.get("status") == "done":
            continue
        if state.get("status") != "trained":
            continue
        log = RESULTS / "logs" / f"{job['name']}.log"
        out_json = RESULTS / f"{job['name']}.json"
        weight = best_weights(job)
        cmd = eval_cmd(job, weight, out_json)
        append_log(log, cmd)
        print(f"EVAL {job['name']}", flush=True)
        with log.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode:
            mark_failed(manifest, job, f"evaluation exited with code {proc.returncode}")
            continue
        manifest["jobs"][job["name"]].update(
            {"status": "done", "result": str(out_json.resolve())}
        )
        write_manifest(manifest)
        print(f"SELESAI {job['name']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    ap.add_argument(
        "--external-pid",
        type=int,
        default=0,
        help="PID of a training process already running; it occupies one slot",
    )
    ap.add_argument(
        "--external-job",
        default="rfdetr_l_rgb_s42_i1280",
        help="job name corresponding to --external-pid",
    )
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    all_jobs = jobs(args.seeds)
    by_name = {job["name"]: job for job in all_jobs}
    if args.external_job not in by_name:
        raise ValueError(f"external job not in matrix: {args.external_job}")

    active: dict[str, dict] = {}
    external_name = args.external_job if args.external_pid and pid_alive(args.external_pid) else ""
    if external_name:
        active[external_name] = {
            "job": by_name[external_name],
            "pid": args.external_pid,
            "external": True,
        }
        print(
            f"ADOPT {external_name} pid={args.external_pid}; "
            "training process will not be interrupted",
            flush=True,
        )

    started: set[str] = set(active)
    while True:
        # Reap child training processes.
        for name, item in list(active.items()):
            if item.get("external"):
                if pid_alive(item["pid"]):
                    continue
                print(f"EXTERNAL DONE {name}", flush=True)
                if manifest["jobs"].get(name, {}).get("status") == "training":
                    mark_trained(manifest, item["job"])
                del active[name]
                continue
            proc: subprocess.Popen = item["proc"]
            code = proc.poll()
            if code is None:
                continue
            print(f"TRAIN EXIT {name} code={code}", flush=True)
            if code == 0:
                mark_trained(manifest, item["job"])
            else:
                mark_failed(manifest, item["job"], f"training exited with code {code}")
            del active[name]

        # Start as many safe jobs as the current VRAM budget allows. RF jobs
        # are held while the adopted RF job is active; this avoids two unknown
        # RF allocator peaks competing during the handoff.
        made_progress = True
        while made_progress and len(active) < MAX_ACTIVE:
            made_progress = False
            for job in all_jobs:
                name = job["name"]
                state = manifest["jobs"].get(name, {})
                if name in started or state.get("status") in {"done", "trained", "failed"}:
                    continue
                if state.get("status") == "training":
                    continue
                if external_name in active and job["arch"] == "rfdetr_l":
                    continue
                ok, reason = can_start(job, active)
                if not ok:
                    continue
                log = RESULTS / "logs" / f"{name}.log"
                cmd = train_cmd(job)
                append_log(log, cmd)
                print(
                    f"START {name} active={len(active)+1}/{MAX_ACTIVE} "
                    f"vram={gpu_memory()[0]}/{gpu_memory()[1]} MiB",
                    flush=True,
                )
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=log.open("a", encoding="utf-8"),
                    stderr=subprocess.STDOUT,
                )
                active[name] = {"job": job, "proc": proc, "external": False}
                started.add(name)
                manifest["jobs"][name] = {
                    **job,
                    "status": "training",
                    "log": str(log.resolve()),
                }
                write_manifest(manifest)
                made_progress = True
                break

        pending = [
            job
            for job in all_jobs
            if manifest["jobs"].get(job["name"], {}).get("status")
            not in {"done", "trained", "failed"}
            and job["name"] not in started
        ]
        if not active and not pending:
            break
        if active:
            time.sleep(POLL_SECONDS)
        elif pending:
            # A pending job may have been blocked only by a stale GPU
            # allocation; wait briefly and let the guard re-check it.
            time.sleep(POLL_SECONDS)

    run_serial_evaluations(manifest, all_jobs)
    print(f"Manifest: {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
