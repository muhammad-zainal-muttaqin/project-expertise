"""Runner pasca-training RF-DETR: konversi, lock VAL, lalu infer split.

Proses ini boleh dijalankan saat training masih aktif. Ia menunggu PID dengan
verifikasi command line, sehingga tidak salah menganggap PID lain sebagai
training RF-DETR. Seleksi dan aturan akses split tetap berada di
``seleksi_rfdetr_damimas.py``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def commandline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def tunggu(pid: int, interval: int) -> None:
    while True:
        cmd = commandline(pid)
        if not cmd:
            return
        if "train_rfdetr_damimas.py" not in cmd:
            raise RuntimeError(f"PID {pid} berubah menjadi proses lain: {cmd}")
        print(f"training masih aktif PID={pid}; tunggu {interval}s", flush=True)
        time.sleep(interval)


def jalankan(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(args) + "\n")
        f.flush()
        p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT,
                           check=False)
    if p.returncode:
        raise SystemExit(f"perintah gagal ({p.returncode}), lihat {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tag", default="rfdetr_l_damimas_selected")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--log", type=Path,
                    default=Path("pipeline-pertandan/logs_ringkas/rfdetr_postprocess.log"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = str(root / ".venv" / "bin" / "python")
    converter = str(root / "scripts" / "convert_rfdetr_checkpoints.py")
    selector = str(root / "scripts" / "seleksi_rfdetr_damimas.py")
    best_ema = args.run / "checkpoint_best_ema.pth"
    best_regular = args.run / "checkpoint_best_regular.pth"
    if not best_ema.is_file() or not best_regular.is_file():
        raise FileNotFoundError("checkpoint best EMA/regular belum ada")

    tunggu(args.pid, max(1, args.interval))
    jalankan([python, converter, "--run", str(args.run),
              "--template", str(best_ema), "--overwrite"], args.log)

    candidates = [("best_ema", best_ema), ("best_regular", best_regular)]
    for p in sorted(args.run.glob("*_infer.pth")):
        candidates.append((p.stem.replace("_infer", ""), p))
    total = args.run / "checkpoint_best_total.pth"
    if total.is_file():
        candidates.append(("best_total", total))
    unique = {}
    for name, path in candidates:
        unique[name] = path
    cmd = [python, selector, "--tag", args.tag,
           "--output", str(root / "pipeline-pertandan" / "results" /
                            f"{args.tag}.json")]
    for name, path in unique.items():
        cmd += ["--checkpoint", f"{name}={path}"]
    jalankan(cmd, args.log)
    print("RF-DETR postprocess selesai", flush=True)


if __name__ == "__main__":
    main()
