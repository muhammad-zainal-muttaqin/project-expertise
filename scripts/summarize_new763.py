"""Aggregate the completed 3-architecture x 3-seed baseline matrix."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev


ARCH_FROM_KIND = {"yolo": "yolo26l", "rtdetr": "rtdetr_l", "rfdetr": "rfdetr_l"}


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": round(mean(values), 6) if values else None,
            "std": round(stdev(values), 6) if len(values) > 1 else 0.0 if values else None,
            "min": round(min(values), 6) if values else None,
            "max": round(max(values), 6) if values else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/workspace/project-expertise/results/new763"))
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/project-expertise/results/new763_summary.json"))
    args = ap.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("*_rgb_s*_i1280.json")):
        result = json.loads(path.read_text())
        match = re.search(r"_s(\d+)_i(\d+)$", result["run_name"])
        if not match:
            continue
        row = {"run_name": result["run_name"], "kind": result.get("kind"),
               "arch": ARCH_FROM_KIND.get(result.get("kind"), result.get("kind")),
               "seed": int(match.group(1)), "imgsz": int(match.group(2)),
               "result": str(path.resolve())}
        for split, metrics in result.get("splits", {}).items():
            row[f"{split}_mAP50"] = metrics.get("mAP50")
            row[f"{split}_mAP50_95"] = metrics.get("mAP50_95")
        rows.append(row)

    grouped = {}
    for row in rows:
        group = grouped.setdefault(row["arch"], {})
        for metric in ("val_mAP50", "val_mAP50_95", "test_mAP50", "test_mAP50_95"):
            value = row.get(metric)
            if value is not None:
                group.setdefault(metric, []).append(float(value))
    aggregate = {arch: {metric: stats(values) for metric, values in metrics.items()}
                 for arch, metrics in grouped.items()}
    best_val = max(rows, key=lambda r: r.get("val_mAP50", float("-inf")), default=None)
    best_test = max(rows, key=lambda r: r.get("test_mAP50", float("-inf")), default=None)
    output = {"n_completed": len(rows), "runs": rows, "by_architecture": aggregate,
              "best_by_validation_mAP50": best_val,
              "best_by_test_mAP50": best_test}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    csv_path = args.output.with_suffix(".csv")
    fields = ["run_name", "arch", "seed", "val_mAP50", "val_mAP50_95",
              "test_mAP50", "test_mAP50_95", "result"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    print(f"completed={len(rows)}")
    print(f"json={args.output}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
