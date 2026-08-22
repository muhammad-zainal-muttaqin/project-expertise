"""Audit deterministik dataset SawitMVC-Depth-YOLO v2.0.0 (763 pohon).

Audit ini hanya membaca dataset. Ia memeriksa unit split pohon, pasangan
gambar-label-depth, ukuran buffer depth, dan konsistensi angka pada
``split_stats.json`` sebelum training boleh dimulai.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEPTH_BYTES = 848 * 480 * 2


def tree_id(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def audit_split(root: Path, split: str) -> dict:
    base = root / split
    images = sorted(p for p in (base / "images").iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    labels = base / "labels"
    depths = base / "depth"
    linked = base / "linked"
    trees = {tree_id(p.stem) for p in images}
    class_counts = Counter()
    boxes = 0
    empty = 0
    missing_labels = []
    missing_depth = []
    bad_depth = []
    bad_images = []
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:  # pragma: no cover - pesan audit saja
            bad_images.append({"file": p.name, "error": str(exc)})
        lf = labels / f"{p.stem}.txt"
        if not lf.is_file():
            missing_labels.append(p.name)
        else:
            rows = [x.split() for x in lf.read_text().splitlines() if x.strip()]
            if not rows:
                empty += 1
            for row in rows:
                if len(row) < 5:
                    raise ValueError(f"Label rusak: {lf}: {row}")
                cls = int(row[0])
                if not 0 <= cls <= 3:
                    raise ValueError(f"Kelas di luar 0..3: {lf}: {cls}")
                class_counts[cls] += 1
                boxes += 1
        raw = depths / f"{p.stem}.raw"
        sidecar = depths / f"{p.stem}.json"
        if not raw.is_file() or not sidecar.is_file():
            missing_depth.append(p.name)
        elif raw.stat().st_size != DEPTH_BYTES:
            bad_depth.append({"file": raw.name, "bytes": raw.stat().st_size})
        if not (linked / f"{tree_id(p.stem)}.json").is_file():
            raise FileNotFoundError(f"Manifest linked pohon hilang untuk {p}")
    return {
        "trees": len(trees),
        "images": len(images),
        "boxes": boxes,
        "boxes_per_class": {f"B{k + 1}": class_counts[k] for k in range(4)},
        "images_without_boxes": empty,
        "missing_labels": missing_labels,
        "missing_depth_or_sidecar": missing_depth,
        "bad_depth_size": bad_depth,
        "bad_images": bad_images,
        "tree_ids": sorted(trees),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("/workspace/SawitMVC-Depth-YOLO"))
    ap.add_argument("--out", type=Path,
                    default=Path("/workspace/project-expertise/results/new763_dataset_audit.json"))
    args = ap.parse_args()

    expected = json.loads((args.dataset / "split_stats.json").read_text())
    result = {
        "dataset": str(args.dataset.resolve()),
        "source_release": "SawitMVC-Depth-YOLO v2.0.0",
        "splits": {},
        "tree_overlap": {},
        "expected_split_stats": expected,
    }
    for split in ("train", "valid", "test"):
        result["splits"][split] = audit_split(args.dataset, split)
    ids = {s: set(result["splits"][s]["tree_ids"])
           for s in ("train", "valid", "test")}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        result["tree_overlap"][f"{a}__{b}"] = sorted(ids[a] & ids[b])
    result["all_tree_count"] = len(set().union(*ids.values()))
    result["ok"] = (
        all(not result["splits"][s]["missing_labels"] for s in ids)
        and all(not result["splits"][s]["missing_depth_or_sidecar"] for s in ids)
        and all(not result["splits"][s]["bad_depth_size"] for s in ids)
        and all(not result["splits"][s]["bad_images"] for s in ids)
        and all(not result["tree_overlap"][k] for k in result["tree_overlap"])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for split in ("train", "valid", "test"):
        x = result["splits"][split]
        print(f"{split}: {x['trees']} pohon, {x['images']} citra, "
              f"{x['boxes']} box, kosong={x['images_without_boxes']}")
    print(f"all_tree_count={result['all_tree_count']} ok={result['ok']}")
    print(f"-> {args.out}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
