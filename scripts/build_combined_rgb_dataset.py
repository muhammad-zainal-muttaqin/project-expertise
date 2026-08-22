"""Build a provenance-preserving RGB union of SawitMVC and SawitMVC-Depth.

The two source datasets contain 352 common tree IDs. Every output filename is
prefixed by its source, so all 3,992 + 3,052 RGB views are retained. For split
integrity, a common original tree is assigned to one split as a group using
the SawitMVC split as the precedence rule; this prevents its two capture
records from landing in different train/validation/test splits.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


NAMES = ["B1", "B2", "B3", "B4"]


def tree_id(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def normalized_split(split: str) -> str:
    return "valid" if split in {"val", "valid"} else split


def collect_source(
    root: Path, source: str, layout: dict[str, tuple[str, str]]
) -> tuple[list[dict], dict[str, str]]:
    records: list[dict] = []
    tree_splits: dict[str, str] = {}
    for raw_split, (image_rel, label_rel) in layout.items():
        split = normalized_split(raw_split)
        image_dir = root / image_rel
        label_dir = root / label_rel
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"missing label for {image}: {label}")
            tid = tree_id(image.stem)
            previous = tree_splits.setdefault(tid, split)
            if previous != split:
                raise ValueError(f"tree crosses source splits: {source} {tid}: {previous}, {split}")
            records.append({
                "source": source,
                "source_tree": tid,
                "source_split": split,
                "image": image,
                "label": label,
            })
    return records, tree_splits


def class_counts(records: list[dict]) -> Counter:
    out: Counter = Counter()
    for rec in records:
        for line in rec["label"].read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                out[NAMES[int(fields[0])]] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sawit", type=Path, default=Path("/workspace/SawitMVC-YOLO"))
    ap.add_argument("--depth-rgb", type=Path,
                    default=Path("/workspace/SawitMVC-Depth-YOLO-RGB"))
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/SawitMVC-Combined-1716-RGB"))
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    sawit_records, sawit_splits = collect_source(
        args.sawit.resolve(), "sawitmvc",
        {
            "train": ("images/train", "labels/train"),
            "valid": ("images/val", "labels/val"),
            "test": ("images/test", "labels/test"),
        },
    )
    depth_records, depth_splits = collect_source(
        args.depth_rgb.resolve(), "depth_rgb",
        {
            "train": ("train/images", "train/labels"),
            "valid": ("valid/images", "valid/labels"),
            "test": ("test/images", "test/labels"),
        },
    )
    all_records = sawit_records + depth_records
    overlap = sorted(set(sawit_splits) & set(depth_splits))

    # The old SawitMVC split is the stable anchor for common trees. Depth-only
    # trees keep their v2 split. Thus duplicate captures are never split apart.
    group_split: dict[str, str] = dict(sawit_splits)
    for tid, split in depth_splits.items():
        group_split.setdefault(tid, split)

    split_counts = Counter()
    source_counts = Counter()
    source_split_counts = Counter()
    class_by_split: dict[str, Counter] = defaultdict(Counter)
    manifest_records = []
    for rec in all_records:
        split = group_split[rec["source_tree"]]
        source_prefix = "SAWIT" if rec["source"] == "sawitmvc" else "DEPTH"
        output_stem = f"{source_prefix}_{rec['image'].stem}"
        image_out = out / split / "images" / f"{output_stem}{rec['image'].suffix.lower()}"
        label_out = out / split / "labels" / f"{output_stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec["image"], image_out)
        shutil.copy2(rec["label"], label_out)
        n_boxes = 0
        for line in rec["label"].read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                class_by_split[split][NAMES[int(fields[0])]] += 1
                n_boxes += 1
        split_counts[split] += 1
        source_counts[rec["source"]] += 1
        source_split_counts[f"{rec['source']}:{rec['source_split']}->{split}"] += 1
        manifest_records.append({
            "output_image": str(image_out.relative_to(out)),
            "output_label": str(label_out.relative_to(out)),
            "source": rec["source"],
            "source_tree": rec["source_tree"],
            "group_tree": rec["source_tree"],
            "source_split": rec["source_split"],
            "combined_split": split,
            "boxes": n_boxes,
        })

    (out / "data.yaml").write_text(
        "path: " + str(out) + "\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 4\n"
        "names:\n"
        "  0: B1\n  1: B2\n  2: B3\n  3: B4\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset": "SawitMVC-YOLO + SawitMVC-Depth-YOLO RGB",
        "sources": {
            "sawitmvc": str(args.sawit.resolve()),
            "depth_rgb": str(args.depth_rgb.resolve()),
        },
        "output": str(out),
        "classes": NAMES,
        "source_trees": {"sawitmvc": len(sawit_splits), "depth_rgb": len(depth_splits)},
        "overlap_tree_ids": len(overlap),
        "union_tree_groups": len(group_split),
        "source_records": dict(source_counts),
        "combined_images": dict(split_counts),
        "source_split_to_combined_split": dict(source_split_counts),
        "class_boxes_by_combined_split": {
            split: dict(counts) for split, counts in sorted(class_by_split.items())
        },
        "split_policy": "SawitMVC split takes precedence for 352 common tree IDs; depth-only IDs keep depth v2 split",
        "filename_policy": "SAWIT_ and DEPTH_ prefixes retain every source view and prevent filename collisions",
        "records": manifest_records,
    }
    (out / "combined_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(out),
        "sawitmvc_trees": len(sawit_splits),
        "depth_trees": len(depth_splits),
        "overlap_trees": len(overlap),
        "union_tree_groups": len(group_split),
        "images_by_split": dict(split_counts),
        "source_split_to_combined_split": dict(source_split_counts),
        "boxes_by_split": {split: dict(c) for split, c in sorted(class_by_split.items())},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
