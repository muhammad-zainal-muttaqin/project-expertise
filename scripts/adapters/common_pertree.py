"""Helper bersama untuk adaptor detektor -> JSON per-pohon.

Skema output mengikuti kontrak di docs/SCHEMA-PERTREE.md, dicocokkan dari
predictions/y26mv2_per_tree/*.json di Baseline-SawitMVC:

    {
      "tree_name": str,
      "split": "train"|"val"|"test",
      "detector": str,
      "images": {
        "side_<n>": {
          "side_index": int,   # 0-based
          "annotations": [
            {"class_name": "B1".."B4", "bbox_yolo": [cx,cy,w,h], "conf": float}
          ]
        }
      }
    }
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["B1", "B2", "B3", "B4"]  # class_id 0-3, sama dengan data_rgb.yaml


def parse_tree_and_side(image_path: Path) -> tuple[str, int]:
    """'DAMIMAS_A21B_0001_1.jpg' -> ('DAMIMAS_A21B_0001', 1)"""
    stem = image_path.stem
    tree_name, _, side_str = stem.rpartition("_")
    return tree_name, int(side_str)


def group_images_by_tree(image_dir: Path, stems: list[str] | None = None) -> dict[str, list[Path]]:
    """Kelompokkan file citra per pohon. `stems` membatasi ke daftar nama file
    (tanpa ekstensi) tertentu, mis. dari train.txt/val.txt/test.txt."""
    trees: dict[str, list[Path]] = defaultdict(list)
    for img in sorted(image_dir.glob("*.jpg")):
        if stems is not None and img.stem not in stems:
            continue
        tree_name, _ = parse_tree_and_side(img)
        trees[tree_name].append(img)
    return trees


def load_split_stems(split_file: Path) -> set[str]:
    """train.txt/val.txt/test.txt di SawitMVC berisi path relatif/absolut ke
    citra, satu per baris. Kembalikan set of stems (tanpa ekstensi)."""
    stems = set()
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stems.add(Path(line).stem)
    return stems


def write_pertree_json(
    out_dir: Path,
    tree_name: str,
    split: str,
    detector: str,
    images: dict[str, dict],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tree_name": tree_name,
        "split": split,
        "detector": detector,
        "images": images,
    }
    out_path = out_dir / f"{tree_name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
