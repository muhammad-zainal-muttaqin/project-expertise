"""Menyusun dump prediksi detektor 953 dan 763 pada partisi uji 1716 (V2-E-050g).

Korpus 1716 menyalin citra dari kedua korpus sumber secara utuh
(`build_combined_rgb_dataset.py`, `shutil.copy2`) dengan awalan `SAWIT_` atau
`DEPTH_`. Prediksi pada partisi uji 1716 karena itu dapat disusun dari dump yang
sudah ada, ditambah inferensi baru untuk citra yang belum pernah diprediksi.

    953 -> 1716 (257 pohon, 1.052 citra)
        SAWIT_  : dump dalam domain 953
        DEPTH_  : 66 pohon dari dump 953 -> 763, 50 pohon dari inferensi baru
    763 -> 1716 (207 pohon, 828 citra)
        SAWIT_  : dump 763 -> 953
        DEPTH_  : 66 pohon dari dump dalam domain 763
        50 pohon DEPTH lain dikeluarkan karena citranya berada pada partisi
        latih atau validasi 763.

Pemakaian:
    python scripts/gabung_dump_1716.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kalibrasi_koefisien_pencacahan import muat_gt_763
from kalibrasi_pencacahan_perkorpus import DETEKTOR, pohon_bocor_763

PRED = "results/cross_eval/predictions"


def pohon(kunci: str) -> str:
    return kunci.rsplit("_", 1)[0]


def ambil(path: Path, awalan: str, saring: set[str] | None = None) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        f"{awalan}{k}": data[k]
        for k in data.files
        if saring is None or f"{awalan}{pohon(k)}" in saring
    }


def susun(bagian: list[tuple[Path, str, set[str] | None]]) -> tuple[dict[str, np.ndarray], list[dict]]:
    hasil: dict[str, np.ndarray] = {}
    catatan = []
    for path, awalan, saring in bagian:
        isi = ambil(path, awalan, saring)
        tumpang = set(isi) & set(hasil)
        if tumpang:
            raise SystemExit(f"kunci ganda dari {path.name}: {sorted(tumpang)[:3]}")
        hasil.update(isi)
        catatan.append({
            "sumber": path.as_posix(),
            "awalan": awalan,
            "citra": len(isi),
            "pohon": len({pohon(k) for k in isi}),
        })
    return hasil, catatan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-root", default="D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO")
    ap.add_argument("--project-root", default=".")
    arg = ap.parse_args()

    akar = Path(arg.project_root).resolve()
    gt763 = muat_gt_763(Path(arg.depth_root))
    bocor = pohon_bocor_763(akar, gt763)
    manifest = {"eksperimen": "V2-E-050g", "pohon_dikeluarkan_763": sorted(bocor), "dump": {}}

    for slug, _ in DETEKTOR:
        acuan = np.load(
            akar / f"results/combined1716/predictions/combined1716_{slug}_rgb_s42_i1280__test.npz",
            allow_pickle=True,
        ).files
        kunci_acuan = set(acuan)
        depth_uji = {pohon(k) for k in kunci_acuan if k.startswith("DEPTH_")}

        rencana = {
            f"v2repro953_{slug}__on_1716__test.npz": (
                [
                    (akar / f"results/pred_{slug}_v2repro_953_test.npz", "SAWIT_", None),
                    (akar / f"{PRED}/v2repro953_{slug}__on_763__test.npz", "DEPTH_", depth_uji),
                    (akar / f"{PRED}/v2repro953_{slug}__on_1716_depth50__test.npz", "DEPTH_", None),
                ],
                kunci_acuan,
            ),
            f"new763_{slug}__on_1716__test.npz": (
                [
                    (akar / f"{PRED}/new763_{slug}__on_953__test.npz", "SAWIT_", None),
                    (akar / f"results/new763/predictions/{slug}_rgb_s42_i1280__test.npz", "DEPTH_", depth_uji),
                ],
                {k for k in kunci_acuan if pohon(k) not in bocor},
            ),
        }
        for nama, (bagian, harapan) in rencana.items():
            hasil, catatan = susun(bagian)
            if set(hasil) != harapan:
                kurang, lebih = harapan - set(hasil), set(hasil) - harapan
                raise SystemExit(f"{nama}: kurang {len(kurang)}, lebih {len(lebih)}")
            np.savez_compressed(akar / PRED / nama, **hasil)
            for c in catatan:
                c["sumber"] = str(Path(c["sumber"]).relative_to(akar)).replace("\\", "/")
            manifest["dump"][nama] = {
                "citra": len(hasil),
                "pohon": len({pohon(k) for k in hasil}),
                "bagian": catatan,
            }
            print(f"{nama}: {len(hasil)} citra, {manifest['dump'][nama]['pohon']} pohon")

    (akar / PRED / "gabungan_1716_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
