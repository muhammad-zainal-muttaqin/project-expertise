"""Nilai kepala linker yang masing-masing dikunci oleh objektif VAL berbeda.

Model, graf TEST, dan konfigurasi berasal dari run ``linker_global_damimas``.
Tidak ada pencarian di TEST: setiap kepala mengambil argmax/argmin yang sudah
tersimpan di ``terbaik_per_metrik_val``, lalu TEST hanya dievaluasi sekali.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib


SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import linker_global_damimas as LG  # noqa: E402


def kunci_config(q):
    return (q["sumber"], q["assembler"], q["metode"], q["max_mode"], q["ambang"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasil", type=Path,
                    default=SUB / "results" / "damimas_linker_global.json")
    ap.add_argument("--model", type=Path,
                    default=SUB / "runs" / "linker_global_damimas" / "model.joblib")
    ap.add_argument("--cache-test", type=Path,
                    default=SUB / "results" / "cache_linker_damimas_damimas_test.joblib")
    args = ap.parse_args()

    hasil = json.loads(args.hasil.read_text())
    bundle = joblib.load(args.model)
    graphs = joblib.load(args.cache_test)["graf"]
    kandidat = {"utility": hasil["terpilih_di_val"],
                **hasil["terbaik_per_metrik_val"]}

    perlu_model = sorted({n for q in kandidat.values() for n in q["bobot_skor"]})
    skor_model = {n: LG.score_model(bundle["models"][n], graphs) for n in perlu_model}
    cache = {}
    kepala = {}
    for nama, q in kandidat.items():
        key = kunci_config(q)
        if key not in cache:
            score = LG.gabung_score(q["bobot_skor"], skor_model)
            cache[key] = LG.nilai(graphs, score, q["assembler"], q["ambang"],
                                  q["metode"], q["max_mode"])
        kepala[nama] = {"aturan_seleksi": f"terbaik {nama} di VAL",
                        "config": q, "test": cache[key]}

    # Pemeriksaan bahwa evaluasi ulang kepala utility identik dengan run asal.
    for k, v in hasil["test"].items():
        if isinstance(v, (int, float)) and abs(v - kepala["utility"]["test"][k]) > 1e-10:
            raise RuntimeError(f"Reproduksi utility berbeda pada {k}")
    hasil["kepala_terkunci_per_objektif"] = kepala
    args.hasil.write_text(json.dumps(LG.serial(hasil), indent=2, ensure_ascii=False))
    for nama, q in kepala.items():
        m = q["test"]
        print(f"{nama:26s} F1={m['f1']:.4f} coverage={m['cakupan_atas_semua']:.4f} "
              f"poolMAE={m['mae_jumlah_pool']:.3f}")
    print(f"-> {args.hasil}")


if __name__ == "__main__":
    main()
