"""Ukur pergeseran temporal antara SawitMVC-YOLO (953) dan SawitMVC-Depth (352).

Kedua dataset memakai tree ID yang sama untuk 352 pohon DAMIMAS, sehingga
sepanjang Fase 1-6 keduanya diperlakukan sebagai "pohon yang sama, satu dengan
depth satu tanpa". Probe ini menguji asumsi itu dan menemukannya SALAH: kedua
dataset adalah dua sesi akuisisi yang terpisah ~80 hari, sehingga tandan yang
difoto bukan tandan yang sama.

Konsekuensinya besar untuk seluruh Volume 2:
  - "kelangkaan B3/B4 di dataset 352" bukan artefak dataset yang lebih kecil,
    melainkan fase kematangan kebun yang berbeda pada pohon yang sama;
  - pretrain 953 -> finetune 352 bukan transfer di dalam satu domain, tapi
    transfer melintasi pergeseran domain temporal dengan distribusi kematangan
    yang nyaris terbalik;
  - label LOKALISASI ("ada tandan") bertahan melintasi jeda itu, label
    KEMATANGAN ("ini B3") tidak — persis pola AP50 agnostik 0,733 vs mAP50
    class-aware 0,45 yang selama ini tidak terjelaskan.

Read-only. Tidak menyentuh bobot atau hasil mana pun.

Usage:
    .venv/bin/python scripts/probe_pergeseran_temporal.py --out results/pergeseran_temporal.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

D953 = Path("/workspace/SawitMVC-YOLO")
D352 = Path("/workspace/SawitMVC-Depth")
NAMA = ["B1", "B2", "B3", "B4"]


def baca_yolo(p: Path) -> np.ndarray:
    baris = []
    for ln in p.read_text().splitlines():
        q = ln.split()
        if len(q) >= 5 and int(q[0]) >= 0:
            baris.append([int(q[0])] + [float(x) for x in q[1:5]])
    return np.array(baris, float) if baris else np.zeros((0, 5))


def tanggal_953() -> Counter:
    """Tanggal akuisisi per pohon, dari sidecar JSON dataset 953."""
    c = Counter()
    for p in sorted((D953 / "json").rglob("*.json")):
        try:
            c[json.loads(p.read_text())["metadata"]["date"]] += 1
        except Exception:
            pass
    return c


def tanggal_352() -> list[dict]:
    m = json.loads((D352 / "MERGE_VERIFICATION.json").read_text())
    return [{"arsip": a["file"], "pohon": a["treeCount"]} for a in m["sourceArchives"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pergeseran_temporal.json")
    args = ap.parse_args()

    lab953 = {p.stem: p for p in (D953 / "labels").rglob("*.txt")}
    lab352 = {p.stem: p for p in (D352 / "labels").glob("*.txt")}
    stems = sorted(set(lab953) & set(lab352))

    print(f"citra ber-ID sama di kedua dataset : {len(stems)}")
    print(f"  (dari {len(lab953)} citra 953 dan {len(lab352)} citra 352)")
    if not stems:
        print("FATAL: tidak ada irisan — struktur dataset berubah?")
        return 1

    hit = {"953": Counter(), "352": Counter()}
    luas = {"953": [], "352": []}
    n_kotak = {"953": 0, "352": 0}
    for s in stems:
        for tag, lab in (("953", lab953), ("352", lab352)):
            a = baca_yolo(lab[s])
            n_kotak[tag] += len(a)
            for r in a:
                hit[tag][int(r[0])] += 1
                luas[tag].append(r[3] * r[4])          # ternormalisasi, bebas resolusi

    hasil = {
        "n_citra_ber_id_sama": len(stems),
        "catatan_resolusi": "953 = 960x1280 potret, 352 = 1280x800 lanskap; "
                            "citra BUKAN berkas yang sama, jadi luas dilaporkan "
                            "ternormalisasi (bebas resolusi)",
        "akuisisi": {"953_per_tanggal": dict(sorted(tanggal_953().items())),
                     "352_arsip_sumber": tanggal_352()},
        "kotak": {},
    }

    print(f"\ntotal kotak pada {len(stems)} citra ber-ID sama:")
    for tag in ("953", "352"):
        tot = n_kotak[tag]
        dist = {NAMA[k]: hit[tag][k] for k in range(4)}
        pers = {NAMA[k]: round(100 * hit[tag][k] / max(tot, 1), 1) for k in range(4)}
        hasil["kotak"][tag] = {"total": tot, "per_kelas": dist, "persen": pers,
                               "luas_ternormalisasi_rata2": round(float(np.mean(luas[tag])), 6)}
        print(f"  {tag}: {tot:5d} kotak   " +
              "  ".join(f"{NAMA[k]}={dist[NAMA[k]]:4d} ({pers[NAMA[k]]:4.1f}%)" for k in range(4)))

    r = n_kotak["953"] / max(n_kotak["352"], 1)
    hasil["rasio_jumlah_kotak_953_per_352"] = round(r, 2)
    hasil["rasio_B3"] = round(hit["953"][2] / max(hit["352"][2], 1), 2)
    hasil["rasio_B4"] = round(hit["953"][3] / max(hit["352"][3], 1), 2)
    print(f"\nrasio jumlah kotak 953/352 = {r:.2f}x")
    print(f"rasio B3 = {hasil['rasio_B3']:.1f}x   rasio B4 = {hasil['rasio_B4']:.1f}x")
    print("\ntanggal akuisisi 953:", ", ".join(sorted(tanggal_953())[:1]),
          "s.d.", sorted(tanggal_953())[-1])
    print("arsip sumber 352    :", ", ".join(a["arsip"][:34] for a in tanggal_352()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hasil, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
