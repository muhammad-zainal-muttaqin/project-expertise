"""Ringkasan tabel dari hasil kalibrasi koefisien pengali (V2-E-050)."""
from __future__ import annotations

import json
from pathlib import Path

SUMBER = Path("results/counting_koefisien_2026-09-16/koefisien_pencacahan.json")
CLASSES = ["B1", "B2", "B3", "B4"]
NAMA_METODE = {
    "naif": "Naif",
    "global": "k global",
    "k_perkelas": "k per kelas",
    "k_tau_perkelas": "k + tau per kelas",
}


def koma(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def metrik(baris: dict, basis: str) -> dict:
    if basis == "irisan" and baris.get("uji_metrik_irisan763"):
        return baris["uji_metrik_irisan763"]
    return baris["uji_metrik"]


def main() -> None:
    data = json.loads(SUMBER.read_text(encoding="utf-8"))
    baris = data["baris"]

    for korpus_uji, basis in [("953-test", "penuh"), ("763-test", "irisan")]:
        subset = [b for b in baris if b["uji"] == korpus_uji]
        subset.sort(key=lambda b: metrik(b, basis)["mae_makro"])
        print(f"\n### UJI {korpus_uji} (basis {basis}, n={metrik(subset[0], basis)['n_pohon']})")
        print("rank | detektor | latih | kalibrasi | metode | MAE | RMSE | Acc±1 | MAE total | bias|abs| | mAP50")
        for i, b in enumerate(subset[:8], 1):
            m = metrik(b, basis)
            print(
                f"{i} | {b['detektor']} | {b['korpus_latih']} | {b['kalibrasi']}"
                f"{' (silang)' if b['silang'] else ''} | {NAMA_METODE[b['metode']]} | "
                f"{koma(m['mae_makro'])} | {koma(m['rmse_makro'])} | {koma(m['acc_pm1_makro'])} | "
                f"{koma(m['mae_total_pohon'])} | {koma(m['bias_abs_makro'])} | "
                f"{koma(b['map50_uji'],4) if b['map50_uji'] else '-'}"
            )

    print("\n### KOEFISIEN konfigurasi unggulan")
    unggulan = []
    for korpus_uji, basis in [("953-test", "penuh"), ("763-test", "irisan")]:
        subset = [b for b in baris if b["uji"] == korpus_uji and b["metode"] != "naif"]
        subset.sort(key=lambda b: metrik(b, basis)["mae_makro"])
        unggulan.extend(subset[:3])
    for b in unggulan:
        basis = "irisan" if b["uji"] == "763-test" else "penuh"
        m = metrik(b, basis)
        k = " | ".join(koma(v, 2) for v in b["k"])
        t = " | ".join(koma(v, 2) for v in b["tau"])
        bias = " | ".join(koma(m["per_kelas"][c]["bias"], 3) for c in CLASSES)
        mae_c = " | ".join(koma(m["per_kelas"][c]["mae"], 3) for c in CLASSES)
        print(
            f"{b['detektor']} | {b['korpus_latih']} | {b['kalibrasi']} -> {b['uji']} | "
            f"{NAMA_METODE[b['metode']]} || k: {k} || tau: {t} || MAE/kelas: {mae_c} || bias: {bias}"
        )

    print("\n### EFEK KALIBRASI (naif vs terbaik per skenario)")
    kunci = lambda b: (b["detektor"], b["korpus_latih"], b["kalibrasi"], b["uji"])
    skenario = {}
    for b in baris:
        skenario.setdefault(kunci(b), []).append(b)
    for kk, rows in skenario.items():
        basis = "irisan" if kk[3] == "763-test" else "penuh"
        naif = next(r for r in rows if r["metode"] == "naif")
        terbaik = min(
            (r for r in rows if r["metode"] != "naif"),
            key=lambda r: metrik(r, basis)["mae_makro"],
        )
        mn = metrik(naif, basis)["mae_makro"]
        mt = metrik(terbaik, basis)["mae_makro"]
        print(
            f"{kk[0]} | {kk[1]} | {kk[2]} -> {kk[3]} | naif {koma(mn,3)} -> "
            f"{NAMA_METODE[terbaik['metode']]} {koma(mt,3)} | "
            f"turun {koma((mn-mt)/mn*100,1)}% | Acc±1 {koma(metrik(naif,basis)['acc_pm1_makro'],3)}"
            f" -> {koma(metrik(terbaik,basis)['acc_pm1_makro'],3)}"
        )

    print("\n### RERATA PER METODE (lintas 6 skenario)")
    for metode in ["naif", "global", "k_perkelas", "k_tau_perkelas"]:
        rows = [b for b in baris if b["metode"] == metode]
        nilai = [
            metrik(b, "irisan" if b["uji"] == "763-test" else "penuh")["mae_makro"]
            for b in rows
        ]
        acc = [
            metrik(b, "irisan" if b["uji"] == "763-test" else "penuh")["acc_pm1_makro"]
            for b in rows
        ]
        print(
            f"{NAMA_METODE[metode]} | MAE rerata {koma(sum(nilai)/len(nilai),4)} | "
            f"Acc±1 rerata {koma(sum(acc)/len(acc),4)} | n={len(rows)}"
        )


if __name__ == "__main__":
    main()
