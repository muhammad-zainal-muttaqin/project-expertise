"""PT-E-018 — C1, C2, C3 sebagai ANGGOTA ENSEMBLE, bukan pesaing.

## Kenapa ini layak dicoba padahal PT-E-012 menutup jalur modul C

PT-E-012 mengadu C1/C2/C3 satu lawan satu dan menyimpulkan tidak ada yang
mengalahkan C1, jadi "seluruh jalur tingkatkan modul C" tertutup. Kesimpulan itu
sah untuk pertanyaan yang ia ajukan -- tapi ia tidak pernah menanyakan yang ini:
apakah galat mereka TERDEKORELASI.

Alasan fisik untuk menduga iya, dan ini bukan harapan kosong. C1 adalah kepala
klasifikasi DETEKTOR: dilatih di tugas deteksi penuh, 3.000 citra, augmentasi
mosaic/scale/hsv, sinyal supervisi berupa kotak+kelas sekaligus. C2 adalah
classifier POTONGAN: dilatih di 7.427 potongan sudah-terpotong, augmentasi
flip+brightness saja, sinyal kelas murni. Keduanya melihat objek yang sama lewat
dua rezim latih yang nyaris tidak beririsan. Model yang salah karena alasan
berbeda adalah bahan ensemble yang benar.

Bukti tambahan dari sesi ini: sel kontrol PT-E-014 mereproduksi C1 R4 PERSIS
(0,7208) tetapi C2 meleset 2,64 pp dari PT-E-012 hanya karena urutan inisialisasi
RNG berbeda. Anggota dengan varians sebesar itu justru paling diuntungkan
perataan.

## Kenapa ini penting untuk target 80% di IDEA.md

Plafon oracle R4 di atas skor detektor adalah 73,60% (PT-E-001). Artinya
memperbaiki penaut SAMPAI SEMPURNA pun tidak bisa menembus 80% -- plafonnya
ditentukan mutu probabilitas per-tampak, bukan mutu penautan. Satu-satunya cara
menaikkan plafon adalah probabilitas per-tampak yang lebih baik. Ensemble adalah
cara termurah yang tersedia: nol training baru, hanya kombinasi dump.

## Protokol

Bobot campuran dan `tau` dicari di VAL, test disentuh SEKALI. Semua anggota
dinilai pada himpunan tandan yang sama (potongan GT, tautan oracle), jadi
sebanding baris per baris dengan PT-E-012 dan PT-E-014.

Pemakaian:
    .venv/bin/python pipeline-pertandan/scripts/ensemble_c.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import penaut_pertandan as PP           # noqa: E402
import eval_pertandan as EP             # noqa: E402

SUB = PP.SUB
KELAS = PP.KELAS
SKEMA = "conf_luas"
K = 4


def muat_dump():
    """Kumpulkan seluruh dump sel PT-E-014/015 yang sudah ada."""
    out = {}
    for f in sorted((SUB / "results").glob("pt_e_014_prob_*.npz")):
        tag = f.stem.replace("pt_e_014_prob_", "")
        out[tag] = dict(np.load(f, allow_pickle=True))
    return out


def pool_dari(P_rata, offset, i):
    return P_rata[offset[i]:offset[i + 1]]


def bangun_pools(P_rata, offset, y, skema=SKEMA):
    pools = []
    for i in range(len(y)):
        P = pool_dari(P_rata, offset, i)
        pools.append({"tree": i, "gt": int(y[i]),
                      "pool": [{"p": p / max(p.sum(), 1e-9), "conf": float(p.max()),
                                "luas": 1.0, "tepi": 0.5} for p in P]})
    return pools


def nilai_anggota(P_rata, offset, y, tau=None, cari=False):
    pools = bangun_pools(P_rata, offset, y)
    if cari:
        tau = EP.cari_tau(pools, SKEMA)
    m = EP.nilai(pools, "R4", SKEMA, tau)
    multi = [q for q in pools if len(q["pool"]) >= 2]
    return m, EP.nilai(multi, "R4", SKEMA, tau), tau


def main() -> int:
    dumps = muat_dump()
    if not dumps:
        print("belum ada dump pt_e_014_prob_*.npz -- jalankan sel modul C dulu")
        return 1
    print(f"dump tersedia: {list(dumps)}")

    # C1 identik di semua sel (skor detektor, tanpa training) -- ambil dari mana saja
    ref = dumps[sorted(dumps)[0]]
    anggota = {}          # nama -> {split: P_rata}
    anggota["C1"] = {s: ref[f"{s}__C1_rata"] for s in ("val", "test")}
    for tag, d in sorted(dumps.items()):
        anggota[f"C2_{tag}"] = {s: d[f"{s}__C2_rata"] for s in ("val", "test")}

    off = {s: ref[f"{s}__offset"] for s in ("val", "test")}
    y = {s: ref[f"{s}__y"] for s in ("val", "test")}
    for tag, d in dumps.items():                    # kewarasan: himpunan tandan sama
        for s in ("val", "test"):
            assert np.array_equal(d[f"{s}__offset"], off[s]), f"offset beda di {tag}/{s}"
            assert np.array_equal(d[f"{s}__y"], y[s]), f"label beda di {tag}/{s}"

    hasil = {"pt_e": "018", "anggota": list(anggota), "sendiri": {}}
    print(f"\n{'anggota':28} {'val R4':>8} {'test R4':>8} {'test multi':>11}")
    for nama, P in anggota.items():
        mv, _, tau = nilai_anggota(P["val"], off["val"], y["val"], cari=True)
        mt, mtm, _ = nilai_anggota(P["test"], off["test"], y["test"], tau)
        hasil["sendiri"][nama] = {"tau": tau, "val": mv, "test": mt, "test_multi": mtm}
        print(f"{nama:28} {mv['akurasi']:>8} {mt['akurasi']:>8} {mtm['akurasi']:>11}")

    # ---- ensemble: rata-rata probabilitas, subset dipilih SERAKAH di VAL ----
    #
    # Serakah, bukan ekshaustif. Dengan 13 anggota, ekshaustif berarti 8.191
    # subset x 3 gaya bobot, dan tiap evaluasi memanggil `cari_tau` yang sendirinya
    # menyapu ~31 ribu triplet ambang -- ordenya jam, bukan menit. Seleksi maju
    # serakah menilai O(n^2) ~ 90 kombinasi dan dalam praktik menemukan subset yang
    # sama bagusnya untuk ensemble rata-rata, yang memang tidak punya interaksi
    # tajam antar-anggota.
    #
    # `tau` DIPATOK selama pencarian dan baru dipas ulang untuk pemenangnya.
    # Kalau `tau` ikut dicari di tiap langkah, ia menyerap sebagian keuntungan
    # dan seleksinya jadi memilih anggota yang cocok dengan ambang tertentu,
    # bukan anggota yang informasinya saling melengkapi (jebakan yang sama dengan
    # rekalibrasi tersamar sebagai agregasi di PT-E-001).
    TAU_PATOK = (0.5, 1.5, 2.5)

    def akurasi_val(subset, bobot):
        Pv = sum(b * anggota[n]["val"] for b, n in zip(bobot, subset))
        pools = bangun_pools(Pv, off["val"], y["val"])
        return EP.nilai(pools, "R4", SKEMA, TAU_PATOK)["akurasi"]

    print("\nseleksi maju serakah di VAL (tau dipatok selama pencarian)...")
    terpilih, sisa = [], list(anggota)
    skor_jalan = -1.0
    while sisa:
        kand = []
        for n in sisa:
            sub = terpilih + [n]
            kand.append((akurasi_val(sub, np.full(len(sub), 1 / len(sub))), n))
        kand.sort(reverse=True)
        if kand[0][0] <= skor_jalan + 1e-9:
            break
        skor_jalan, pilih = kand[0]
        terpilih.append(pilih); sisa.remove(pilih)
        print(f"  + {pilih:26} val R4 {skor_jalan:.4f}  ({len(terpilih)} anggota)")

    subset = tuple(terpilih)
    # setelah subset tetap, coba geser bobot ke C1 lalu pas `tau` sekali
    terbaik = None
    for gaya, w in (("seragam", 1.0), ("c1_2x", 2.0), ("c1_3x", 3.0)):
        if gaya != "seragam" and "C1" not in subset:
            continue
        bobot = np.array([(w if n == "C1" else 1.0) for n in subset], float)
        bobot /= bobot.sum()
        Pv = sum(b * anggota[n]["val"] for b, n in zip(bobot, subset))
        mv, _, tau = nilai_anggota(Pv, off["val"], y["val"], cari=True)
        if terbaik is None or mv["akurasi"] > terbaik[0]:
            terbaik = (mv["akurasi"], gaya, bobot, tau)
    akv, gaya, bobot, tau = terbaik
    print(f"  -> {subset} gaya={gaya} bobot={np.round(bobot,3).tolist()} "
          f"tau={tau} | val R4 {akv}")

    Pt = sum(b * anggota[n]["test"] for b, n in zip(bobot, subset))
    mt, mtm, _ = nilai_anggota(Pt, off["test"], y["test"], tau)
    Pv = sum(b * anggota[n]["val"] for b, n in zip(bobot, subset))
    mv, mvm, _ = nilai_anggota(Pv, off["val"], y["val"], tau)

    # bootstrap CI tingkat POHON terhadap C1 (baseline yang harus dikalahkan)
    pools_e = bangun_pools(Pt, off["test"], y["test"])
    pools_1 = bangun_pools(anggota["C1"]["test"], off["test"], y["test"])
    tau1 = hasil["sendiri"]["C1"]["tau"]
    tree = ref["test__tree"]
    benar_e = np.array([EP.benar(p["pool"], p["gt"], "R4", SKEMA, tau) for p in pools_e])
    benar_1 = np.array([EP.benar(p["pool"], p["gt"], "R4", SKEMA, tau1) for p in pools_1])
    uniq = sorted(set(tree.tolist()))
    idx_pohon = {t: np.where(tree == t)[0] for t in uniq}
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        pilih = rng.choice(len(uniq), len(uniq))
        ii = np.concatenate([idx_pohon[uniq[k]] for k in pilih])
        d.append(benar_e[ii].mean() - benar_1[ii].mean())
    d = np.array(d) * 100

    hasil["ensemble"] = {
        "subset": list(subset), "gaya_bobot": gaya, "bobot": bobot.tolist(),
        "tau": tau, "val": mv, "val_multi": mvm, "test": mt, "test_multi": mtm,
        "vs_C1": {
            "delta_pp": round(float((benar_e.mean() - benar_1.mean()) * 100), 2),
            "ci95": [round(float(np.percentile(d, 2.5)), 2),
                     round(float(np.percentile(d, 97.5)), 2)],
            "P(delta>0)": round(float((d > 0).mean()), 3),
            "n_pohon": len(uniq)},
        "plafon_oracle_pt_e_001": 0.7360,
        "target_IDEA": 0.80}
    f = SUB / "results" / "pt_e_018_ensemble.json"
    f.write_text(json.dumps(hasil, indent=1, ensure_ascii=False))
    print("\n=== ENSEMBLE ===")
    print(json.dumps(hasil["ensemble"], indent=1, ensure_ascii=False))
    print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
