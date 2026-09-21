"""Pencacahan per pohon dengan deduplikasi lintas sisi (bilangan bulat per tandan fisik).

Metode koefisien V2-E-050 menghitung round(k_c * n_c(t)), dengan n_c(t) jumlah
deteksi dari semua sisi. Koefisien k_c adalah rerata populasi, sehingga tidak
mengikuti perbedaan jumlah kemunculan tandan antarpohon. Skrip ini menguji
pengganti yang menghitung tandan fisik secara langsung:

1. Deteksi per sisi dirapikan: kotak kembar beda kelas digabung, lalu NMS.
2. Model pasangan, dilatih pada kotak acuan partisi latih, memberi peluang dua
   kotak pada sisi bersebelahan merupakan tandan yang sama.
3. Tautan digabung secara rakus dengan union-find; satu komponen tidak boleh
   memuat dua kotak dari sisi yang sama.
4. Setiap komponen dihitung satu tandan; kelasnya dari agregasi peluang kelas.

Ambang dipilih dengan lipat-silang lima lipatan pada pohon uji, sama dengan
protokol V2-E-050, sehingga kedua metode dibandingkan setara. Dua analisis
batas atas ikut dihitung: kedua metode pada kotak acuan (deteksi sempurna) dan
deduplikasi dengan tautan acuan pada deteksi nyata.

Pemakaian:
    python scripts/pencacahan_dedup_pohon.py \
        --baseline-root "D:/Work/Assisten-Dosen/Baseline-SawitMVC" \
        --depth-root "D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalibrasi_koefisien_pencacahan import (  # noqa: E402
    CLASSES,
    TAU_GRID,
    matriks_hitung,
    muat_prediksi,
    pasang_koefisien,
)

N_LIPATAN = 5
SEED = 42
LANTAI_CONF = 0.05
IOU_KEMBAR = 0.90
IOU_NMS = 0.60
IOU_COCOK = 0.50
TAU_DET = np.round(np.arange(0.15, 0.61, 0.05), 2)
TAU_TAUT = np.round(np.concatenate([[0.05, 0.1, 0.15], np.arange(0.2, 0.81, 0.1)]), 2)
TAMBAH_SIMPAN = np.round(np.arange(0.0, 0.21, 0.05), 2)
OFFSET_KELAS = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
METODE_KOEFISIEN = ["global", "k_perkelas", "k_tau_perkelas"]
DETEKTOR = [("yolo26l", "YOLO26l"), ("rtdetr_l", "RT-DETR-L"), ("rfdetr_l", "RF-DETR-L")]


# --------------------------------------------------------------------------
# Nilai acuan per pohon: kotak per sisi, identitas tandan, dan cacah per kelas
# --------------------------------------------------------------------------
def muat_pohon_json(berkas: Path) -> tuple[str, dict]:
    data = json.loads(berkas.read_text(encoding="utf-8-sig"))
    peta_tandan = {}
    for tandan in data.get("bunches", []):
        for muncul in tandan["appearances"]:
            peta_tandan[(muncul["side_index"], muncul["box_index"])] = tandan["bunch_id"]
    sisi = sorted(data["images"].values(), key=lambda s: s["side_index"])
    kotak, kelas, identitas, ukuran = [], [], [], []
    lepas = -1
    for s in sisi:
        k_s, c_s, t_s = [], [], []
        for ann in s["annotations"]:
            if ann["class_name"] not in CLASSES:
                continue
            k_s.append(ann["bbox_yolo"])
            c_s.append(CLASSES.index(ann["class_name"]))
            nomor = peta_tandan.get((s["side_index"], ann["box_index"]))
            if nomor is None:
                nomor, lepas = lepas, lepas - 1
            t_s.append(nomor)
        kotak.append(np.array(k_s, dtype=float).reshape(-1, 4))
        kelas.append(np.array(c_s, dtype=int))
        identitas.append(np.array(t_s, dtype=int))
        ukuran.append((float(s["width"]), float(s["height"])))
    cacah = np.zeros(4)
    for tandan in data.get("bunches", []):
        if tandan["class"] in CLASSES:
            cacah[CLASSES.index(tandan["class"])] += 1
    kemunculan = [sorted({m["side_index"] for m in t["appearances"]}) for t in data.get("bunches", [])]
    return data["tree_id"], {
        "S": len(sisi), "ukuran": ukuran, "kotak": kotak, "kelas": kelas, "tandan": identitas, "cacah_json": cacah,
        "kemunculan": kemunculan,
    }


def sebaran_kemunculan(pohon: dict[str, dict]) -> dict:
    """Jumlah tandan menurut banyaknya sisi tempat ia muncul, dan apakah sisinya bersambung."""
    jumlah, bersambung, jamak = {}, 0, 0
    for isi in pohon.values():
        s_total = isi["S"]
        for sisi in isi["kemunculan"]:
            kunci = f"{s_total}_sisi"
            jumlah.setdefault(kunci, {})
            jumlah[kunci][len(sisi)] = jumlah[kunci].get(len(sisi), 0) + 1
            if len(sisi) >= 2:
                jamak += 1
                terisi = set(sisi)
                bersambung += any(all((a + k) % s_total in terisi for k in range(len(sisi))) for a in sisi)
    return {"per_jumlah_sisi": jumlah, "tandan_multisisi": jamak, "multisisi_bersambung": bersambung}


def muat_korpus(args: argparse.Namespace, korpus: str) -> tuple[dict[str, dict], dict[str, dict[str, np.ndarray]]]:
    """-> (pohon per id, cacah acuan per split dari sumber yang sama dengan laporan)."""
    pohon, cacah = {}, {"train": {}, "val": {}, "test": {}}
    if korpus == "953":
        akar = args.baseline_root
        with (akar / "ground_truth" / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
            for baris in csv.DictReader(fh):
                split = baris["new_split"].strip()
                if split in cacah:
                    cacah[split][baris["tree_id"].strip()] = np.array([float(baris[c]) for c in CLASSES])
        for berkas in sorted((akar / "ground_truth" / "annotations").glob("*.json")):
            nama, isi = muat_pohon_json(berkas)
            pohon[nama] = isi
    else:
        for folder, split in {"train": "train", "valid": "val", "test": "test"}.items():
            for berkas in sorted((args.depth_root / folder / "linked").glob("*.json")):
                nama, isi = muat_pohon_json(berkas)
                pohon[nama] = isi
                by_class = json.loads(berkas.read_text(encoding="utf-8-sig")).get("summary", {}).get("by_class", {})
                cacah[split][nama] = np.array([float(by_class.get(c, 0)) for c in CLASSES])
    return pohon, cacah


def sumber_dump(akar: Path, korpus: str, slug: str) -> Path:
    if korpus == "953":
        return akar / f"results/pred_{slug}_v2repro_953_test.npz"
    return akar / f"results/new763/predictions/{slug}_rgb_s42_i1280__test.npz"


# --------------------------------------------------------------------------
# Deteksi per sisi
# --------------------------------------------------------------------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    irisan = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    luas_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    luas_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return irisan / np.maximum(luas_a[:, None] + luas_b[None, :] - irisan, 1e-9)


def rapikan_deteksi(larik: np.ndarray, lebar: float, tinggi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 6) xyxy-conf-kelas -> (kotak cxcywh ternormalisasi, skor, peluang kelas)."""
    kosong = (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)))
    if larik.size == 0:
        return kosong
    kelas, conf = larik[:, 5].astype(int), larik[:, 4].astype(float)
    sah = (kelas >= 0) & (kelas <= 3) & (conf >= LANTAI_CONF)
    larik, kelas, conf = larik[sah], kelas[sah], conf[sah]
    if len(larik) == 0:
        return kosong
    urut = np.argsort(-conf)
    larik, kelas, conf = larik[urut], kelas[urut], conf[urut]

    # Kueri DETR sering memancarkan kotak yang sama dengan kelas berbeda
    wakil, peluang = [], []
    for i in range(len(larik)):
        if wakil:
            iou = iou_xyxy(larik[i:i + 1, :4], np.array(wakil))[0]
            g = int(np.argmax(iou))
            if iou[g] >= IOU_KEMBAR:
                peluang[g][kelas[i]] = max(peluang[g][kelas[i]], conf[i])
                continue
        wakil.append(larik[i, :4])
        vektor = np.zeros(4)
        vektor[kelas[i]] = conf[i]
        peluang.append(vektor)
    wakil, peluang = np.array(wakil), np.array(peluang)
    skor = peluang.max(1)

    iou = iou_xyxy(wakil, wakil)
    tersisih = np.zeros(len(wakil), dtype=bool)
    simpan = []
    for i in np.argsort(-skor):
        if not tersisih[i]:
            simpan.append(i)
            tersisih |= iou[i] >= IOU_NMS
    wakil, peluang, skor = wakil[simpan], peluang[simpan], skor[simpan]
    kotak = np.column_stack([
        (wakil[:, 0] + wakil[:, 2]) / 2 / lebar, (wakil[:, 1] + wakil[:, 3]) / 2 / tinggi,
        (wakil[:, 2] - wakil[:, 0]) / lebar, (wakil[:, 3] - wakil[:, 1]) / tinggi,
    ])
    return kotak, skor, peluang / peluang.sum(1, keepdims=True)


def muat_deteksi(path: Path, pohon: dict[str, dict], daftar: list[str]) -> dict[str, list[tuple]]:
    data = np.load(path, allow_pickle=True)
    hasil = {}
    for nama in daftar:
        isi = pohon[nama]
        hasil[nama] = []
        for s in range(isi["S"]):
            kunci = f"{nama}_{s + 1}"
            larik = data[kunci] if kunci in data.files else np.zeros((0, 6))
            hasil[nama].append(rapikan_deteksi(larik, *isi["ukuran"][s]))
    return hasil


def fusi_sisi(bagian: list[tuple], iou_fusi: float = 0.55) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fusi gaya WBF antardetektor pada satu sisi; skor = jumlah skor anggota / jumlah detektor."""
    kotak = np.concatenate([b[0] for b in bagian]).reshape(-1, 4)
    skor = np.concatenate([b[1] for b in bagian])
    peluang = np.concatenate([b[2] for b in bagian]).reshape(-1, 4)
    asal = np.concatenate([np.full(len(b[1]), i) for i, b in enumerate(bagian)])
    if len(skor) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4))
    jumlah_kotak, bobot, jumlah_peluang, anggota = [], [], [], []
    for i in np.argsort(-skor):
        pilih = -1
        if jumlah_kotak:
            gabungan = np.array(jumlah_kotak) / np.array(bobot)[:, None]
            iou = iou_xyxy(cxcywh_ke_xyxy(kotak[i:i + 1]), cxcywh_ke_xyxy(gabungan))[0]
            for j in np.argsort(-iou):
                if iou[j] < iou_fusi:
                    break
                if asal[i] not in anggota[j]:
                    pilih = j
                    break
        if pilih < 0:
            jumlah_kotak.append(np.zeros(4))
            bobot.append(0.0)
            jumlah_peluang.append(np.zeros(4))
            anggota.append(set())
            pilih = len(bobot) - 1
        jumlah_kotak[pilih] = jumlah_kotak[pilih] + skor[i] * kotak[i]
        bobot[pilih] += skor[i]
        jumlah_peluang[pilih] = jumlah_peluang[pilih] + skor[i] * peluang[i]
        anggota[pilih].add(int(asal[i]))
    bobot = np.array(bobot)
    return (np.array(jumlah_kotak) / bobot[:, None], bobot / len(bagian),
            np.array(jumlah_peluang) / bobot[:, None])


def fusi_deteksi(semua: list[dict[str, list[tuple]]], daftar: list[str]) -> dict[str, list[tuple]]:
    return {n: [fusi_sisi([d[n][s] for d in semua]) for s in range(len(semua[0][n]))] for n in daftar}


def prediksi_koefisien_dari_deteksi(deteksi: dict[str, list[tuple]]) -> dict[str, list[np.ndarray]]:
    """Format muat_prediksi: skor tiap deteksi, dikelompokkan menurut kelas argmax."""
    hasil = {}
    for nama, sisi in deteksi.items():
        wadah = [[], [], [], []]
        for _, skor, peluang in sisi:
            for sk, kelas in zip(skor, np.argmax(peluang, 1) if len(skor) else []):
                wadah[int(kelas)].append(float(sk))
        hasil[nama] = [np.sort(np.array(v, dtype=float))[::-1] for v in wadah]
    return hasil


# --------------------------------------------------------------------------
# Model pasangan lintas sisi
# --------------------------------------------------------------------------
def konteks_sisi(kotak: np.ndarray) -> tuple:
    n = len(kotak)
    x, y = kotak[:, 0], kotak[:, 1]
    peringkat = np.argsort(np.argsort(y)) / max(n - 1, 1)
    iqr_x = max(np.percentile(x, 75) - np.percentile(x, 25), 0.05)
    iqr_y = max(np.percentile(y, 75) - np.percentile(y, 25), 0.05)
    return peringkat, np.median(x), np.median(y), iqr_x, iqr_y


def sudut_silinder(kotak: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posisi horizontal relatif terhadap sumbu tajuk: u = (x - pusat) / jari-jari, dan peringkat x."""
    x = kotak[:, 0]
    pusat = (x.min() + x.max()) / 2
    jari = max((x.max() - x.min()) / 2, 0.05)
    return (x - pusat) / jari, np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def fitur_pasangan(ka: np.ndarray, kb: np.ndarray, jumlah_sisi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fitur geometri untuk semua pasangan (kotak sisi s, kotak sisi s+1)."""
    ra, mxa, mya, ixa, iya = konteks_sisi(ka)
    rb, mxb, myb, ixb, iyb = konteks_sisi(kb)
    ia = np.repeat(np.arange(len(ka)), len(kb))
    ib = np.tile(np.arange(len(kb)), len(ka))
    a, b = ka[ia], kb[ib]
    wa, ha = np.maximum(a[:, 2], 1e-4), np.maximum(a[:, 3], 1e-4)
    wb, hb = np.maximum(b[:, 2], 1e-4), np.maximum(b[:, 3], 1e-4)
    rel_ya, rel_yb = (a[:, 1] - mya) / iya, (b[:, 1] - myb) / iyb
    ua, pxa = sudut_silinder(ka)
    ub, pxb = sudut_silinder(kb)
    ua, ub, pxa, pxb = ua[ia], ub[ib], pxa[ia], pxb[ib]
    fitur = np.column_stack([
        ua, ub, ua ** 2 + ub ** 2, ua * ub, pxa, pxb, pxa + pxb,
        a[:, 0], a[:, 1], wa, ha, b[:, 0], b[:, 1], wb, hb,
        b[:, 0] - a[:, 0], b[:, 1] - a[:, 1], np.log(wb / wa), np.log(hb / ha),
        (a[:, 0] - mxa) / ixa, (b[:, 0] - mxb) / ixb, rel_ya, rel_yb, rel_yb - rel_ya,
        ra[ia], rb[ib], rb[ib] - ra[ia],
        np.full(len(ia), len(ka)), np.full(len(ia), len(kb)), np.full(len(ia), jumlah_sisi),
    ])
    return fitur, ia, ib


def data_latih_pasangan(daftar: list[dict], rng: np.random.Generator, ulang: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Pasangan dari kotak acuan partisi latih; ulangan >0 diberi derau mirip deteksi."""
    semua = np.concatenate([k for p in daftar for k in p["kotak"] if len(k)])
    x_kumpul, y_kumpul = [], []
    lepas = -10**6
    for p in daftar:
        for u in range(ulang):
            sisi_k, sisi_t = [], []
            for kotak, tandan in zip(p["kotak"], p["tandan"]):
                k, t = kotak.copy(), tandan.copy()
                if u > 0 and len(k):
                    n = len(k)
                    k[:, 0] += rng.normal(0, 0.05, n) * k[:, 2]
                    k[:, 1] += rng.normal(0, 0.05, n) * k[:, 3]
                    k[:, 2:] *= np.exp(rng.normal(0, 0.1, (n, 2)))
                    tetap = rng.random(n) > 0.15
                    k, t = k[tetap], t[tetap]
                    tambah = rng.poisson(0.6)
                    if tambah:
                        k = np.vstack([k, semua[rng.integers(0, len(semua), tambah)]])
                        t = np.concatenate([t, np.arange(lepas, lepas - tambah, -1)])
                        lepas -= tambah
                sisi_k.append(k)
                sisi_t.append(t)
            for s in range(p["S"]):
                s2 = (s + 1) % p["S"]
                if len(sisi_k[s]) == 0 or len(sisi_k[s2]) == 0:
                    continue
                fitur, ia, ib = fitur_pasangan(sisi_k[s], sisi_k[s2], p["S"])
                x_kumpul.append(fitur)
                y_kumpul.append((sisi_t[s][ia] == sisi_t[s2][ib]) & (sisi_t[s][ia] >= 0))
    return np.concatenate(x_kumpul), np.concatenate(y_kumpul)


def latih_model_pasangan(daftar: list[dict]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = data_latih_pasangan(daftar, np.random.default_rng(SEED))
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED,
    ).fit(x, y)
    return model, {"n_pasangan": int(len(y)), "positif": int(y.sum())}


def tepi_semua_pohon(model, sisi_per_pohon: list[list[np.ndarray]], jumlah_sisi: list[int]) -> list[tuple]:
    """Peluang tautan untuk semua pohon sekaligus -> [(p, i, j) terurut turun per pohon]."""
    fitur_kumpul, penanda = [], []
    for t, (sisi, s_total) in enumerate(zip(sisi_per_pohon, jumlah_sisi)):
        geser = np.concatenate([[0], np.cumsum([len(k) for k in sisi])])
        for s in range(s_total):
            s2 = (s + 1) % s_total
            if len(sisi[s]) == 0 or len(sisi[s2]) == 0:
                continue
            fitur, ia, ib = fitur_pasangan(sisi[s], sisi[s2], s_total)
            fitur_kumpul.append(fitur)
            penanda.append((t, ia + geser[s], ib + geser[s2]))
    hasil = [(np.zeros(0), np.zeros(0, int), np.zeros(0, int)) for _ in sisi_per_pohon]
    if not fitur_kumpul:
        return hasil
    peluang = model.predict_proba(np.concatenate(fitur_kumpul))[:, 1]
    potong = np.cumsum([0] + [len(f) for f in fitur_kumpul])
    per_pohon: dict[int, list] = {}
    for (t, gi, gj), a, b in zip(penanda, potong[:-1], potong[1:]):
        per_pohon.setdefault(t, []).append((peluang[a:b], gi, gj))
    for t, bagian in per_pohon.items():
        p = np.concatenate([b[0] for b in bagian])
        i = np.concatenate([b[1] for b in bagian])
        j = np.concatenate([b[2] for b in bagian])
        urut = np.argsort(-p)
        hasil[t] = (p[urut], i[urut], j[urut])
    return hasil


def ukuran_maks(jumlah_sisi: int) -> int:
    """Pada acuan, tandan pohon empat sisi muncul di paling banyak tiga sisi."""
    return max(2, (3 * jumlah_sisi) // 4)


def gabung_komponen(sisi_simpul: np.ndarray, tepi: tuple, tau: float, jumlah_sisi: int) -> np.ndarray:
    """Union-find rakus; satu komponen memuat paling banyak satu kotak per sisi."""
    induk = list(range(len(sisi_simpul)))
    topeng = [1 << int(s) for s in sisi_simpul]
    batas = ukuran_maks(jumlah_sisi)

    def cari(i: int) -> int:
        while induk[i] != i:
            induk[i] = induk[induk[i]]
            i = induk[i]
        return i

    for p, i, j in zip(*tepi):
        if p < tau:
            break
        ri, rj = cari(int(i)), cari(int(j))
        if ri == rj or topeng[ri] & topeng[rj] or bin(topeng[ri] | topeng[rj]).count("1") > batas:
            continue
        induk[rj] = ri
        topeng[ri] |= topeng[rj]
    return np.array([cari(i) for i in range(len(sisi_simpul))], dtype=int)


def ringkas_komponen(akar: np.ndarray, skor: np.ndarray, peluang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (skor komponen = 1 - prod(1 - skor anggota), log peluang kelas teragregasi)."""
    if len(akar) == 0:
        return np.zeros(0), np.zeros((0, 4))
    _, balik = np.unique(akar, return_inverse=True)
    n = balik.max() + 1
    log_tidak = np.zeros(n)
    np.add.at(log_tidak, balik, np.log1p(-np.clip(skor, 0, 0.999999)))
    jumlah = np.zeros((n, 4))
    np.add.at(jumlah, balik, skor[:, None] * peluang)
    return 1 - np.exp(log_tidak), np.log(jumlah / np.maximum(jumlah.sum(1, keepdims=True), 1e-9) + 1e-6)


def cacah_dari_komponen(skor_komp: np.ndarray, logit: np.ndarray, tau_simpan: float, offset: np.ndarray) -> np.ndarray:
    pilih = skor_komp >= tau_simpan
    if not pilih.any():
        return np.zeros(4)
    return np.bincount(np.argmax(logit[pilih] + offset, 1), minlength=4).astype(float)


# --------------------------------------------------------------------------
# Tautan acuan: deteksi dicocokkan ke kotak acuan, lalu dikelompokkan per tandan
# --------------------------------------------------------------------------
def akar_tautan_acuan(isi: dict, sisi_deteksi: list[np.ndarray]) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    # Deteksi tak tercocok diberi nomor negatif unik di luar rentang nomor kotak acuan
    label, lepas = [], -10**7
    for s, kotak in enumerate(sisi_deteksi):
        nomor = np.arange(lepas, lepas - len(kotak), -1)
        lepas -= len(kotak)
        acuan = isi["kotak"][s]
        if len(kotak) and len(acuan):
            iou = iou_xyxy(cxcywh_ke_xyxy(kotak), cxcywh_ke_xyxy(acuan))
            baris, kolom = linear_sum_assignment(-iou)
            for r, c in zip(baris, kolom):
                if iou[r, c] >= IOU_COCOK:
                    nomor[r] = isi["tandan"][s][c]
        label.append(nomor)
    semua = np.concatenate(label) if label else np.zeros(0, int)
    _, balik = np.unique(semua, return_inverse=True)
    return balik


def urai_galat(deteksi: dict[str, list[tuple]], pohon: dict[str, dict], daftar: list[str],
               tau_det: float, tau_simpan: float) -> dict:
    """Galat per pohon dengan tautan acuan: tandan terlewat, komponen palsu, dan salah kelas."""
    from scipy.optimize import linear_sum_assignment

    terlewat, palsu, salah, jumlah = [], [], [], []
    for nama in daftar:
        isi = pohon[nama]
        kunci, skor, peluang = [], [], []
        for s, (kotak, sk, pl) in enumerate(deteksi[nama]):
            pilih = sk >= tau_det
            kotak, sk, pl = kotak[pilih], sk[pilih], pl[pilih]
            nomor: list = [("palsu", s, i) for i in range(len(kotak))]
            acuan = isi["kotak"][s]
            if len(kotak) and len(acuan):
                iou = iou_xyxy(cxcywh_ke_xyxy(kotak), cxcywh_ke_xyxy(acuan))
                for r, c in zip(*linear_sum_assignment(-iou)):
                    if iou[r, c] >= IOU_COCOK:
                        nomor[r] = int(isi["tandan"][s][c])
            kunci += nomor
            skor.append(sk)
            peluang.append(pl)
        skor = np.concatenate(skor)
        peluang = np.concatenate(peluang).reshape(-1, 4)
        kelas_acuan = {int(t): int(c) for ts, cs in zip(isi["tandan"], isi["kelas"]) for t, c in zip(ts, cs)}
        grup: dict = {}
        for i, k in enumerate(kunci):
            grup.setdefault(k, []).append(i)
        disimpan = {k: v for k, v in grup.items() if 1 - np.prod(1 - np.clip(skor[v], 0, 0.999999)) >= tau_simpan}
        terdeteksi = {k for k in disimpan if not isinstance(k, tuple)}
        terlewat.append(len(set(kelas_acuan) - terdeteksi))
        palsu.append(sum(isinstance(k, tuple) for k in disimpan))
        salah.append(sum(int(np.argmax((skor[v, None] * peluang[v]).sum(0))) != kelas_acuan[k]
                         for k, v in disimpan.items() if not isinstance(k, tuple)))
        jumlah.append(len(kelas_acuan))
    terlewat, palsu, salah, jumlah = map(np.array, (terlewat, palsu, salah, jumlah))
    return {
        "tau_det": tau_det, "tau_simpan": tau_simpan,
        "tandan_per_pohon": round(float(jumlah.mean()), 4),
        "terlewat_per_pohon": round(float(terlewat.mean()), 4),
        "persen_terlewat": round(float(terlewat.sum() / jumlah.sum()), 4),
        "komponen_palsu_per_pohon": round(float(palsu.mean()), 4),
        "salah_kelas_per_pohon": round(float(salah.mean()), 4),
        "persen_salah_kelas_terdeteksi": round(float(salah.sum() / max(jumlah.sum() - terlewat.sum(), 1)), 4),
    }


def cxcywh_ke_xyxy(k: np.ndarray) -> np.ndarray:
    return np.column_stack([k[:, 0] - k[:, 2] / 2, k[:, 1] - k[:, 3] / 2, k[:, 0] + k[:, 2] / 2, k[:, 1] + k[:, 3] / 2])


# --------------------------------------------------------------------------
# Tabel cacah untuk semua kombinasi ambang, lalu pemilihan lipat-silang
# --------------------------------------------------------------------------
def tabel_dedup(model, deteksi: dict[str, list[tuple]], pohon: dict[str, dict], daftar: list[str], tautan_acuan: bool):
    """-> komponen[(i_det, i_taut)][t] = (skor komponen, logit kelas)."""
    komponen = {}
    daftar_taut = [0.0] if tautan_acuan else list(TAU_TAUT)
    for i_det, tau_det in enumerate(TAU_DET):
        sisi_per_pohon, skor_per_pohon, peluang_per_pohon, sisi_simpul = [], [], [], []
        for nama in daftar:
            sisi, skor, peluang, penanda = [], [], [], []
            for s, (kotak, sk, pl) in enumerate(deteksi[nama]):
                pilih = sk >= tau_det
                sisi.append(kotak[pilih])
                skor.append(sk[pilih])
                peluang.append(pl[pilih])
                penanda.append(np.full(pilih.sum(), s))
            sisi_per_pohon.append(sisi)
            skor_per_pohon.append(np.concatenate(skor))
            peluang_per_pohon.append(np.concatenate(peluang).reshape(-1, 4))
            sisi_simpul.append(np.concatenate(penanda))
        if tautan_acuan:
            akar_semua = [akar_tautan_acuan(pohon[n], s) for n, s in zip(daftar, sisi_per_pohon)]
            komponen[(i_det, 0)] = [ringkas_komponen(a, sk, pl) for a, sk, pl in
                                    zip(akar_semua, skor_per_pohon, peluang_per_pohon)]
            continue
        tepi = tepi_semua_pohon(model, sisi_per_pohon, [pohon[n]["S"] for n in daftar])
        for i_taut, tau_taut in enumerate(daftar_taut):
            komponen[(i_det, i_taut)] = [
                ringkas_komponen(gabung_komponen(ss, tp, tau_taut, pohon[n]["S"]), sk, pl)
                for n, ss, tp, sk, pl in zip(daftar, sisi_simpul, tepi, skor_per_pohon, peluang_per_pohon)
            ]
    return komponen, daftar_taut


def lipat_silang_dedup(komponen: dict, daftar_taut: list[float], y: np.ndarray, pakai_offset: bool = True):
    kombinasi = [(i_det, i_taut, tambah) for i_det in range(len(TAU_DET)) for i_taut in range(len(daftar_taut))
                 for tambah in TAMBAH_SIMPAN]
    nol = np.zeros(4)
    cacah = np.array([
        [cacah_dari_komponen(sk, lg, TAU_DET[i_det] + tambah, nol) for sk, lg in komponen[(i_det, i_taut)]]
        for i_det, i_taut, tambah in kombinasi
    ])
    galat = np.abs(cacah - y[None]).mean(2)
    urutan = np.random.default_rng(SEED).permutation(len(y))
    prediksi, pilihan = np.zeros_like(y), []
    kisi_offset = [np.array([a, b, 0.0, d]) for a in OFFSET_KELAS for b in OFFSET_KELAS for d in OFFSET_KELAS]
    for tahan in np.array_split(urutan, N_LIPATAN):
        latih = np.setdiff1d(urutan, tahan)
        terbaik = int(np.argmin(galat[:, latih].mean(1)))
        i_det, i_taut, tambah = kombinasi[terbaik]
        daftar_komp = komponen[(i_det, i_taut)]
        offset = nol
        if pakai_offset:
            nilai = [np.mean([np.abs(cacah_dari_komponen(*daftar_komp[t], TAU_DET[i_det] + tambah, o) - y[t]).mean()
                              for t in latih]) for o in kisi_offset]
            offset = kisi_offset[int(np.argmin(nilai))]
        for t in tahan:
            prediksi[t] = cacah_dari_komponen(*daftar_komp[t], TAU_DET[i_det] + tambah, offset)
        pilihan.append({"tau_det": float(TAU_DET[i_det]), "tau_taut": float(daftar_taut[i_taut]),
                        "tau_simpan": round(float(TAU_DET[i_det] + tambah), 2), "offset_kelas": offset.tolist()})
    return prediksi, pilihan


def lipat_silang_koefisien(n: np.ndarray, y: np.ndarray, metode: str) -> np.ndarray:
    """Sama dengan metrik_lipat_silang pada kalibrasi_pencacahan_perkorpus.py, tetapi mengembalikan prediksi."""
    urutan = np.random.default_rng(SEED).permutation(len(y))
    prediksi = np.zeros_like(y)
    for tahan in np.array_split(urutan, N_LIPATAN):
        latih = np.setdiff1d(urutan, tahan)
        tau, k = pasang_koefisien(n[:, latih, :], y[latih], metode)
        for c in range(4):
            j = int(np.argmin(np.abs(TAU_GRID - tau[c])))
            prediksi[tahan, c] = np.rint(k[c] * n[j, tahan, c])
    return prediksi


# --------------------------------------------------------------------------
# Metrik per pohon
# --------------------------------------------------------------------------
def metrik_pohon(prediksi: np.ndarray, y: np.ndarray) -> dict:
    d = prediksi - y
    total = prediksi.sum(1) - y.sum(1)
    per_kelas = {
        nama: {
            "mae": round(float(np.abs(d[:, c]).mean()), 4),
            "bias": round(float(d[:, c].mean()), 4),
            "tepat": round(float((d[:, c] == 0).mean()), 4),
            "pm1": round(float((np.abs(d[:, c]) <= 1).mean()), 4),
        }
        for c, nama in enumerate(CLASSES)
    }
    return {
        "n_pohon": int(len(y)),
        "mae_makro": round(float(np.abs(d).mean()), 4),
        "tepat_makro": round(float((d == 0).mean()), 4),
        "pm1_makro": round(float((np.abs(d) <= 1).mean()), 4),
        "bias_abs_makro": round(float(np.abs(d.mean(0)).mean()), 4),
        "mae_total": round(float(np.abs(total).mean()), 4),
        "tepat_total": round(float((total == 0).mean()), 4),
        "pm1_total": round(float((np.abs(total) <= 1).mean()), 4),
        "pohon_semua_kelas_tepat": round(float((d == 0).all(1).mean()), 4),
        "pohon_semua_kelas_pm1": round(float((np.abs(d) <= 1).all(1).mean()), 4),
        "per_kelas": per_kelas,
    }


def bootstrap_selisih(pa: np.ndarray, pb: np.ndarray, y: np.ndarray, ulang: int = 2000) -> dict:
    """Selisih berpasangan A - B per pohon; selang kepercayaan 95% dari resampel pohon."""
    rng = np.random.default_rng(SEED)
    indeks = rng.integers(0, len(y), (ulang, len(y)))
    ukuran = {
        "mae_makro": np.abs(pa - y).mean(1) - np.abs(pb - y).mean(1),
        "mae_total": np.abs(pa.sum(1) - y.sum(1)) - np.abs(pb.sum(1) - y.sum(1)),
        "pohon_semua_kelas_pm1": (np.abs(pa - y) <= 1).all(1).astype(float) - (np.abs(pb - y) <= 1).all(1).astype(float),
    }
    hasil = {}
    for nama, d in ukuran.items():
        sampel = d[indeks].mean(1)
        hasil[nama] = {"selisih": round(float(d.mean()), 4),
                       "sk95": [round(float(np.percentile(sampel, 2.5)), 4), round(float(np.percentile(sampel, 97.5)), 4)]}
    return hasil


def kualitas_tautan(akar: np.ndarray, tandan: np.ndarray) -> tuple[int, int, int]:
    """(tp, fp, fn) pasangan simpul pada komponen yang sama vs tandan yang sama."""
    sama_pred = akar[:, None] == akar[None, :]
    sama_acuan = (tandan[:, None] == tandan[None, :]) & (tandan[:, None] >= 0)
    atas = np.triu(np.ones_like(sama_pred), 1)
    tp = int((sama_pred & sama_acuan & atas).sum())
    return tp, int((sama_pred & ~sama_acuan & atas).sum()), int((~sama_pred & sama_acuan & atas).sum())


# --------------------------------------------------------------------------
# Batas atas pada kotak acuan
# --------------------------------------------------------------------------
def analisis_kotak_acuan(model, pohon: dict[str, dict], daftar: list[str], y: np.ndarray) -> dict:
    # Koefisien: n_c(t) = jumlah kotak acuan kelas c pada semua sisi
    n = np.array([[float(sum((k == c).sum() for k in pohon[p]["kelas"])) for c in range(4)] for p in daftar])
    n_tau = np.repeat(n[None], len(TAU_GRID), 0)
    pred_koef = lipat_silang_koefisien(n_tau, y, "k_perkelas")

    sisi_per_pohon = [pohon[p]["kotak"] for p in daftar]
    tepi = tepi_semua_pohon(model, sisi_per_pohon, [pohon[p]["S"] for p in daftar])
    simpul = [np.concatenate([np.full(len(k), s) for s, k in enumerate(pohon[p]["kotak"])]) for p in daftar]
    kelas = [np.concatenate(pohon[p]["kelas"]) for p in daftar]
    tandan = [np.concatenate(pohon[p]["tandan"]) for p in daftar]
    cacah_per_tau, taut_per_tau = [], []
    for tau in TAU_TAUT:
        cacah, jumlah = [], np.zeros(3)
        for p, ss, tp, kl, td in zip(daftar, simpul, tepi, kelas, tandan):
            akar = gabung_komponen(ss, tp, tau, pohon[p]["S"])
            _, logit = ringkas_komponen(akar, np.ones(len(akar)), np.eye(4)[kl])
            cacah.append(np.bincount(np.argmax(logit, 1), minlength=4).astype(float) if len(akar) else np.zeros(4))
            jumlah += kualitas_tautan(akar, td)
        cacah_per_tau.append(np.array(cacah))
        taut_per_tau.append(jumlah)
    cacah_per_tau = np.array(cacah_per_tau)

    urutan = np.random.default_rng(SEED).permutation(len(y))
    pred_dedup, tau_pilih = np.zeros_like(y), []
    for tahan in np.array_split(urutan, N_LIPATAN):
        latih = np.setdiff1d(urutan, tahan)
        i = int(np.argmin(np.abs(cacah_per_tau[:, latih] - y[latih][None]).mean((1, 2))))
        pred_dedup[tahan] = cacah_per_tau[i, tahan]
        tau_pilih.append(float(TAU_TAUT[i]))
    i_tengah = int(np.argmin(np.abs(TAU_TAUT - np.median(tau_pilih))))
    tp, fp, fn = taut_per_tau[i_tengah]
    return {
        "koefisien_k_perkelas": metrik_pohon(pred_koef, y),
        "dedup": metrik_pohon(pred_dedup, y),
        "tau_taut_per_lipatan": tau_pilih,
        "kualitas_tautan": {"tau": float(TAU_TAUT[i_tengah]), "presisi": round(tp / max(tp + fp, 1), 4),
                            "recall": round(tp / max(tp + fn, 1), 4), "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)},
        "bootstrap_dedup_minus_koefisien": bootstrap_selisih(pred_dedup, pred_koef, y),
    }


# --------------------------------------------------------------------------
# Utama
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-root", type=Path, default=Path("D:/Work/Assisten-Dosen/Baseline-SawitMVC"))
    parser.add_argument("--depth-root", type=Path,
                        default=Path("D:/Work/Assisten-Dosen/SawitMVC-Depth/SawitMVC-Depth-YOLO"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--keluaran", type=Path, default=Path("results/pencacahan_dedup_2026-09-21/ringkasan.json"))
    parser.add_argument("--korpus", nargs="+", default=["953", "763"])
    args = parser.parse_args()

    ringkasan: dict = {"protokol": {
        "lipatan": N_LIPATAN, "seed": SEED, "tau_det": TAU_DET.tolist(), "tau_taut": TAU_TAUT.tolist(),
        "tambah_simpan": TAMBAH_SIMPAN.tolist(), "offset_kelas": OFFSET_KELAS.tolist(),
        "iou_kembar": IOU_KEMBAR, "iou_nms": IOU_NMS, "iou_cocok": IOU_COCOK, "lantai_conf": LANTAI_CONF,
    }, "korpus": {}}
    for korpus in args.korpus:
        mulai = time.time()
        pohon, cacah = muat_korpus(args, korpus)
        latih = [pohon[n] for n in sorted(cacah["train"]) + sorted(cacah["val"]) if n in pohon]
        model, info_latih = latih_model_pasangan(latih)
        print(f"[{korpus}] model pasangan: {info_latih}, {time.time() - mulai:.0f} s", flush=True)

        beda_json = sum(int(not np.array_equal(cacah["test"][n], pohon[n]["cacah_json"]))
                        for n in cacah["test"] if n in pohon)
        hasil_korpus: dict = {"model_pasangan": info_latih, "pohon_uji_beda_cacah_json": beda_json,
                              "sebaran_kemunculan": sebaran_kemunculan(pohon), "detektor": {}}

        daftar_uji = sorted(n for n in cacah["test"] if n in pohon)
        y_uji = np.array([cacah["test"][n] for n in daftar_uji])
        hasil_korpus["kotak_acuan"] = analisis_kotak_acuan(model, pohon, daftar_uji, y_uji)
        print(f"[{korpus}] kotak acuan: {hasil_korpus['kotak_acuan']['dedup']['mae_makro']} dedup vs "
              f"{hasil_korpus['kotak_acuan']['koefisien_k_perkelas']['mae_makro']} koefisien", flush=True)

        semua_deteksi = []
        for slug, label in DETEKTOR + [("fusi", "Fusi 3 detektor")]:
            if slug == "fusi":
                deteksi = fusi_deteksi(semua_deteksi, daftar)
                prediksi_koef = prediksi_koefisien_dari_deteksi(deteksi)
            else:
                path = sumber_dump(args.project_root, korpus, slug)
                prediksi_koef = muat_prediksi(path)
            n, y, daftar = matriks_hitung(prediksi_koef, cacah["test"], TAU_GRID)
            assert all(p in pohon for p in daftar), "pohon uji tanpa JSON acuan"
            if slug != "fusi":
                deteksi = muat_deteksi(path, pohon, daftar)
                semua_deteksi.append(deteksi)
            hasil = {"n_pohon": len(daftar), "koefisien": {}, "prediksi_per_pohon": {"pohon": daftar}}
            prediksi = {}
            for metode in METODE_KOEFISIEN:
                prediksi[metode] = lipat_silang_koefisien(n, y, metode)
                hasil["koefisien"][metode] = metrik_pohon(prediksi[metode], y)

            komponen, daftar_taut = tabel_dedup(model, deteksi, pohon, daftar, tautan_acuan=False)
            prediksi["dedup"], pilihan = lipat_silang_dedup(komponen, daftar_taut, y)
            prediksi["dedup_tanpa_offset"], _ = lipat_silang_dedup(komponen, daftar_taut, y, pakai_offset=False)
            komp_acuan, taut_acuan = tabel_dedup(model, deteksi, pohon, daftar, tautan_acuan=True)
            prediksi["dedup_tautan_acuan"], _ = lipat_silang_dedup(komp_acuan, taut_acuan, y)

            for nama in ["dedup", "dedup_tanpa_offset", "dedup_tautan_acuan"]:
                hasil[nama] = metrik_pohon(prediksi[nama], y)
            hasil["dedup_pilihan_per_lipatan"] = pilihan
            modus = max({(p["tau_det"], p["tau_simpan"]) for p in pilihan},
                        key=lambda t: sum((p["tau_det"], p["tau_simpan"]) == t for p in pilihan))
            hasil["urai_galat_tautan_acuan"] = urai_galat(deteksi, pohon, daftar, *modus)
            hasil["bootstrap_dedup_minus_koefisien"] = {
                metode: bootstrap_selisih(prediksi["dedup"], prediksi[metode], y) for metode in METODE_KOEFISIEN
            }
            hasil["prediksi_per_pohon"].update({k: v.astype(int).tolist() for k, v in prediksi.items()})
            hasil["prediksi_per_pohon"]["acuan"] = y.astype(int).tolist()
            hasil_korpus["detektor"][label] = hasil
            print(f"[{korpus}] {label}: dedup {hasil['dedup']['mae_makro']} | k_perkelas "
                  f"{hasil['koefisien']['k_perkelas']['mae_makro']} | tautan acuan "
                  f"{hasil['dedup_tautan_acuan']['mae_makro']} | {time.time() - mulai:.0f} s", flush=True)
        ringkasan["korpus"][korpus] = hasil_korpus

    args.keluaran.parent.mkdir(parents=True, exist_ok=True)
    args.keluaran.write_text(json.dumps(ringkasan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[selesai] {args.keluaran}")


if __name__ == "__main__":
    main()
