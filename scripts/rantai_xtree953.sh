#!/bin/bash
# Rantai kontrol cross_tree 953: tunggu build -> gerbang TIFF korup -> training.
#
# Ditulis 2026-08-15. Tujuannya menghapus jeda menunggu manual di antara tiga
# langkah yang sudah pasti urutannya. BUKAN runner generik — sengaja satu
# rantai sekali-pakai, karena runner generik di repo ini sudah pernah gagal
# dan aturannya (project-expertise/CLAUDE.md) melarang mengandalkannya lagi.
#
# Tiga gerbang yang menghentikan rantai, masing-masing karena pernah kejadian:
#   1. meta.json harus ada        -> builder yang mati di tengah tidak lolos
#   2. jumlah TIFF harus 3992     -> build parsial tidak lolos
#   3. TIFF korup harus nol       -> ultralytics MELEWATI citra korup diam-diam
#                                    dan tetap "sukses" (39 berkas, V2-E-028)
set -u

AKAR=/workspace/project-expertise
DS=/workspace/d953_rgbmono_xtree
NAMA=xtree953_rgbmono
DIHARAP=3992
LOG="$AKAR/logs"
PY="$AKAR/.venv/bin/python"

cd "$AKAR" || exit 1
lapor() { echo "[$(date -Is)] $*"; }

# --- 1. tunggu build selesai -------------------------------------------------
pid_build=$(pgrep -f 'buat_dataset_nch.py.*d953_rgbmono_xtree' | head -1)
if [ -n "$pid_build" ]; then
  lapor "menunggu build PID $pid_build"
  while kill -0 "$pid_build" 2>/dev/null; do sleep 60; done
fi
lapor "build tidak lagi berjalan"

if [ ! -f "$DS/meta.json" ]; then
  lapor "GAGAL: $DS/meta.json tidak ada — build mati di tengah. Rantai berhenti."
  exit 1
fi

n=$(find "$DS" -name '*.tiff' | wc -l)
if [ "$n" -ne "$DIHARAP" ]; then
  lapor "GAGAL: $n TIFF, diharapkan $DIHARAP — build parsial. Rantai berhenti."
  exit 1
fi
lapor "build OK: $n TIFF, meta.json ada"

# --- 2. gerbang TIFF korup ---------------------------------------------------
lapor "memindai TIFF korup"
"$PY" scripts/perbaiki_tiff_korup.py --root "$DS" \
      --out results/tiff_korup_xtree953.json >> "$LOG/scan_xtree953.log" 2>&1
rusak=$($PY - <<PY
import json
d = json.load(open("results/tiff_korup_xtree953.json"))
print(sum(s["rusak"] for ds in d.values() for s in ds.values()))
PY
)
if [ "$rusak" != "0" ]; then
  lapor "GAGAL: $rusak TIFF korup. Perbaiki dulu (--hapus lalu bangun ulang). Rantai berhenti."
  exit 1
fi
lapor "gerbang korup lolos: 0 rusak"

# --- 3. training -------------------------------------------------------------
# Resep WAJIB identik dengan sel 5 dan sel 6 — kalau tidak, kontrolnya tidak
# mengontrol apa pun.
lapor "MULAI training $NAMA (60 epoch, batch 4, imgsz 1280, seed 42)"
"$PY" scripts/train_yolo_4ch_screening.py \
  --data "$DS/data.yaml" \
  --epochs 60 --patience 60 --imgsz 1280 --batch 4 --seed 42 \
  --weights yolo26l.pt --name "$NAMA" >> "$LOG/latih_$NAMA.log" 2>&1
rc=$?
ep=$(awk 'NR>1 && NF' "runs/$NAMA/results.csv" 2>/dev/null | wc -l)
if [ $rc -ne 0 ]; then
  lapor "training keluar rc=$rc setelah $ep epoch — lihat $LOG/latih_$NAMA.log"
  exit 1
fi
lapor "SELESAI training $NAMA ($ep epoch)"
