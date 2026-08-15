#!/usr/bin/env bash
# Pipeline matriks monocular-depth (V2-E-027 dst) — jalan tanpa ditunggui.
#
# Setiap langkah IDEMPOTEN: kalau keluarannya sudah ada, dilewati. Jadi skrip
# ini aman dijalankan ulang kapan saja — termasuk setelah sesi terputus atau
# pod di-restart, yang di workspace ini sudah pernah terjadi.
#
# Dua GERBANG memblokir pekerjaan mahal:
#   GERBANG A  reproduksi mAP50 0,5435 pada bobot sel 5 dengan evaluator yang
#              sama. Split 953 asli yang dipakai v2repro sudah hilang; kalau
#              rekonstruksinya salah, angka sel 6 tidak sebanding dengan apa
#              pun. Gagal di sini = SELURUH pipeline berhenti.
#   GERBANG B  kelayakan 5 kanal di ultralytics 8.4.103 (4 kanal sudah
#              terbukti, 5 belum pernah). Gagal di sini hanya membatalkan
#              sel 4; sel 6 dan sel 3 tetap jalan.
#
# Urutan sengaja menaruh sel 6 lebih dulu: itu satu-satunya sel dengan daya
# statistik memadai (test 2.612 kotak vs 410 di split 352).
#
# Jalankan terlepas dari sesi:
#   cd /workspace/project-expertise
#   setsid nohup bash scripts/jalankan_matriks.sh > logs/matriks.log 2>&1 &

set -uo pipefail

AKAR=/workspace/project-expertise
PY="$AKAR/.venv/bin/python"
LOG="$AKAR/logs"
mkdir -p "$LOG"

# Resep matched — SAMA untuk semua sel, disalin dari args sel 1/2/5 yang sudah
# diverifikasi identik satu sama lain. Jangan diubah sebagian.
EPOCHS=60; PATIENCE=60; IMGSZ=1280; BATCH=4; SEED=42; BOBOT_AWAL=yolo26l.pt

lapor() { echo "[$(date '+%F %T')] $*"; }
mati()  { lapor "FATAL: $*"; exit 1; }

jalan() {  # jalan <berkas-penanda> <nama-langkah> <perintah...>
  local penanda="$1" nama="$2"; shift 2
  if [ -e "$penanda" ]; then lapor "LEWAT  $nama (sudah ada: $penanda)"; return 0; fi
  lapor "MULAI  $nama"
  if "$@" >> "$LOG/$nama.log" 2>&1; then
    lapor "SELESAI $nama"
    return 0
  fi
  lapor "GAGAL  $nama — lihat $LOG/$nama.log"
  return 1
}

cd "$AKAR" || mati "tidak bisa masuk $AKAR"

# ---- KUNCI INSTANSI TUNGGAL ----------------------------------------------
# Ini bukan kehati-hatian berlebihan; kegagalannya sudah terjadi. Dua instansi
# runner sempat jalan bersamaan pada 2026-08-14 dan melatih dua sel sekaligus
# di satu GPU 16 GB. Akibatnya:
#   - sel 6 mati CUDA OOM di 91% epoch 1 setelah berjalan 7,5 menit;
#   - ultralytics MENURUNKAN batch sendiri (sel 3 -> 1, sel 4 -> 2) lalu tetap
#     lanjut, sementara args.yaml tetap mencatat batch: 4.
# Yang terakhir itu racunnya: resep tercatat berbeda dari resep yang benar-benar
# dijalankan, jadi angkanya tidak sebanding TANPA jejak apa pun di metadata.
KUNCI="$AKAR/.matriks.lock"
exec 9>"$KUNCI" || mati "tidak bisa membuat kunci $KUNCI"
if ! flock -n 9; then
  mati "runner lain sedang jalan (kunci $KUNCI dipegang). Satu GPU = satu training."
fi
echo $$ >&9

# GPU harus benar-benar kosong sebelum mulai — sisa proses yatim dari sesi yang
# terbunuh pernah menyisakan 14 dari 16 GB terpakai.
sisa_vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
if [ "${sisa_vram:-0}" -gt 1000 ]; then
  mati "VRAM sudah terpakai ${sisa_vram} MiB sebelum mulai — ada proses lain. Hentikan dulu."
fi

# KUNCI EKSKLUSIF — satu instance saja, titik.
# Ini bukan kehati-hatian teoretis: 2026-08-14 dua instance sempat jalan
# bersamaan, masing-masing menyalakan training di GPU yang sama, dan keduanya
# saling mendorong sampai CUDA OOM. Korbannya sel 6 — satu-satunya sel yang
# punya daya statistik memadai — mati di epoch 1. GPU ini hanya muat SATU
# training @1280 batch 4 (sel 6 sendirian sudah 12 GB dari 15,6 GB).
exec 9>"$AKAR/.matriks.lock"
if ! flock -n 9; then
  lapor "SUDAH ADA instance jalankan_matriks.sh yang jalan — instance ini berhenti."
  lapor "Cek dengan: pgrep -af jalankan_matriks"
  exit 0
fi
lapor "kunci didapat (PID $$)"

# ---------------------------------------------------------------- 1. dataset
jalan /workspace/d352_rgbmono/data.yaml bangun_sel3 \
  "$PY" scripts/buat_dataset_nch.py --dataset 352 --kanal mono \
        --out /workspace/d352_rgbmono || mati "sel 3 tidak terbangun"

jalan /workspace/d352_rgbedgemono/data.yaml bangun_sel4 \
  "$PY" scripts/buat_dataset_nch.py --dataset 352 --kanal edge mono \
        --out /workspace/d352_rgbedgemono || lapor "sel 4 tidak terbangun — sel 4 dilewati nanti"

jalan /workspace/d953_rgbmono/data.yaml bangun_sel6 \
  "$PY" scripts/buat_dataset_nch.py --dataset 953 --kanal mono \
        --out /workspace/d953_rgbmono || mati "sel 6 tidak terbangun"

# ------------------------------------------------------------- 2. GERBANG A
# Sekaligus menghasilkan dump prediksi sel 5 yang selama ini tidak ada,
# sehingga CI berpasangan sel 6 vs 5 bisa dihitung.
jalan results/eval_sel5_953_rgb_test.json gerbang_a_reproduksi_sel5 \
  "$PY" scripts/eval_nch.py \
        --bobot models/yolo26l_e60_i1280_v2repro/best.pt \
        --ds-root /workspace/SawitMVC-YOLO --tata-letak images_split \
        --split test --nama sel5_953_rgb --harap-map50 0.5435 \
  || mati "GERBANG A gagal — split/protokol 953 tidak cocok. Tidak ada angka yang boleh dibandingkan."

# ------------------------------------------------------------- 3. GERBANG B
LAYAK5=1
if [ -f /workspace/d352_rgbedgemono/data.yaml ]; then
  jalan runs/uji5ch/results.csv gerbang_b_kelayakan_5kanal \
    "$PY" scripts/train_yolo_4ch_screening.py \
          --data /workspace/d352_rgbedgemono/data.yaml \
          --epochs 1 --patience 1 --name uji5ch \
    || { LAYAK5=0; lapor "GERBANG B gagal — 5 kanal tidak didukung, sel 4 dibatalkan"; }
else
  LAYAK5=0
fi

# ------------------------------------------------------------- 4. training
# Jumlah epoch yang BENAR-BENAR selesai, dibaca dari results.csv.
#
# JANGAN pakai keberadaan weights/best.pt sebagai penanda selesai: ultralytics
# menulis best.pt setiap epoch, jadi file itu sudah ada sejak epoch 1. Versi
# skrip ini sebelumnya memakainya sebagai penanda `jalan`, yang berarti sebuah
# training yang mati di epoch 30 akan dianggap selesai saat runner dinyalakan
# ulang, lalu dievaluasi dan masuk matriks sebagai hasil resmi — salah tanpa
# gejala apa pun. results.csv hanya bertambah satu baris per epoch yang tuntas.
epoch_selesai() {  # epoch_selesai <nama> -> jumlah epoch tuntas
  local f="runs/$1/results.csv"
  [ -f "$f" ] || { echo 0; return; }
  awk 'NR>1 && NF' "$f" | wc -l
}

latih_dan_eval() {  # latih_dan_eval <nama> <data.yaml> <ds-root> <tata-letak>
  local nama="$1" data="$2" root="$3" tata="$4" ep
  ep=$(epoch_selesai "$nama")

  if [ -f "runs/$nama/DIHENTIKAN_LEBIH_AWAL" ]; then
    # Penghentian yang disengaja oleh pengguna. Tanpa cabang ini, gerbang
    # 60-epoch di bawah akan menganggapnya run terputus dan me-resume-nya.
    lapor "LEWAT  latih_$nama — dihentikan lebih awal atas keputusan pengguna ($ep/$EPOCHS epoch)"
  elif [ "$ep" -ge "$EPOCHS" ]; then
    lapor "LEWAT  latih_$nama (sudah tuntas $ep/$EPOCHS epoch)"
  elif [ "$ep" -gt 0 ] && [ -f "runs/$nama/weights/last.pt" ]; then
    # Run terputus (sesi mati / pod restart) tapi checkpoint-nya utuh.
    # Resume, bukan latih ulang dari nol: mengulang berarti membuang berjam-jam
    # GPU, dan resep tetap terkunci karena ultralytics membacanya dari args.yaml.
    lapor "LANJUT latih_$nama — resume dari epoch $ep/$EPOCHS"
    if ! "$PY" scripts/train_yolo_4ch_screening.py \
           --resume "runs/$nama/weights/last.pt" >> "$LOG/latih_$nama.log" 2>&1; then
      lapor "resume $nama gagal — lihat $LOG/latih_$nama.log"; return 1
    fi
    lapor "SELESAI latih_$nama (resume)"
  else
    # Tidak lewat `jalan`: kedua kondisi "sudah selesai" dan "bisa di-resume"
    # sudah ditangani di atas, jadi di titik ini training memang harus mulai.
    lapor "MULAI  latih_$nama"
    if ! "$PY" scripts/train_yolo_4ch_screening.py --data "$data" \
           --epochs $EPOCHS --patience $PATIENCE --imgsz $IMGSZ --batch $BATCH \
           --seed $SEED --weights $BOBOT_AWAL --name "$nama" \
           >> "$LOG/latih_$nama.log" 2>&1; then
      lapor "training $nama gagal — lihat $LOG/latih_$nama.log"; return 1
    fi
    lapor "SELESAI latih_$nama"
  fi

  # Gerbang terakhir sebelum eval: hitung ulang dari disk, bukan percaya exit code.
  ep=$(epoch_selesai "$nama")
  if [ "$ep" -lt "$EPOCHS" ] && [ ! -f "runs/$nama/DIHENTIKAN_LEBIH_AWAL" ]; then
    lapor "FATAL $nama: baru $ep/$EPOCHS epoch tuntas — TIDAK dievaluasi."
    lapor "       jalankan runner lagi untuk melanjutkan (resume otomatis)."
    return 1
  fi

  # Ultralytics menurunkan batch sendiri saat OOM lalu TETAP LANJUT, sementara
  # args.yaml tetap mencatat batch semula. Run seperti itu tidak sebanding
  # dengan sel pembandingnya dan tidak boleh masuk matriks diam-diam.
  if grep -q "Reducing to batch" "$LOG/latih_$nama.log" 2>/dev/null; then
    local turun; turun=$(grep -o "Reducing to batch=[0-9]*" "$LOG/latih_$nama.log" | tail -1)
    mv "runs/$nama" "runs/${nama}_INVALID_batch_turun" 2>/dev/null
    lapor "FATAL $nama: batch diturunkan otomatis ($turun) padahal resep menuntut $BATCH."
    lapor "       run dipindah ke runs/${nama}_INVALID_batch_turun, TIDAK dievaluasi."
    return 1
  fi
  jalan "results/eval_${nama}_test.json" "eval_$nama" \
    "$PY" scripts/eval_nch.py --bobot "runs/$nama/weights/best.pt" \
          --ds-root "$root" --tata-letak "$tata" --split test --nama "$nama" \
    || { lapor "eval $nama gagal"; return 1; }
}

latih_dan_eval sel6_953_rgbmono /workspace/d953_rgbmono/data.yaml \
               /workspace/d953_rgbmono images_split

latih_dan_eval sel3_352_rgbmono /workspace/d352_rgbmono/data.yaml \
               /workspace/d352_rgbmono images_split

if [ "$LAYAK5" = "1" ]; then
  latih_dan_eval sel4_352_rgbedgemono /workspace/d352_rgbedgemono/data.yaml \
                 /workspace/d352_rgbedgemono images_split
else
  lapor "sel 4 DILEWATI (gerbang B gagal)"
fi

# ------------------------------------------------------------- 5. bootstrap
# Hanya pasangan yang kedua .npz-nya benar-benar ada.
boot() {  # boot <keluaran> <gt-root> <tata-letak> <npz-a> <npz-b> <nama-a> <nama-b>
  local out="$1" root="$2" tata="$3" a="$4" b="$5" na="$6" nb="$7"
  if [ ! -f "$a" ] || [ ! -f "$b" ]; then
    lapor "LEWAT bootstrap $na vs $nb — dump belum lengkap"; return 0
  fi
  jalan "$out" "boot_${na}_vs_${nb}" \
    "$PY" scripts/bootstrap_nch.py --gt-root "$root" --tata-letak "$tata" \
          --split test --sumber "$a" "$b" --nama "$na" "$nb" --out "$out" \
    || lapor "bootstrap $na vs $nb gagal"
}

# GT diambil dari dataset ASLI (bukan dataset N-kanal turunan): labelnya sama
# persis, dan ini membuat kedua lengan dibandingkan terhadap sumber yang sama.
boot results/boot_sel6_vs_sel5.json /workspace/SawitMVC-YOLO images_split \
     results/pred_sel6_953_rgbmono_test.npz results/pred_sel5_953_rgb_test.npz \
     sel6_mono sel5_rgb
boot results/boot_sel3_vs_sel1.json /workspace/SawitMVC-Depth-YOLO split_images \
     results/pred_sel3_352_rgbmono_test.npz results/pred_rgb352_test.npz \
     sel3_mono sel1_rgb
boot results/boot_sel4_vs_sel2.json /workspace/SawitMVC-Depth-YOLO split_images \
     results/pred_sel4_352_rgbedgemono_test.npz results/pred_edge_test.npz \
     sel4_edgemono sel2_edge

lapor "PIPELINE SELESAI"
lapor "hasil: results/eval_sel*_test.json, results/boot_*.json"
