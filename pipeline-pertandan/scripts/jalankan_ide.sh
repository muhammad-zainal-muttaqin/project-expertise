#!/usr/bin/env bash
# Rantai sequential untuk implementasi IDEA.md sec.4 di korpus 953.
#
# Runner dipakai di sini karena seluruh langkahnya training GPU panjang
# (puluhan menit sampai berjam-jam) -- sesuai ../CLAUDE.md "Runner hanya untuk
# training panjang". Langkah pendek/eval TIDAK masuk sini.
#
# Urutan sengaja begini:
#   1-3  prasyarat re-ID (cache potongan + embedding penuh + dua fold)
#   4    PT-E-016 GNN        <- nilai tertinggi, menyerang akar masalah (cakupan 29%)
#   5    PT-E-014 KONTROL    <- resnet18+ce, HARUS mereproduksi PT-E-012
#   6-8  PT-E-014/015 sel baru
#
# Kontrol di langkah 5 sengaja SEBELUM sel baru: kalau ia tidak mereproduksi
# PT-E-012, implementasi ulangnya yang salah dan angka sel 6-8 tidak boleh
# dipercaya.
set -u
SUB=/workspace/project-expertise/pipeline-pertandan
PY=/workspace/project-expertise/.venv/bin/python
LOG=$SUB/logs; mkdir -p "$LOG" "$SUB/runs" "$SUB/results"
STATUS=$LOG/status.jsonl

jalankan() {           # jalankan <nama> <perintah...>
  local nama=$1; shift
  local mulai; mulai=$(date -u +%FT%TZ); local t0; t0=$(date +%s)
  echo "{\"langkah\":\"$nama\",\"keadaan\":\"mulai\",\"waktu\":\"$mulai\"}" >> "$STATUS"
  echo "=== [$nama] mulai $mulai ==="
  "$@" > "$LOG/$nama.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "{\"langkah\":\"$nama\",\"keadaan\":\"$([ $rc -eq 0 ] && echo selesai || echo GAGAL)\",\"rc\":$rc,\"detik\":$dt}" >> "$STATUS"
  echo "=== [$nama] rc=$rc setelah ${dt}s ==="
  return $rc
}

cd "$SUB/.." || exit 1
: > "$STATUS"

# ---- prasyarat re-ID (dipakai fitur penaut; cache potongan dipakai modul C) ----
jalankan reid_penuh $PY $SUB/scripts/reid_pertandan.py --epoch 30            || exit 1
jalankan reid_fold0 $PY $SUB/scripts/reid_pertandan.py --epoch 30 --fold 0 --nfold 2 --tag _f0 || exit 1
jalankan reid_fold1 $PY $SUB/scripts/reid_pertandan.py --epoch 30 --fold 1 --nfold 2 --tag _f1 || exit 1

# ---- PT-E-016: penaut GNN (IDEA.md sec.4 butir 3) ----
jalankan pt_e_016_gnn $PY $SUB/scripts/gnn_penaut.py --epoch 40

# ---- PT-E-014/015: modul C, dua faktor (IDEA.md sec.4 butir 1 dan 2) ----
# sel kontrol dulu: harus mereproduksi PT-E-012 (C2_vs_C1 -1,21 / C3_vs_C2 -3,06)
jalankan pt_e_014_kontrol_resnet18_ce   $PY $SUB/scripts/c_backbone_ordinal.py --backbone resnet18      --loss ce    --epoch 25
jalankan pt_e_014_convnext_ce           $PY $SUB/scripts/c_backbone_ordinal.py --backbone convnext_tiny --loss ce    --epoch 25
jalankan pt_e_015_resnet18_coral        $PY $SUB/scripts/c_backbone_ordinal.py --backbone resnet18      --loss coral --epoch 25
jalankan pt_e_015_convnext_coral        $PY $SUB/scripts/c_backbone_ordinal.py --backbone convnext_tiny --loss coral --epoch 25

echo "=== RANTAI SELESAI ==="
echo "{\"langkah\":\"RANTAI\",\"keadaan\":\"selesai\"}" >> "$STATUS"
