# Log Eksperimen — Volume 2 (append-only)

Aturan (dari `CLAUDE.md`): satu entri = satu hipotesis falsifiable. Append-only
— entri lama tidak pernah diedit; koreksi ditulis sebagai entri baru yang
mereferensikan entri yang dikoreksi. Hasil negatif dicatat dengan bobot yang
sama dengan hasil positif. Setiap angka harus terlacak ke skrip/JSON/log.

Penomoran: **`V2-E-0xx`**, mulai dari `V2-E-001`. Sengaja terpisah dari
`E-0xx`/`F-0xx` Volume 1 (repo/riwayat berbeda; `E-021` sudah punya arti
spesifik di Volume 1, `V2-E-0xx` menghindari tabrakan referensi silang).

Bukti mentah per entri disimpan di `results/V2-E-0xx/`.

## Template

```markdown
## V2-E-0xx — <satu kalimat hipotesis falsifiable>

**Tanggal:** YYYY-MM-DD
**Hipotesis:** <pernyataan falsifiable, satu saja>
**Dataset & split:** <dataset persis + path split>
**Metode:** <skrip + invokasi CLI persis + identitas bobot/commit>
**Hasil:** <angka apa adanya, tanpa dibungkus>
**Sumber:** <path JSON/CSV/log yang membuktikan tiap angka>
**Verdict:** CONFIRMED | FALSIFIED | INCONCLUSIVE
```

---

<!-- Entri berikutnya ditambahkan di bawah baris ini, tidak pernah menyisip di atas. -->
