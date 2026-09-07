---
name: markdown-ke-pdf
description: >-
  Renders a Markdown document to a typeset PDF via LaTeX and tectonic, without pandoc. Built for dense Indonesian technical documents: wide comparison tables with automatic column-width balancing, inline math with Indonesian decimal commas, GitHub alert callouts, superscript footnote markers, fenced code blocks, and nested lists. Activate when the user asks to render, typeset, export, or convert a Markdown file (report, recap, laporan, dokumentasi) into PDF or LaTeX.
---

# Merender Markdown menjadi PDF

Skill ini mengubah berkas Markdown menjadi LaTeX, lalu mengompilasinya menjadi
PDF memakai **tectonic**. Pandoc tidak dibutuhkan.

## 1. Prasyarat

| Kebutuhan | Keterangan |
|---|---|
| `tectonic` | Wajib, harus ada pada `PATH`. Paket LaTeX diunduh otomatis pada kompilasi pertama, sehingga jalankan sekali dalam keadaan daring. |
| Python 3.9+ | Hanya memakai pustaka standar. |
| Fonta | Windows memakai Times New Roman dan Consolas apabila tersedia. Selain itu jatuh ke TeX Gyre Termes dan TeX Gyre Cursor yang disediakan bundel tectonic. |

Periksa ketersediaan dengan `tectonic --version`. Apabila tidak ada, jalankan
dengan `--tex-only` dan serahkan berkas `.tex` kepada pengguna.

## 2. Cara Pemakaian

```bash
python .agents/skills/markdown-ke-pdf/scripts/md2pdf.py DOKUMEN.md
```

| Opsi | Fungsi |
|---|---|
| `-o, --outdir DIR` | Direktori keluaran. Baku: sebelah berkas sumber. |
| `--title TEKS` | Judul dokumen. Baku: judul `#` pertama, atau nama berkas. |
| `--running-head TEKS` | Teks kepala halaman. Baku: mengikuti judul. |
| `--toc` | Menyisipkan daftar isi. |
| `--paper` / `--margin` | Baku `a4paper` dan `2cm`. |
| `--fontsize {10,11,12}` | Ukuran fonta dasar. Baku `10`. |
| `--main-font` / `--mono-font` | Menimpa pemilihan fonta otomatis. |
| `--keep-tex` | Mempertahankan berkas `.tex` perantara. |
| `--tex-only` | Berhenti pada tahap LaTeX, tanpa kompilasi. |

Judul `#` pertama dipakai sebagai judul dokumen dan tidak dicetak ulang sebagai
bagian. Judul `##` dan seterusnya menjadi bagian tanpa penomoran otomatis,
sehingga penomoran manual pada sumber tetap terjaga.

## 3. Konstruksi Markdown yang Didukung

| Konstruksi | Hasil |
|---|---|
| Tabel beralinea | `xltabular` dengan lebar kolom terbobot; kolom rata kanan berlebar tetap |
| Matematika `$...$` | Diteruskan apa adanya |
| `> [!NOTE]`, `TIP`, `IMPORTANT`, `WARNING`, `CAUTION` | Kotak sorot berwarna beserta labelnya |
| Kutipan polos `>` | Kotak bergaris tepi kiri |
| Blok kode berpagar | `Verbatim` bergaris tepi kiri |
| Daftar berbutir dan bernomor | `itemize` dan `enumerate`, termasuk sarang |
| `**tebal**`, `*miring*`, `***tebal miring***`, `~~coret~~`, `` `kode` `` | Sesuai padanannya |
| Tautan `[teks](url)` | Hanya teksnya; berkas rujukan tidak dapat dibuka dari PDF |
| Gambar `![alt](path)` | `\includegraphics` selebar kolom apabila berkasnya ada |
| Superskrip `¹`–`⁹` dan `<sup>n</sup>` | `\textsuperscript` |
| Akhiran dua spasi | Pemutus baris |
| `---` | Diabaikan; pemisah bagian sudah ditangani judul |

## 4. Keputusan Tata Letak yang Penting

**Koma desimal dalam mode matematika.** TeX memperlakukan koma sebagai tanda
baca dan menyisipkan spasi setelahnya, sehingga `$0,6569$` tercetak `0, 6569`.
Pengonversi mengurung setiap koma di antara dua angka menjadi `{,}`. Ini wajib
untuk dokumen berbahasa Indonesia yang memakai koma desimal.

**Pembobotan lebar kolom.** Kolom rata kanan (`---:`) dianggap numerik dan diberi
lebar tetap. Kolom teks dibobot menurut panjang isi persentil 80 dan panjang
judulnya, kemudian dinaikkan sampai token terpanjang yang tidak dapat dipatahkan
tetap muat. Token bergaya monospace dihitung 1,45 kali lebih lebar karena fonta
monospace lebih lebar daripada fonta serif pada ukuran yang sama. Tanpa batas
bawah ini, kata panjang seperti `combined1716` akan meluber menabrak kolom
sebelahnya.

**Ukuran fonta tabel** mengikuti jumlah kolom: `\small` sampai 6 kolom,
`\footnotesize` pada 7–8 kolom, `\scriptsize` pada 9 kolom atau lebih. Dengan
pembobotan tersebut, tabel berkolom banyak umumnya tetap muat pada halaman
potret sehingga halaman melintang tidak diperlukan.

## 5. Batasan yang Perlu Disampaikan kepada Pengguna

- Tautan dirender sebagai teks biasa. Rujukan silang antar-bagian dan tautan
  berkas tidak dapat diklik pada PDF.
- Blok matematika `$$...$$` dirender sebaris, bukan sebagai persamaan tersendiri.
- HTML sebarang di dalam Markdown diabaikan, kecuali `<sup>`.
- Tabel bersarang, definisi catatan kaki Markdown (`[^1]`), dan daftar tugas
  (`- [ ]`) tidak ditangani secara khusus.
- Peringatan `Overfull \hbox` di bawah sekitar 10 pt setara kurang dari 4 mm dan
  tidak tampak pada hasil cetak; peringatan itu boleh diabaikan.

## 6. Alur Kerja yang Disarankan

1. Jalankan pengonversi, lalu periksa keluaran tectonic untuk galat dan
   peringatan `Overfull`.
2. Apabila ada peringatan `Overfull` yang besar pada tabel, periksa halaman
   terkait secara visual sebelum menyerahkan hasilnya.
3. Serahkan PDF kepada pengguna, sebutkan jumlah halaman, dan jelaskan setiap
   penyesuaian tata letak yang dilakukan.
