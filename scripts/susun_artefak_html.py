"""Mengubah laporan Markdown menjadi satu halaman HTML untuk artefak.

Halaman dibangkitkan dari berkas Markdown yang sama dengan PDF, sehingga seluruh
angkanya identik. Gambar dirujuk relatif pada folder `assets/`.

Pemakaian:
    python scripts/susun_artefak_html.py
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

GAYA = """
<style>
  :root {
    --ground: #F7F8F9;
    --surface: #FFFFFF;
    --ink: #101820;
    --ink-soft: #46535F;
    --ink-mute: #6F7C88;
    --line: #DFE4E8;
    --line-soft: #EDF0F2;
    --biru: #0072B2;
    --jingga: #E69F00;
    --hijau: #009E73;
    --sorot: #EAF3F9;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #0E1418;
      --surface: #151D23;
      --ink: #E8EDF1;
      --ink-soft: #B3BFC9;
      --ink-mute: #8494A1;
      --line: #27333C;
      --line-soft: #1D262D;
      --biru: #55B3E8;
      --jingga: #F0B848;
      --hijau: #3FC79B;
      --sorot: #14252F;
    }
  }
  :root[data-theme="dark"] {
    --ground: #0E1418;
    --surface: #151D23;
    --ink: #E8EDF1;
    --ink-soft: #B3BFC9;
    --ink-mute: #8494A1;
    --line: #27333C;
    --line-soft: #1D262D;
    --biru: #55B3E8;
    --jingga: #F0B848;
    --hijau: #3FC79B;
    --sorot: #14252F;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, Segoe UI, sans-serif;
    font-size: 15px;
    line-height: 1.6;
  }
  .bingkai { display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 40px;
             max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 0 72px; }
  .navigasi { position: sticky; top: env(safe-area-inset-top, 0px); align-self: start;
              padding-block: 40px 24px; max-height: 100vh; overflow-y: auto; }
  .navigasi .merek { font-family: "Source Serif 4", Georgia, serif; font-size: 19px;
                     font-weight: 600; line-height: 1.25; margin: 0 0 6px; }
  .navigasi .sub { font-size: 12.5px; color: var(--ink-mute); margin: 0 0 20px; }
  .navigasi ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
  .navigasi a { display: flex; gap: 10px; padding: 5px 8px; border-radius: 4px;
                color: var(--ink-soft); text-decoration: none; font-size: 13px; }
  .navigasi a:hover { background: var(--line-soft); color: var(--ink); }
  .navigasi a.aktif { background: var(--sorot); color: var(--ink); font-weight: 600;
                      box-shadow: inset 2px 0 0 var(--biru); }
  .navigasi a .no { color: var(--ink-mute); font-variant-numeric: tabular-nums; min-width: 16px; }
  .navigasi a.aktif .no { color: var(--biru); }
  main { padding-block: 40px 0; min-width: 0; }
  .kepala { border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 30px; }
  .eyebrow { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
             letter-spacing: 0.12em; text-transform: uppercase; color: var(--biru); margin: 0 0 10px; }
  h1 { font-family: "Source Serif 4", Georgia, serif; font-size: clamp(27px, 4vw, 38px);
       line-height: 1.15; font-weight: 600; margin: 0 0 12px; text-wrap: balance; letter-spacing: -0.01em; }
  .ikhtisar { color: var(--ink-soft); font-size: 15.5px; max-width: 62ch; margin: 0; }
  h2 { font-family: "Source Serif 4", Georgia, serif; font-size: 23px; font-weight: 600;
       margin: 52px 0 14px; scroll-margin-top: 24px; text-wrap: balance; }
  h3 { font-family: "Source Serif 4", Georgia, serif; font-size: 18px; font-weight: 600;
       margin: 32px 0 10px; color: var(--ink-soft); scroll-margin-top: 24px; }
  p { max-width: 68ch; color: var(--ink-soft); margin: 0 0 14px; }
  .tabel-bungkus { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px;
                   background: var(--surface); margin: 0 0 20px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th, td { padding: 7px 12px; text-align: left; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
  thead th { position: sticky; top: 0; background: var(--surface); font-size: 11.5px;
             letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-mute);
             font-weight: 600; border-bottom: 1px solid var(--line); }
  tbody tr:last-child td { border-bottom: 0; }
  tbody tr:hover { background: var(--line-soft); }
  td strong { font-weight: 600; color: var(--ink); }
  td code, p code, li code { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
                             background: var(--line-soft); padding: 1px 5px; border-radius: 3px; }
  figure { margin: 0 0 24px; }
  figure img { width: 100%; max-width: 100%; display: block; background: #FFFFFF;
               border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
  figcaption { font-size: 12.5px; color: var(--ink-mute); margin-top: 8px;
               font-family: "IBM Plex Mono", ui-monospace, monospace; }
  .kartu-baris { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                 gap: 12px; margin: 0 0 26px; }
  .kartu { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
  .kartu .judul { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
                  letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-mute); margin: 0 0 6px; }
  .kartu .angka { font-family: "Source Serif 4", Georgia, serif; font-size: 27px; font-weight: 600;
                  line-height: 1.1; font-variant-numeric: tabular-nums; margin: 0; }
  .kartu .ket { font-size: 12.5px; color: var(--ink-mute); margin: 4px 0 0; }
  .kartu.utama { box-shadow: inset 3px 0 0 var(--biru); }
  .kartu.dua { box-shadow: inset 3px 0 0 var(--jingga); }
  .kartu.tiga { box-shadow: inset 3px 0 0 var(--hijau); }
  footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line);
           font-size: 12.5px; color: var(--ink-mute); }
  a { color: var(--biru); }
  :focus-visible { outline: 2px solid var(--biru); outline-offset: 2px; }
  @media (max-width: 900px) {
    .bingkai { grid-template-columns: 1fr; gap: 0; padding-inline: 16px; }
    .navigasi { position: static; max-height: none; padding-block: 28px 0; }
    .navigasi ol { flex-direction: row; flex-wrap: wrap; }
    main { padding-block: 24px 0; }
  }
  @media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
</style>
"""

SKRIP = """
<script>
  (function () {
    var tautan = Array.prototype.slice.call(document.querySelectorAll('.navigasi a'));
    var bagian = tautan.map(function (a) { return document.querySelector(a.getAttribute('href')); });
    function sorot() {
      var posisi = window.scrollY + 120, aktif = 0;
      bagian.forEach(function (el, i) { if (el && el.offsetTop <= posisi) { aktif = i; } });
      tautan.forEach(function (a, i) { a.classList.toggle('aktif', i === aktif); });
    }
    window.addEventListener('scroll', sorot, { passive: true });
    sorot();
  })();
</script>
"""


def inline(teks: str) -> str:
    teks = html.escape(teks, quote=False)
    teks = re.sub(r"`([^`]+)`", r"<code>\1</code>", teks)
    teks = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", teks)
    teks = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", teks)
    # notasi matematika sederhana
    teks = re.sub(r"\$([^$]+)\$", lambda m: "<em>" + bersih_matematika(m.group(1)) + "</em>", teks)
    return teks


def bersih_matematika(m: str) -> str:
    ganti = {
        r"\text{--}": "–", r"\text{conf}": "conf", r"\tau": "τ", r"\hat{y}": "ŷ",
        r"\operatorname{round}": "round", r"\cdot": "·", r"\ge": "≥", r"\le": "≤",
    }
    for a, b in ganti.items():
        m = m.replace(a, b)
    m = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", m)
    m = re.sub(r"_(\w)", r"<sub>\1</sub>", m)
    m = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", m)
    return m.replace("\\", "")


def sel(baris: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<!\\)\|", baris.strip())[1:-1]]


def ubah(md: str, gambar_awalan: str) -> tuple[str, list[tuple[str, str]]]:
    baris = md.splitlines()
    keluaran: list[str] = []
    daftar_bagian: list[tuple[str, str]] = []
    i = 0
    n_bagian = 0
    while i < len(baris):
        b = baris[i]
        if b.startswith("# "):
            i += 1
            continue
        if b.startswith("## ") or b.startswith("### "):
            taraf = 2 if b.startswith("## ") else 3
            judul = b[taraf + 1:].strip()
            if taraf == 2:
                n_bagian += 1
                anchor = f"bagian-{n_bagian}"
                daftar_bagian.append((anchor, judul))
                keluaran.append(f'<h2 id="{anchor}">{inline(judul)}</h2>')
            else:
                keluaran.append(f"<h3>{inline(judul)}</h3>")
            i += 1
            continue
        if b.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", b)
            if m:
                nama = Path(m.group(2)).name
                keluaran.append(
                    f'<figure><img src="{gambar_awalan}{nama}" alt="{html.escape(m.group(1))}">'
                    f"<figcaption>{inline(m.group(1))}</figcaption></figure>"
                )
            i += 1
            continue
        if b.startswith("|") and i + 1 < len(baris) and set(baris[i + 1].replace("|", "").strip()) <= {"-", ":", " "}:
            kepala = sel(b)
            i += 2
            isi = []
            while i < len(baris) and baris[i].startswith("|"):
                isi.append(sel(baris[i]))
                i += 1
            th = "".join(f"<th>{inline(x)}</th>" for x in kepala)
            tr = "".join("<tr>" + "".join(f"<td>{inline(x)}</td>" for x in r) + "</tr>" for r in isi)
            keluaran.append(f'<div class="tabel-bungkus"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>')
            continue
        if b.strip():
            keluaran.append(f"<p>{inline(b.strip())}</p>")
        i += 1
    return "\n".join(keluaran), daftar_bagian


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", default="docs/LAPORAN-KINERJA-PENCACAHAN-2026-09-16.md")
    ap.add_argument("--keluaran", default="docs/assets/artefak/laporan.html")
    ap.add_argument("--gambar-awalan", default="assets/")
    arg = ap.parse_args()

    md = Path(arg.sumber).read_text(encoding="utf-8")
    isi, bagian = ubah(md, arg.gambar_awalan)

    # kartu ringkasan diambil dari tabel ringkasan eksekutif
    kartu = []
    for baris in md.splitlines():
        if re.match(r"^\| (953|763|1716) \| RF-DETR-L \|", baris):
            k = sel(baris)
            kartu.append((k[0], k[5], k[2], k[7]))
        if len(kartu) == 3:
            break
    kelas = ["utama", "dua", "tiga"]
    kartu_html = "".join(
        f'<div class="kartu {kelas[i]}"><p class="judul">Korpus {k[0]}</p>'
        f'<p class="angka">{k[1]}</p>'
        f'<p class="ket">MAE makro · mAP50 {k[2]} · akurasi ±1 {k[3]}</p></div>'
        for i, k in enumerate(kartu)
    )

    nav = "".join(
        f'<li><a href="#{a}"><span class="no">{i + 1}</span><span>{html.escape(j.split(". ", 1)[-1])}</span></a></li>'
        for i, (a, j) in enumerate(bagian)
    )

    halaman = f"""<title>Kinerja Pencacahan Sawit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
{GAYA}
<div class="bingkai">
  <nav class="navigasi">
    <p class="merek">Kinerja Pencacahan Sawit</p>
    <p class="sub">Tiga korpus latih, tiga detektor, 16 September 2026</p>
    <ol>{nav}</ol>
  </nav>
  <main>
    <header class="kepala">
      <p class="eyebrow">V2-E-050 sampai V2-E-050f</p>
      <h1>Laporan Kinerja per Korpus Latih: Deteksi dan Pencacahan</h1>
      <p class="ikhtisar">Perbandingan tiga arsitektur detektor pada korpus 953, 763, dan gabungan 1716,
      beserta kalibrasi koefisien pengali per kelas untuk pencacahan tandan per pohon.</p>
    </header>
    <div class="kartu-baris">{kartu_html}</div>
    {isi}
    <footer>Seluruh angka dibangkitkan dari berkas JSON hasil melalui
    <code>scripts/susun_laporan_pencacahan.py</code>, lalu halaman ini disusun oleh
    <code>scripts/susun_artefak_html.py</code>.</footer>
  </main>
</div>
{SKRIP}
"""
    keluaran = Path(arg.keluaran)
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    keluaran.write_text(halaman, encoding="utf-8")
    print(f"Ditulis: {keluaran} ({len(halaman)} karakter, {len(bagian)} bagian)")


if __name__ == "__main__":
    main()
