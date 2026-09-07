#!/usr/bin/env python3
"""Merender berkas Markdown menjadi LaTeX lalu PDF melalui tectonic.

Pengonversi ini ditulis untuk dokumen teknis berbahasa Indonesia: tabel padat,
matematika sebaris, kotak sorot GitHub, dan catatan kaki superskrip. Lebar kolom
tabel dihitung otomatis agar muat pada halaman potret tanpa tumpang tindih.

    python md2pdf.py DOKUMEN.md
    python md2pdf.py DOKUMEN.md -o keluaran/ --toc --fontsize 11
    python md2pdf.py DOKUMEN.md --tex-only
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BS = chr(92)
SUPO, SUPC = chr(0xE000), chr(0xE001)      # penanda sementara <sup> ... </sup>
SLOT = re.compile(r"@@S(\d+)@@")
BLOCK = re.compile(r"^(\||>|#{1,6}\s|[-*+]\s|\d+[.)]\s|```|~~~|(-{3,}|\*{3,}|_{3,})$)")

SUP = {"\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2070": "0",
       "\u2074": "4", "\u2075": "5", "\u2076": "6",
       "\u2077": "7", "\u2078": "8", "\u2079": "9"}

SPECIAL = {"&": BS + "&", "%": BS + "%", "#": BS + "#",
           "_": BS + "_", "{": BS + "{", "}": BS + "}",
           "$": BS + "$", "^": BS + "textasciicircum{}",
           "~": BS + "textasciitilde{}", BS: BS + "textbackslash{}"}

# Kotak sorot GitHub -> nama lingkungan dan judulnya
ALERTS = {"NOTE": ("kotakcatatan", "Catatan"),
          "TIP": ("kotaksaran", "Saran"),
          "IMPORTANT": ("kotakpenting", "Penting"),
          "WARNING": ("kotakperingatan", "Peringatan"),
          "CAUTION": ("kotakawas", "Perhatian")}

WIN_FONTS = [("Times New Roman", "times.ttf"), ("Cambria", "cambria.ttc"),
             ("Georgia", "georgia.ttf")]
WIN_MONO = [("Consolas", "consola.ttf"), ("Courier New", "cour.ttf")]


# --------------------------------------------------------------------------- #
# Pemilihan fonta
# --------------------------------------------------------------------------- #

def pick_fonts(main, mono):
    """Tentukan fonta utama dan monospace yang tersedia di sistem."""
    if main and mono:
        return main, mono
    if os.name == "nt":
        d = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts"
        main = main or next((n for n, f in WIN_FONTS if (d / f).exists()), None)
        mono = mono or next((n for n, f in WIN_MONO if (d / f).exists()), None)
    # TeX Gyre tersedia di dalam bundel tectonic pada semua platform
    return main or "TeX Gyre Termes", mono or "TeX Gyre Cursor"


def preamble(opt):
    main, mono = pick_fonts(opt.main_font, opt.mono_font)
    head = opt.running_head or opt.title
    return rf"""\documentclass[{opt.fontsize}pt,{opt.paper}]{{article}}
\usepackage{{fontspec}}
\setmainfont{{{main}}}
\setmonofont{{{mono}}}[Scale=0.86]
\usepackage[{opt.paper},margin={opt.margin}]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{array,booktabs,longtable,xltabular,ragged2e}}
\usepackage{{graphicx}}
\usepackage{{fancyvrb}}
\usepackage[dvipsnames]{{xcolor}}
\usepackage[most]{{tcolorbox}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\usepackage[normalem]{{ulem}}
\usepackage[hidelinks]{{hyperref}}

\renewcommand{{\contentsname}}{{Daftar Isi}}
\newcolumntype{{Z}}[1]{{>{{\hsize=#1\hsize\RaggedRight\arraybackslash}}X}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\setlist[itemize]{{leftmargin=1.4em,itemsep=2pt,topsep=3pt}}
\setlist[enumerate]{{leftmargin=1.6em,itemsep=3pt,topsep=3pt}}
\setkeys{{Gin}}{{width=\linewidth,keepaspectratio}}

\definecolor{{birunota}}{{HTML}}{{1F4E79}}
\definecolor{{hijaunota}}{{HTML}}{{1E6F41}}
\definecolor{{jinggapenting}}{{HTML}}{{8A5A00}}
\definecolor{{merahperingatan}}{{HTML}}{{9C3B0F}}
\definecolor{{abukutipan}}{{HTML}}{{6B6B6B}}
\newtcolorbox{{kotakcatatan}}{{colback=birunota!4,colframe=birunota,boxrule=0.4pt,
  left=6pt,right=6pt,top=5pt,bottom=5pt,sharp corners,before skip=8pt,after skip=8pt}}
\newtcolorbox{{kotaksaran}}{{colback=hijaunota!4,colframe=hijaunota,boxrule=0.4pt,
  left=6pt,right=6pt,top=5pt,bottom=5pt,sharp corners,before skip=8pt,after skip=8pt}}
\newtcolorbox{{kotakpenting}}{{colback=jinggapenting!5,colframe=jinggapenting,boxrule=0.4pt,
  left=6pt,right=6pt,top=5pt,bottom=5pt,sharp corners,before skip=8pt,after skip=8pt}}
\newtcolorbox{{kotakperingatan}}{{colback=merahperingatan!5,colframe=merahperingatan,boxrule=0.4pt,
  left=6pt,right=6pt,top=5pt,bottom=5pt,sharp corners,before skip=8pt,after skip=8pt}}
\newtcolorbox{{kotakawas}}{{colback=merahperingatan!3,colframe=merahperingatan!70,boxrule=0.4pt,
  left=6pt,right=6pt,top=5pt,bottom=5pt,sharp corners,before skip=8pt,after skip=8pt}}
\newtcolorbox{{kotakkutipan}}{{colback=white,colframe=abukutipan,boxrule=0pt,
  leftrule=2pt,left=8pt,right=6pt,top=4pt,bottom=4pt,sharp corners,
  before skip=8pt,after skip=8pt}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\footnotesize {tex_escape_plain(head)}}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}

\hypersetup{{pdftitle={{{tex_escape_plain(opt.title)}}}}}

\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries {tex_escape_plain(opt.title)}}}{BS + BS}
\end{{center}}
\vspace{{2pt}}
\hrule
\vspace{{10pt}}
"""


def tex_escape_plain(t):
    return "".join(SPECIAL.get(ch, ch) for ch in t or "")


# --------------------------------------------------------------------------- #
# Konversi span sebaris
# --------------------------------------------------------------------------- #

def esc(t):
    """Escape teks biasa; superskrip Unicode jadi \\textsuperscript."""
    out = []
    for ch in t:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in SUP:
            out.append(BS + "textsuperscript{" + SUP[ch] + "}")
        else:
            out.append(ch)
    return "".join(out)


def inline(t, base=None):
    """Konversi span sebaris Markdown menjadi LaTeX."""
    slots = []

    def stash(tex):
        slots.append(tex)
        return "@@S%d@@" % (len(slots) - 1)

    def image(alt, src):
        p = (base / src) if base and not re.match(r"^[a-z]+://", src) else None
        if p and p.exists():
            return stash(BS + "includegraphics{" + str(p).replace(BS, "/") + "}")
        return stash(BS + "textit{[gambar: " + esc(alt or src) + "]}")

    def render(x):
        x = x.replace("&mdash;", "\u2014").replace("&ndash;", "\u2013")
        x = x.replace("&nbsp;", "~").replace("&amp;", "&")
        x = x.replace("<sup>", SUPO).replace("</sup>", SUPC)
        x = re.sub(r"</?(?:br|b|i|em|strong|span|div)[^>]*>", "", x)
        # matematika diteruskan apa adanya; koma desimal dikurung agar TeX tidak
        # menyisipkan spasi tanda baca (0,6569 bukan 0, 6569)
        x = re.sub(r"\$\$(.+?)\$\$",
                   lambda m: stash("$" + _decimal(m.group(1)) + "$"), x)
        x = re.sub(r"\$([^$]+)\$",
                   lambda m: stash("$" + _decimal(m.group(1)) + "$"), x)
        x = re.sub(r"`([^`]+)`",
                   lambda m: stash(BS + "texttt{" + esc(m.group(1)) + "}"), x)
        x = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: image(m.group(1), m.group(2)), x)
        x = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: m.group(1), x)
        x = re.sub(r"~~(.+?)~~", lambda m: stash(BS + "sout{" + render(m.group(1)) + "}"), x)
        x = re.sub(r"\*\*\*(.+?)\*\*\*",
                   lambda m: stash(BS + "textbf{" + BS + "textit{" + render(m.group(1)) + "}}"), x)
        x = re.sub(r"\*\*(.+?)\*\*",
                   lambda m: stash(BS + "textbf{" + render(m.group(1)) + "}"), x)
        x = re.sub(r"(?<![\*\w])_([^_]+)_(?![\*\w])",
                   lambda m: stash(BS + "textit{" + render(m.group(1)) + "}"), x)
        x = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)",
                   lambda m: stash(BS + "textit{" + render(m.group(1)) + "}"), x)
        x = esc(x)
        return x.replace(SUPO, BS + "textsuperscript{").replace(SUPC, "}")

    out = render(t)
    while SLOT.search(out):
        out = SLOT.sub(lambda m: slots[int(m.group(1))], out)
    return out


def _decimal(math):
    return re.sub(r"(?<=\d),(?=\d)", "{,}", math)


# --------------------------------------------------------------------------- #
# Tabel
# --------------------------------------------------------------------------- #

def _tokens(cell):
    """Panjang setiap token tak-terpatahkan dalam satu sel tabel."""
    for part in re.split(r"(`[^`]*`)", cell):
        if not part:
            continue
        mono = part.startswith("`") and part.endswith("`") and len(part) > 1
        txt = re.sub(r"[^\w\s()/+-]", "", part)
        for tok in re.split(r"[\s\-/]+", txt):
            if tok:
                yield len(tok) * (1.45 if mono else 1.0)


def table(rows, base=None):
    """Tabel Markdown menjadi xltabular dengan lebar kolom terbobot."""
    head, align, body = rows[0], rows[1], rows[2:]
    cols = [c.strip() for c in head.strip().strip("|").split("|")]
    aligns = [c.strip() for c in align.strip().strip("|").split("|")]
    nc = len(cols)
    aligns += ["---"] * (nc - len(aligns))

    grid = []
    for r in body:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        grid.append((cells + [""] * nc)[:nc])

    size_name = "scriptsize" if nc >= 9 else "footnotesize" if nc >= 7 else "small"

    # kolom numerik memakai lebar tetap; kolom teks dibobot menurut isinya
    flex = [i for i, a in enumerate(aligns) if not a.endswith(":")]
    if not flex:                                   # semua kolom rata kanan
        spec = "r" * nc
    else:
        raw, longest = {}, {}
        for i in flex:
            lens = sorted(len(row[i]) for row in grid) or [len(cols[i])]
            typ = lens[int(0.8 * (len(lens) - 1))]
            cells_i = [cols[i]] + [r[i] for r in grid]
            head_len = len(re.sub(r"[^\w\s()/+-]", "", cols[i]))
            longest[i] = max((t for c in cells_i for t in _tokens(c)), default=1.0)
            raw[i] = max(float(typ), float(head_len), 6.0)

        tot = sum(raw.values()) or 1.0
        w = {i: max(0.40, min(3.00, raw[i] / tot * len(flex))) for i in flex}

        # batas bawah agar token terpanjang muat; sisanya dikompresi proporsional
        per_line = {"small": 96, "footnotesize": 112, "scriptsize": 128}[size_name]
        chars_flex = max(20.0, per_line - 8.0 * (nc - len(flex)))
        floor = {i: min(2.50, longest[i] * 1.28 / chars_flex * len(flex)) for i in flex}
        for _ in range(5):
            w = {i: max(w[i], floor[i]) for i in flex}
            norm = sum(w.values()) or 1.0
            w = {i: v / norm * len(flex) for i, v in w.items()}
        spec = "".join("r" if aligns[i].endswith(":") else "Z{%.3f}" % w[i]
                       for i in range(nc))

    out = ["{" + BS + size_name,
           BS + "begin{xltabular}{" + BS + "linewidth}{@{}" + spec + "@{}}",
           BS + "toprule",
           " & ".join(BS + "textbf{" + inline(c, base) + "}" for c in cols) + " " + BS + BS,
           BS + "midrule" + BS + "endhead"]
    for row in grid:
        out.append(" & ".join(inline(c, base) for c in row) + " " + BS + BS)
    out += [BS + "bottomrule", BS + "end{xltabular}}", ""]
    return out


# --------------------------------------------------------------------------- #
# Blok
# --------------------------------------------------------------------------- #

def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _list_block(lines, i, ordered, base, depth=0):
    """Konversi satu daftar beserta sarangnya; kembalikan (baris LaTeX, indeks)."""
    env = "enumerate" if ordered else "itemize"
    pat = re.compile(r"^\d+[.)]\s+" if ordered else r"^[-*+]\s+")
    other = re.compile(r"^[-*+]\s+" if ordered else r"^\d+[.)]\s+")
    n = len(lines)
    base_ind = _indent(lines[i])
    out = [BS + "begin{" + env + "}"]
    while i < n:
        line = lines[i]
        if not line.strip():
            if i + 1 < n and lines[i + 1].strip() and _indent(lines[i + 1]) >= base_ind \
                    and (pat.match(lines[i + 1].strip()) or other.match(lines[i + 1].strip())):
                i += 1
                continue
            break
        ind = _indent(line)
        if ind < base_ind or not pat.match(line.strip()):
            if ind >= base_ind and other.match(line.strip()) and ind == base_ind:
                break
            if ind < base_ind:
                break
            if not pat.match(line.strip()):
                break
        item = pat.sub("", line.strip())
        i += 1
        # baris lanjutan dan daftar bersarang
        nested = []
        while i < n and lines[i].strip():
            ind2 = _indent(lines[i])
            body = lines[i].strip()
            if ind2 > base_ind and (re.match(r"^[-*+]\s+", body) or re.match(r"^\d+[.)]\s+", body)):
                sub, i = _list_block(lines, i, bool(re.match(r"^\d+[.)]\s+", body)),
                                     base, depth + 1)
                nested += sub
                continue
            if ind2 > base_ind:
                item += " " + body
                i += 1
                continue
            break
        out.append(BS + "item " + inline(item, base))
        out += nested
    out += [BS + "end{" + env + "}", ""]
    return out, i


def convert(md, base=None, first_heading_is_title=True):
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    seen_title = not first_heading_is_title
    while i < n:
        raw = lines[i]
        t = raw.strip()

        if not t:
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", t):
            i += 1
            continue

        if t.startswith("```") or t.startswith("~~~"):
            fence = t[:3]
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith(fence):
                buf.append(lines[i])
                i += 1
            i += 1
            # \end{Verbatim} wajib berdiri sendiri pada barisnya
            out += ["{" + BS + "footnotesize",
                    BS + "begin{Verbatim}[frame=leftline,framesep=6pt,xleftmargin=8pt]"]
            out += buf
            out += [BS + "end{Verbatim}", "}", ""]
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", t)
        if m:
            lvl = len(m.group(1))
            title = inline(m.group(2).rstrip(":"), base)
            if lvl == 1 and not seen_title:
                seen_title = True
                i += 1
                continue
            cmd = {1: "section", 2: "section", 3: "subsection",
                   4: "subsubsection", 5: "paragraph", 6: "paragraph"}[lvl]
            out.append(BS + cmd + "*{" + title + "}")
            out.append(BS + "addcontentsline{toc}{" +
                       (cmd if cmd not in ("paragraph",) else "subsubsection") +
                       "}{" + title + "}")
            i += 1
            continue

        if t.startswith("|"):
            blk = []
            while i < n and lines[i].strip().startswith("|"):
                blk.append(lines[i].strip())
                i += 1
            if len(blk) >= 2 and re.match(r"^\|[\s:|-]+\|?$", blk[1]):
                out += table(blk, base)
            else:
                out += [inline(" ".join(b.strip("|").strip() for b in blk), base), ""]
            continue

        if t.startswith(">"):
            m2 = re.match(r"^>\s*\[!(\w+)\]", t)
            env, label = ALERTS.get((m2.group(1).upper() if m2 else ""), ("kotakkutipan", ""))
            body = []
            while i < n and lines[i].strip().startswith(">"):
                c = re.sub(r"^>\s?", "", lines[i].strip())
                if not re.match(r"^\[!\w+\]\s*$", c.strip()):
                    body.append(c.strip())
                i += 1
            lead = (BS + "textbf{" + label + ".} ") if label else ""
            out += [BS + "begin{" + env + "}",
                    lead + inline(" ".join(x for x in body if x), base),
                    BS + "end{" + env + "}", ""]
            continue

        if re.match(r"^[-*+]\s+", t) or re.match(r"^\d+[.)]\s+", t):
            blk, i = _list_block(lines, i, bool(re.match(r"^\d+[.)]\s+", t)), base)
            out += blk
            continue

        par = []
        while i < n and lines[i].strip() and not BLOCK.match(lines[i].strip()):
            par.append(lines[i])
            i += 1
        out.append(" ".join(inline(p.strip(), base) + ((" " + BS + BS) if p.endswith("  ") else "")
                            for p in par))
        out.append("")
    return out


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="berkas Markdown sumber")
    ap.add_argument("-o", "--outdir", type=Path, help="direktori keluaran (baku: sebelah sumber)")
    ap.add_argument("--title", help="judul dokumen (baku: judul H1 pertama atau nama berkas)")
    ap.add_argument("--running-head", help="teks kepala halaman (baku: judul)")
    ap.add_argument("--paper", default="a4paper", help="a4paper, letterpaper, ... (baku: a4paper)")
    ap.add_argument("--margin", default="2cm", help="margin halaman (baku: 2cm)")
    ap.add_argument("--fontsize", default="10", choices=["10", "11", "12"],
                    help="ukuran fonta dasar (baku: 10)")
    ap.add_argument("--main-font", help="nama fonta utama")
    ap.add_argument("--mono-font", help="nama fonta monospace")
    ap.add_argument("--toc", action="store_true", help="sisipkan daftar isi")
    ap.add_argument("--keep-tex", action="store_true", help="pertahankan berkas .tex")
    ap.add_argument("--tex-only", action="store_true", help="hanya tulis .tex, tanpa kompilasi")
    opt = ap.parse_args(argv)

    if not opt.input.is_file():
        ap.error("berkas tidak ditemukan: %s" % opt.input)

    md = io.open(opt.input, encoding="utf-8").read()
    if not opt.title:
        h1 = re.search(r"^#\s+(.+)$", md, re.M)
        opt.title = re.sub(r"[*`_]", "", h1.group(1)).strip() if h1 else opt.input.stem

    outdir = opt.outdir or opt.input.parent
    outdir.mkdir(parents=True, exist_ok=True)
    tex_path = outdir / (opt.input.stem + ".tex")

    body = convert(md, base=opt.input.parent.resolve())
    toc = (BS + "tableofcontents\n" + BS + "vspace{10pt}\n") if opt.toc else ""
    tex = preamble(opt) + toc + "\n".join(body) + "\n" + BS + "end{document}\n"
    io.open(tex_path, "w", encoding="utf-8", newline="\n").write(tex)
    print("LaTeX:", tex_path)

    if opt.tex_only:
        return 0
    if shutil.which("tectonic") is None:
        print("tectonic tidak ditemukan pada PATH; kompilasi dilewati", file=sys.stderr)
        return 1

    r = subprocess.run(["tectonic", "-X", "compile", str(tex_path), "--outdir", str(outdir)])
    if r.returncode == 0:
        print("PDF  :", outdir / (opt.input.stem + ".pdf"))
        if not opt.keep_tex:
            tex_path.unlink()
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
