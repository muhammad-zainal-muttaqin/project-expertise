"""Render laporan Markdown menjadi PDF melalui skill markdown-ke-pdf.

md2pdf.py dijalankan sampai tahap LaTeX, lalu berkas .tex disesuaikan sebelum
dikompilasi tectonic: gambar tidak diberi indentasi paragraf dan diberi jarak
atas, serta tanda kutip lurus diganti kutip pembuka dan penutup.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


MD2PDF = Path.home() / ".claude" / "skills" / "markdown-ke-pdf" / "scripts" / "md2pdf.py"


def sesuaikan(tex):
    kepala, pemisah, isi = tex.partition("\\begin{document}")
    isi = isi.replace("\\includegraphics{", "\\par\\medskip\\noindent\\includegraphics{")
    isi = re.sub(r'"([^"\n]+)"', lambda m: "“" + m.group(1) + "”", isi)
    return kepala + pemisah + isi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sumber", type=Path)
    parser.add_argument("--md2pdf", type=Path, default=MD2PDF)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as sementara:
        subprocess.run([sys.executable, str(args.md2pdf), str(args.sumber), "--toc", "--tex-only",
                        "-o", sementara], check=True, stdout=subprocess.DEVNULL)
        tex = Path(sementara) / args.sumber.with_suffix(".tex").name
        tex.write_text(sesuaikan(tex.read_text(encoding="utf-8")), encoding="utf-8")
        subprocess.run(["tectonic", "-o", str(args.sumber.parent), str(tex)], check=True)
    print(args.sumber.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
