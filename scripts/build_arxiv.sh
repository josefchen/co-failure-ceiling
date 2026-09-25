#!/usr/bin/env bash
# Build the arXiv source bundle from paper/ (maintainers). Produces arxiv_payload/arxiv/ and arxiv_payload/arxiv-source.tar.gz,
# checks that the bundle compiles standalone with pdflatex (the .bbl is included, so arXiv needs no bibtex run), and copies
# the resulting PDF to arxiv_payload/main.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
( cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null )
rm -rf arxiv_payload/arxiv && mkdir -p arxiv_payload/arxiv/figures
cp paper/main.tex paper/main.bbl paper/numbers.tex paper/numbers_pillars.tex paper/references.bib paper/roster_rows.tex arxiv_payload/arxiv/
cp paper/figures/*.pdf arxiv_payload/arxiv/figures/
T=$(mktemp -d); cp -r arxiv_payload/arxiv/. "$T"/
( cd "$T" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null )
if grep -E "undefined|^!" "$T/main.log" | grep -v "Font shape" >/dev/null; then echo "standalone build has warnings/errors:"; grep -E "undefined|^!" "$T/main.log"; exit 1; fi
cp "$T/main.pdf" arxiv_payload/main.pdf
tar czf arxiv_payload/arxiv-source.tar.gz -C arxiv_payload/arxiv .
echo "[arxiv] bundle ok: $(tar tzf arxiv_payload/arxiv-source.tar.gz | wc -l | tr -d ' ') files, $(du -h arxiv_payload/arxiv-source.tar.gz | cut -f1); PDF $(grep -o 'Output written on main.pdf ([0-9]* pages' "$T/main.log" | grep -o '[0-9]* pages')"
