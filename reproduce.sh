#!/usr/bin/env bash
# Reproduce every number and figure in the paper from the released data, offline (no API key needed).
#   1. download the data at the pinned Hugging Face revision        (scripts/fetch_data.py)
#   2. recompute all numbers and compare with the paper's macros     (scripts/check_numbers.py; exits 1 on mismatch)
#   3. regenerate the figures into paper/figures/
#   4. optionally rebuild the PDF (needs a LaTeX install)
# Expect about 15 minutes on a recent laptop (the ratio bootstrap dominates). Set CANON_BOOT=200 for a quick pass; the
# check will then report bootstrap-interval mismatches, which is expected.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 1/4 fetch data (pinned revision)"
$PY scripts/fetch_data.py

echo "== 2/4 recompute every reported number and compare with paper/numbers*.tex"
$PY scripts/check_numbers.py

echo "== 3/4 figures"
( cd harness && $PY make_fig_realizability.py && $PY paper_figs_v2.py && $PY figures.py )

echo "== 4/4 PDF (optional)"
if command -v latexmk >/dev/null; then (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null && echo "built paper/main.pdf"); else echo "latexmk not found; skipped"; fi
echo "DONE: every number in the paper was recomputed from the released data and matched."
