# First-release analysis scripts (superseded)

These are the scripts that computed the co-failure ratios of the first release (arXiv 2606.27288v1, June 2026), kept
unchanged apart from their opening comments, so that anyone can see exactly how those numbers were produced. They are
**superseded** and not maintained:

- `recompute_tetrachoric.py`, `bootstrap_ratio.py`, `clayton_real.py`, `residual_decomp.py`, `codegen_analyze.py`,
  `analyze_stats.py`: pair-subsampled tetrachoric estimates, bootstrap ratios, the Clayton control and the residual
  decomposition. Replaced by `harness/exact_copula.py` and `harness/canonical.py` (all pairs, machine precision, fixed
  seeds). The first-release ratios they produced measured an evaluation artifact (see `DATA_AUDIT.md`).
- `detruncate.py`: the first truncation control, run only on an older 53-model matrix. Replaced by
  `harness/detruncate_v2.py`.

They import modules from `harness/` and read `runs/`; to run one, copy it into `harness/`.
