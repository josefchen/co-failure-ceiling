# co-failure-ceiling

Code, data pointers and a one-command reproduction for

> Josef Chen. *Combining LLMs Rarely Beats the Single Best Model: A Provable Co-Failure Ceiling Across 67 Frontier Models.*
> arXiv:2606.27288 (arXiv listing title: *When Does Combining Language Models Help? A Co-Failure Ceiling on Routing,
> Voting, and Mixture-of-Agents Across 67 Frontier Models*).

- Paper: https://arxiv.org/abs/2606.27288
- Data (outcome matrices, raw responses, audit files): https://huggingface.co/datasets/josefchen/co-failure-67-models
- Interactive companion: https://huggingface.co/spaces/josefchen/orchestration-is-allocation

## What the paper shows

**Combining LLMs has a hard ceiling, and the frontier is already at it.** For any policy that returns one of its member models'
answers (a router, a vote, a cascade), accuracy cannot exceed 1 − β, where β is the rate at which **every** model is wrong on
the same question. Pairwise error correlation ρ, the number practitioners look at, provably cannot identify β, and a model
calibrated to ρ underprices any common-mode failure by a factor that grows with the pool. A Clopper–Pearson bound on β gives
a $0 certificate on the largest gain any such policy can deliver, before a router is built.

Measured on 67 models from 21 providers and audited question by question:

| benchmark | single best | oracle | all-wrong (graded → genuine) | certified max gain |
|---|---|---|---|---|
| MATH-500 | 0.988 | 1.000 | 0 → 0 | 0.048 |
| MATH-Hard | 0.993 | 0.997 | 1 → 0 | 0.041 |
| AIME 2024+25 | 0.881 | 1.000 | 0 → 0 | 0.358 |
| GPQA-Diamond (MC) | 0.846 | 1.000 | 0 → 0 | 0.281 |
| MMLU-Pro | 0.960 | 0.992 | 1 → 1 | 0.136 |

Not one mathematics, science or competition-code question defeats every model. The best single model already scores 0.988 on MATH-500
and 0.993 on MATH-Hard. The headroom that remains is out of reach: on the 15-model pools every learned router scores
below the single best model on held-out queries, an LLM router just picks that model, and majority voting loses to its own
best member in 75–93% of three-model ensembles.

**Correction notice.** The first release (June 2026) reported a co-failure tail on mathematics and code that pairwise ρ
underpriced about 2.5×. That tail was manufactured by our own evaluation: of 58 all-wrong events (52
distinct questions), 44 were grader defects (reference answers such as "x=5" or "864 \mbox{ inches}^2" that no
response could match), and the rest were wrong or ambiguous references, code outputs exact-match grading rejected, and truncation; the
3 that survive sit on 2 ambiguous MMLU-Pro questions asked for a direct answer. The spurious tail
carried the underpricing signature the theory predicts for a common-mode atom, and is now reported as a finding. Full
details: [DATA_AUDIT.md](DATA_AUDIT.md).

## Certify your own model pool in one line

Grade your models on a held-out set, save a CSV with one row per question and one column per model (1 = correct, 0 = wrong),
and run:

```bash
python3 harness/beta_certificate.py --csv my_grades.csv --overhead 0.02
```

It prints the ceiling 1 − β, the single-best accuracy and the certified upper bound on what any router, vote or cascade over
that pool can gain (95% confidence, Bonferroni-corrected for picking the best model in-sample). No API calls, no training.

## Benchmark errata

[`audit/benchmark_errata.md`](audit/benchmark_errata.md) lists the MATH-500, MATH-Hard, MMLU-Pro, MMLU, GSM8K and ARC questions
our audit found with wrong or ambiguous reference answers, and the MATH-500 / MATH-Hard questions a naive grader marks
all-wrong: references written like "x=5", "10,\!080" or "864 \mbox{ inches}^2", or answers with nested braces. If you evaluate on
these benchmarks, check your grader against that list.

## Reproduce every number

```bash
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` downloads the data at a pinned Hugging Face revision, recomputes every number in the paper from it
(`harness/canonical.py`, `harness/pillars.py`), and **fails if any number differs** from the macros the paper is typeset
from (`paper/numbers.tex`, `paper/numbers_pillars.tex`). It then regenerates the figures and, if LaTeX is installed, the PDF.
No API key is needed. It takes about 15 minutes on a recent laptop, dominated by the bootstrap.

To re-check the grades themselves from the models' own text: `python3 scripts/fetch_data.py --responses`, then
`python3 scripts/verify_grades.py` re-grades every released response with `harness/grade.py` and fails if any grade differs
from the released matrices (`--with-code` also re-executes the code submissions).

## Layout

| path | contents |
|---|---|
| `harness/` | data collection (needs `OPENROUTER_API_KEY`), grading (`grade.py`; `grade_v1.py` is the first-release grader, kept for audit), re-grading (`regrade_final.py`), analysis (`canonical.py`, `pillars.py`, `exact_copula.py`), figures |
| `gpqa_open_v2/` | the free-response GPQA protocol (written before its data were collected), the rater labels and the rewritten questions |
| `audit/` | the all-wrong adjudication: items, the rule (written before the second rater's labels were read), both raters, the result |
| `paper/` | LaTeX source; every number is a generated macro |
| `scripts/` | data fetch, the number check, response export, document rendering, release staging |

## License

Code: MIT. Data and paper text: CC BY 4.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{chen2026cofailure,
  title   = {Combining {LLMs} Rarely Beats the Single Best Model: A Provable Co-Failure Ceiling Across 67 Frontier Models},
  author  = {Chen, Josef},
  journal = {arXiv preprint arXiv:2606.27288},
  year    = {2026}
}
```

See also [CITATION.cff](CITATION.cff).
