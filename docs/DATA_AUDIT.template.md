# Data and analysis audit, version 2 (September 2026)

This file records every problem found in the first release of *Combining LLMs Rarely Beats the Single Best Model*
(arXiv 2606.27288, June 2026), what was changed, how each change was verified, and what it did to the results.
Nothing was deleted: first-release inputs and outputs, and every replaced cache entry, are kept. Every number below is
filled in by `scripts/render_docs.py` from the macros the pipeline writes (`paper/numbers.tex`,
`paper/numbers_pillars.tex`); `scripts/check_numbers.py` fails if the paper and the released data disagree.

## Summary

The first release reported a co-failure tail on open-ended mathematics and code (for example {{MathJuneK}} of {{MathJuneN}} MATH-500
questions failed by all 67 models) and read its underpricing by pairwise correlation as the paper's central empirical
result. **That tail was an evaluation artifact, not model co-failure.** Of the {{TotJune}} all-wrong events of the first
release ({{TotJuneQ}} distinct questions; a question in two pools counts once per pool), {{TotGrader}} were grader defects, {{TotRefErr}} had wrong reference answers, {{TotAmbig}} were ambiguous,
{{TotTrunc}} were resolved by a longer token budget,
{{TotCorrupt}} by re-querying a corrupt response, and {{TotGenuine}} were genuine ({{TotGenuineQ}} distinct MMLU-Pro questions, asked for a direct answer; both raters also judged
both of them ambiguous, so this is an upper bound).
After the audit, no MATH-500, MATH-Hard, AIME, GPQA-Diamond or competition-code question defeats every model.

The theory (the 1 − β ceiling, the non-identification of β from ρ, the pool-size underpricing of any common-mode atom) and
the $0 certificate are unaffected. The title claim stands for a different reason: the ceiling is high, the best single
model already sits near it on MATH-500 and MATH-Hard ({{MathSB}} and {{HardSB}}), and where there is headroom (AIME, G = {{AimeG}};
GPQA, G = {{GmcG}}) it is disagreement a router would have to resolve per query; on the 15-model pools where we train routers,
every learned router scores below the single best model. The spurious tail is itself reported as a finding: evaluation errors
produce the kind of common-mode atom the theory describes, which pairwise ρ cannot see ({{ArtRatioTet}}× "underpriced" by a
tetrachoric single-factor model, growing with pool size), and reading the all-wrong questions catches it.

## 1. Grading defects

* **Math extraction.** `\boxed{...}` was matched with a one-level brace pattern, so `\boxed{\frac{3\sqrt{3}}{4}}` fell
  through to the last line of the response.
* **Math equivalence.** Answers were compared without normalization: a correct "5" failed against the reference "x=5",
  "10080" against "10,\!080", "864" against "864 \mbox{ inches}^2", "5.4" against "5.4 \text{ cents}", "\frac{4}{3}" against
  "\frac43", and "1+\sqrt{19}, 1-\sqrt{19}" against "1 \pm \sqrt{19}". A reference no response can match makes every
  model wrong at once.
* **Multiple choice.** "**Answer:** A" and "The final answer is: I" returned no letter.
* **Code with several correct outputs.** Exact-match grading scored valid alternative outputs as wrong. Each of the five code
  problems all models "failed" accepts any valid answer; with a special checker per problem (`harness/code_checkers.py`,
  results in `runs/code_checker_regrade.json`, every changed cell in `runs/rerun_log.csv`), each is solved by at least three
  models.
* **Numbers.** An answer on the line after "####" was missed.
* **Stale grading.** The 15-model pools, the churn pool and the fusion samples had been scored by still earlier extractors
  (a first-letter extractor that could not read MMLU-Pro's letters E–J), although the paper said every output had been
  re-graded.

**Fix.** `harness/grade.py` now uses balanced-brace extraction and the MATH benchmark's reference answer normalization
(Hendrycks et al., 2021), extended for units, thousands separators, ± and unordered answer lists, with a sympy fallback; the
multiple-choice and number extractors are fixed while keeping the first version's precedence. The first-version grader is
kept unchanged in `harness/grade_v1.py`. Every released artifact is re-scored from its raw responses by
`harness/regrade_final.py` (report: `runs/regrade_final_report.json`). Every cell of the market-scale matrices re-grades
identically to its stored first-version grade under `grade_v1.py`, so the change is entirely due to the grader.

## 2. Corrupt responses

{{CorruptCells}} cells had no usage record and empty or cut-off content (for example "Brief reasoning: To find the value of
$f(") and had been graded wrong; {{CorruptLlamaSmall}} of them are Llama-3.2-3B, the rest scattered over four other models
(per-model counts in `runs/canonical.json`, key `corrupt`). Each was matched to the exact cache key of its original request,
then re-graded from a complete cached response ({{CorruptFromCache}}) or re-queried with the identical request
({{CorruptRequeried}}, ${{CorruptUsd}}); {{CorruptFlip}} became correct. A later scan found {{EmptyRequeried}} more calls that
returned zero output tokens without exhausting their budget; they were re-queried the same way (`harness/requery_empty.py`;
{{EmptyFlip}} became correct). Every cell is logged in `runs/rerun_log.csv`.

## 3. Truncation

The first release called its 67-model matrices "truncation-corrected", but the control had only been run on an older
53-model matrix. Every answer that hit its token budget inside an all-wrong question is now re-asked with 32,768 tokens
(`harness/detruncate_v2.py`); this can only lower β. Every answer that hit its budget
inside a question still failed by every model after the audit was re-asked; {{CappedLeft}} remain unchecked
(`capped_not_reasked` in `runs/canonical.json`). A few re-asks failed (six because the model is no longer
served, per `runs/detrunc_v2.log`); every failed re-ask sits in a question that another check resolved, which is why none
is left unchecked.

## 4. Token budgets and pinned inputs

The loaders' token budgets changed during the project, and the same prompt is often cached at more than one budget. The
pillar-era runs (the 15-model pools, churn, fusion, cascade) used GSM8K 1,024, MMLU/ARC 512, MATH-500 2,048 and MMLU-Pro 768
tokens; stored grades agree 94–100% with the responses at those budgets and 82–96% at the later ones, and the cascade's
cheap-model samples exist only at those budgets. The budgets are pinned in `harness/regrade_final.py` and recorded per cell.
Question text had been fetched live at run time; every item used is now pinned in `runs/items_snapshot.json`, byte-identical
to the live source for every window the paper uses.

## 5. Wrong and ambiguous reference answers

Every question still failed by all models after §1–§4 was solved by two AI raters who saw only the question and its options,
never the reference answer or a model output (`audit/adjudication_items.json`). The classification rule
(`audit/ADJUDICATION_RULE.md`) was written before the second rater's labels were read: genuine if either rater's answer equals
the reference; otherwise ambiguous if either marks the question ill-posed; otherwise a reference error if both agree on a
different answer. Result (`audit/adjudication_result.json`): {{TotRefErr}} reference errors (for example the MATH-Hard
arithmetic-series question, whose reference 3 should be −3), {{TotAmbig}} ambiguous, and genuine co-failures on
{{MixK}} of {{MixN}} mix questions, {{ProK}} of {{ProN}} 15-model MMLU-Pro questions and {{MmluK}} of {{MmluN}} market MMLU-Pro
questions.

## 5b. Questions some model did not answer, and prompts

β is measured on the questions every model answered. Across all benchmarks one question outside that set was failed by
every model that answered it: AIME 2025 question 27 (reference 248, correct), answered by 51 of 52 models, 30 of them at their
token budget. Put through the same truncation control (`harness/detruncate_v2.py --include-partial`;
`runs/detrunc_aime_partial.log`), it is solved by six models at 32,768 tokens, so no question outside the analysed set is failed
by every model that answered it ({{TotPartK}} such questions). Twelve AIME-2024 questions were never run.

MMLU, ARC and MMLU-Pro prompts ask for the answer letter only (a direct-answer protocol); mathematics, AIME, GSM8K, code and
GPQA prompts let the model reason first. The only genuine co-failures are on MMLU-Pro under the direct-answer protocol, which
may itself act as a common mode. Every prompt is in `runs/items_snapshot.json`.

Every first-release all-wrong event that the corrected grader resolved was resolved by answers equal to the reference; the
answers it now accepts are listed per question in `audit/grader_resolved.json`.

## 6. Free-response GPQA

The first release's free-response GPQA tail (β = {{OpenOneBeta}}, {{OpenOneK}}/{{OpenOneN}}) came from questions cut at their
first option: {{OpenOneOpt}} of its all-wrong questions depend on the stripped options and {{OpenOneDoubt}} have doubtful
references; its judges also saw only the first 2,000 characters of each answer. A protocol written before the new data were collected
(`gpqa_open_v2/PROTOCOL.md`) screened all 130 questions with blind raters and re-ran the test with full-text judging:
β = {{OpenBeta}} ({{OpenK}}/{{OpenN}}), as in multiple choice on the same questions.

## 7. Estimator noise

Tetrachoric correlations had been averaged over random subsets of 200–400 model pairs, solved to a tolerance of 1e-3, with
unsolvable pairs clamped; two scripts gave different ratios from the same data. `harness/exact_copula.py` uses all pairs
solved to machine precision (bivariate normal CDF within 1e-14 of scipy), adaptive quadrature, 4-million-draw Monte Carlo
with standard errors, a 2,000-replicate bootstrap with a fixed seed and exact sub-matrix means for the pool-size curve.

## 8. Figures and comparisons

* The equal-quality figure was drawn from the `hardBv2` run (ρ 0.70 / 0.45) while its caption described `eqq2`; it is redrawn.
* The regime and format figures had values typed into the plotting code; they now read `runs/canonical.json`.
* The matched-quality text compared MMLU-Pro at k = 3 with MATH-500 at k = 6; both are now compared at k = 3.

## 9. Reproduction check of the pillar analyses

Before any corrected input was used, all pillar outputs (routers, Pillar B, matched-quality fusion, k*, cascade, held-out
cascade, churn, copula dichotomy, fusion aggregation) were regenerated from the first-version inputs with the pinned loader
and compared field by field with the published files; every field matched.

## 10. Mathematical corrections (first audit, 2026-09-24)

The certificate's direction of the single-best bound (and the same error in `beta_certificate.py`), the cascade integral
identity, the pool-bias monotonicity proof, the optimal ensemble size, notation clashes, and several numeric transcriptions.

## Headline claims, before and after

| | version 1 (June 2026) | version 2 |
|---|---|---|
| MATH-500 all-models-wrong β | {{MathJuneBeta}} ({{MathJuneK}}/{{MathJuneN}}) | {{MathBeta}} ({{MathK}}/{{MathN}}; 95% upper {{MathBetaHi}}) |
| MATH-Hard β | {{HardJuneBeta}} ({{HardJuneK}}/{{HardJuneN}}) | {{HardBeta}} ({{HardK}}/{{HardN}}) |
| code β | {{CodeJuneBeta}} ({{CodeJuneK}}/{{CodeJuneN}}) | {{CodeBeta}} ({{CodeK}}/{{CodeN}}; special checkers) |
| free-response GPQA β | {{OpenOneBeta}} ({{OpenOneK}}/{{OpenOneN}}) | {{OpenBeta}} ({{OpenK}}/{{OpenN}}) |
| MATH-500 best single model | {{MathJuneSB}} | {{MathSB}} |
| ρ underprices frontier co-failure | 2.5× on MATH-500 | not supported: no genuine tail |

## Spend

All re-runs are logged per cell in `runs/rerun_log.csv`. Every API call of the project that returned a response is
listed in `runs/spend_ledger.csv` (request hash, month, logged cost): ${{SpendJune}} in June 2026 and ${{SpendAudit}} for this audit.
The first release quoted an account-meter reading of about $560, reconstructed after concurrent runs overwrote the meter; it
cannot be reconciled with the ledger (calls that failed without a response are not in it), which we report instead.

## Where the old versions are

* First-release matrices and outputs: in the Hugging Face dataset (`runs/matrix_*.json` without a suffix; tag `v1-2026-06`).
* Replaced cache entries: `harness/cache_quarantine/` in the maintainers' working copy (not deleted).
