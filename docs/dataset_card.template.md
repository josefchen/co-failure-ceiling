---
language:
- en
license: cc-by-4.0
pretty_name: Co-Failure Matrices — 67 LLMs
tags:
- llm
- llm-ensemble
- model-routing
- mixture-of-agents
- co-failure
- evaluation
- reproducibility
viewer: false
task_categories:
- other
---

# Co-Failure Matrices: 67 LLMs

Per-model, per-question outcome data, raw model responses and audit files behind **Combining LLMs Rarely Beats the Single
Best Model: A Provable Co-Failure Ceiling Across 67 Frontier Models** (Josef Chen, KAIKAKU; arXiv:2606.27288).

- Paper: https://arxiv.org/abs/2606.27288
- Code and one-command reproduction: https://github.com/josefchen/co-failure-ceiling
- Interactive companion: https://huggingface.co/spaces/josefchen/orchestration-is-allocation

## Version 2 (September 2026): correction notice

The first release (June 2026) reported that all 67 models fail {{MathJuneK}} of {{MathJuneN}} MATH-500 questions together, a co-failure tail
that pairwise correlation underpriced. An audit of every all-wrong question found an evaluation error behind each mathematics one (and a problem exact-match grading
cannot verify behind each code one): of {{TotJune}}
all-wrong events ({{TotJuneQ}} distinct questions), {{TotGrader}} were grader defects (reference answers such as "x=5" or
"864 \mbox{ inches}^2" that no response could match), {{TotRefErr}} had wrong reference answers, {{TotAmbig}} were
ambiguous, {{TotMulti}} were code problems with several correct outputs, {{TotTrunc}} were answered with a longer token
budget, {{TotCorrupt}} after re-querying a corrupt response, and {{TotGenuine}} were genuine, on {{TotGenuineQ}} MMLU-Pro questions asked for a direct answer (both also judged ambiguous by
the raters). After the audit, no MATH-500, MATH-Hard, AIME or GPQA-Diamond question defeats every model. Full account:
[DATA_AUDIT.md](DATA_AUDIT.md). The first-release files are kept under their original names and in the `v1-2026-06` tag.

## What's here

`runs/` — outcome matrices; each query id maps to per-model `{correct, max_tokens, tok_in, tok_out, cost, ...}`.

| file | contents |
|---|---|
| `matrix_marketE3_final.json` | MATH-500 and AIME 2024/25, 67 models |
| `matrix_marketMH_final.json` | MATH-Hard (Level-5), 67 models |
| `matrix_marketE2_final.json` | GPQA-Diamond multiple choice (52-model complete-coverage subset used), MMLU-Pro, a MATH-500 slice |
| `matrix_marketCG_final.json` | code_contests, 18 models, execution-graded; `multi_answer` marks problems with several correct outputs |
| `matrix_marketGPQAOPENv2.json` | free-response GPQA, v2 question set (protocol written before the data), five-judge panel verdicts (`judge_open_v2_votes.json`) |
| `matrix_stageA2v3_final.json`, `matrix_hardAv3_final.json` | the 15-model pools (mix; MMLU-Pro) |
| `matrix_churnD_final.json`, `fusion_*_final.json`, `cascade_*_final.json` | churn pool, matched-quality fusion samples, cascades (Opus 4.8 and Mistral-Large as the strong model) |
| `router_llm_stageA2v3_final.json` | the LLM router's logged pick for every query |
| `*_rr.json`, `*_dt.json` | intermediate correction stages (corrupt responses re-queried; truncation control) |
| `matrix_*.json`, `fusion_*.json`, `cascade_*.json` without `_final` | the first release, unchanged |
| `items_snapshot.json` | every question used: prompt, reference answer, token budget |
| `codegen_problems.json`, `codegen_problems_raw.json` | the 63 code problems used, and the 140 fetched (with reference solutions) |
| `canonical.json` and the other analysis outputs | every reported number, as computed by `harness/canonical.py` and `harness/pillars.py` |
| `rerun_log.csv`, `regrade_final_report.json`, `detrunc_*.log` | per-cell log of every re-run and re-grade |
| `spend_ledger.csv`, `cost_registry.csv` | every API call that returned a response (request hash, month, cost), and the itemized run costs |

`responses/responses.jsonl.gz` — the raw model response behind every graded cell, to re-check any grade.
`gpqa_open_v2/` — the free-response protocol (written before its data were collected), rater labels and rewritten questions.
`audit/` — the all-wrong adjudication (items, rule, two blind raters, result).

## Headline numbers (version 2)

| benchmark | models | questions | single best | all-wrong (graded → genuine) |
|---|---|---|---|---|
| MATH-500 | {{MathM}} | {{MathN}} | {{MathSB}} | {{MathAwGraded}} → {{MathK}} |
| MATH-Hard | {{HardM}} | {{HardN}} | {{HardSB}} | {{HardAwGraded}} → {{HardK}} |
| AIME 2024+25 | {{AimeM}} | {{AimeN}} | {{AimeSB}} | {{AimeAwGraded}} → {{AimeK}} |
| GPQA-Diamond (MC) | {{GmcM}} | {{GmcN}} | {{GmcSB}} | {{GmcAwGraded}} → {{GmcK}} |
| MMLU-Pro | {{MmluM}} | {{MmluN}} | {{MmluSB}} | {{MmluAwGraded}} → {{MmluK}} |

## Scope

Free-response GPQA is graded by a five-judge LLM panel (κ {{KappaLo}}–{{KappaHi}}), not by humans. Code is graded against
private and generated tests by exact output match, which cannot verify multi-answer problems. The all-wrong adjudication used
two AI raters under a rule written before the second rater's labels were read. All generations at temperature 0 unless noted (fusion and cascade samples:
0.7).

## Citation

Chen, J. (2026). *Combining LLMs Rarely Beats the Single Best Model: A Provable Co-Failure Ceiling Across 67 Frontier
Models.* arXiv:2606.27288.
