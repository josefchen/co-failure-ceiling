# Free-response GPQA, v2: item screening and rewrite protocol

Written 2026-09-24, before any v2 model answers or judgements were collected. Changes after this date are
appended under "Amendments" with a date and reason; nothing above that line is edited.

## Why v2 exists
The v1 conversion (`data.py:gpqa_open`) took each GPQA-Diamond item and cut the question at "(A)", dropping the
answer options. Some stems refer to those options ("which of the following...", "among the given options...",
"all the following statements are correct except"). Without the options such an item is either unanswerable or
has no unique answer, so every model is graded wrong and co-failure is manufactured by the conversion rather than
measured. An audit on 2026-09-24 found option-referencing phrasing in 7 of the 10 v1 all-wrong items, against
17 of all 79 complete-coverage items. The v1 judges were also shown only the first 2,000 characters of each
answer (`answer[:2000]`), which can hide a final answer; 118 of 1,422 answers were longer than that.

## Unit
All 130 GPQA-Diamond items used in v1 (`items_original.json`, from `hendrydong/gpqa_diamond_mc`, test split).

## Classification (one label per item)
Classifiers see the stem, the four options, and the gold option. They do not see any model answer, judge vote,
or v1 result.

- **KEEP**: the stem as written is a self-contained question whose unique correct answer is the gold option's
  content.
- **REWRITE**: the stem refers to options, statements, or choices that are not shown, but a self-contained
  question with a unique correct answer equal to the gold exists after editing only the sentence(s) that refer to
  the options. Rules:
  1. Add no information taken from the gold option or from any distractor.
  2. Keep every other part of the stem verbatim.
  3. Ask for the same quantity, entity, or combination the gold names. If the gold has several parts
     (e.g. "A = ..., B = ..."), ask for every part.
- **EXCLUDE**: no such rewrite exists without adding option content. Typical cases: "which statement is
  correct / incorrect / the exception" (the answer is one of several unseen statements); "which step is MOST
  crucial" and other judgements with no unique free-response answer; reagent or route selection where several
  choices are chemically valid and the gold is one of them; and any item whose gold is not the unique answer to a
  self-contained question.

Each decision carries a one- or two-sentence rationale. A first rater (an AI model given only this protocol and
the items) labels every item; a second rater (the auditing author's AI assistant) reviews every label, and any
override is recorded with its reason in `items_v2.json`.

## Analysis
- EXCLUDE items are removed from both the multiple-choice and the free-response analyses, so the two formats are
  compared on the same questions.
- REWRITE items are re-asked to all 18 models with the v1 prompt template and token ladder
  (2,048, then 8,192, then 16,384 tokens if the answer is empty), then judged by the v1 five-judge panel.
- KEEP items reuse their v1 answers (unchanged prompts, served from the response cache).
- Every answer, old or new, is judged with its full text. This replaces the v1 2,000-character window.
- Aggregation is unchanged: majority vote of the available judges, ties count as incorrect, no judge grades its
  own model. The unanimous and lenient rules are reported as robustness checks.
- Primary estimate: beta = share of complete-coverage v2 items that every model gets wrong, with a two-sided 95%
  Clopper-Pearson interval, next to the multiple-choice beta on the same items.
- v1 numbers (beta = 0.127, 10/79) are reported alongside and marked retired, with the reason.

## Amendments

### Amendment 1 (2026-09-25, recorded after rating and before any v2 answer or judgement was collected)
- First-rater labels (four blind rater instances, one per shard of the 130 items): 80 KEEP, 5 REWRITE, 45 EXCLUDE.
  The second rater agreed with all 130 labels (no overrides).
- The second rater tagged each EXCLUDE with its reason: **O**, option-dependent (the stem refers to unseen options
  or statements, or lacks content that existed only in the options; 22 items), or **U**, the stem is
  self-contained but the gold is not a unique or correct free-response answer (23 items). U exclusions go beyond
  option-dependence and rest on the rater's scientific judgement, so we add a pre-specified secondary analysis.
- **Primary set:** KEEP + REWRITE (85 items). **Secondary set:** everything except the O items (108 items; the
  23 U items keep their v1 stems and v1 answers). Both are reported, with the same multiple-choice comparison
  on each set.
