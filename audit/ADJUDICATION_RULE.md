# All-wrong event adjudication rule

Written 2026-09-25 after adjudicator B returned and before adjudicator A's labels were read.

Scope: every question on which every model is graded wrong by the corrected grader (harness/grade.py) and that is not
already explained by a documented grading defect (math reference formatting, multi-answer code judged by exact match,
option-dependent free-response GPQA). These are the 11 items in `adjudication_items.json`.

Two adjudicators solve each item independently from the question text alone. They never see the reference answer or
any model output. Each returns an answer, a confidence, and whether the item is well posed (exactly one defensible
answer).

Classification (applied mechanically by `harness/canonical.py`):
- **genuine co-failure**: at least one adjudicator's answer equals the reference answer;
- **ambiguous item**: otherwise, if either adjudicator marks the item not well posed;
- **reference error**: otherwise, when both adjudicators answer the same thing and it differs from the reference.

Any item that fits none of these (the two adjudicators give different well-posed answers, neither equal to the reference) is
reported as "unresolved" and counted as a possible genuine co-failure in the upper bound.
