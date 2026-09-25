"""Dataset loaders via the token-free HF datasets-server REST API.
Returns normalized items: {qid, dataset, prompt, gold, kind}. Deterministic order.
"""
import urllib.request, urllib.parse, json, time, os

BASE = "https://datasets-server.huggingface.co/rows"

def _rows(dataset, config, split, offset, length):
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                 "offset": offset, "length": length})
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(f"{BASE}?{qs}", timeout=60))["rows"]
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {dataset}/{config}/{split}")

def _fetch(dataset, config, split, n, offset=0):
    out, got, step = [], 0, 100
    while got < n:
        batch = _rows(dataset, config, split, offset + got, min(step, n - got))
        if not batch:
            break
        out.extend(batch); got += len(batch)
    return out[:n]

def gsm8k(n, offset=0):
    items = []
    for r in _fetch("openai/gsm8k", "main", "test", n, offset):
        row = r["row"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"qid": f"gsm8k:{r['row_idx']}", "dataset": "gsm8k", "kind": "number",
                      "prompt": f"Solve the problem. Show brief reasoning, then write the final answer after '####'.\n\n{row['question']}",
                      "gold": gold, "max_tokens": 1536})  # room for reasoning models to finish then answer
    return items

def mmlu(n, offset=0):
    items = []
    for r in _fetch("cais/mmlu", "all", "test", n, offset):
        row = r["row"]
        ch = row["choices"]; letters = ["A", "B", "C", "D"]
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ch))
        items.append({"qid": f"mmlu:{r['row_idx']}", "dataset": "mmlu", "kind": "mc",
                      "prompt": f"Question: {row['question']}\n{body}\nAnswer with the single letter (A, B, C, or D) only.",
                      "gold": letters[row["answer"]], "max_tokens": 1024})  # room for reasoning models to think then answer
    return items

def math500(n, offset=0):
    items = []
    for r in _fetch("HuggingFaceH4/MATH-500", "default", "test", n, offset):
        row = r["row"]
        items.append({"qid": f"math500:{r['row_idx']}", "dataset": "math500", "kind": "math",
                      "prompt": f"Solve the problem. Show brief reasoning, then give the final answer in \\boxed{{}}.\n\n{row['problem']}",
                      "gold": str(row["answer"]).strip(), "max_tokens": 4096})
    return items

def mbpp(n, offset=0):
    items = []
    for r in _fetch("google-research-datasets/mbpp", "full", "test", n, offset):
        row = r["row"]
        tests = "\n".join(row["test_list"])
        items.append({"qid": f"mbpp:{r['row_idx']}", "dataset": "mbpp", "kind": "code",
                      "prompt": f"Write a Python function. Return ONLY the code in a ```python block.\n\n{row['text']}\n\nIt must pass:\n{tests}",
                      "gold": json.dumps(row["test_list"]), "max_tokens": 512})
    return items

def mmlu_pro(n, offset=0):
    items, letters = [], [chr(65 + i) for i in range(10)]  # A..J
    for r in _fetch("TIGER-Lab/MMLU-Pro", "default", "test", n, offset):
        row = r["row"]; opts = row["options"]
        body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        items.append({"qid": f"mmlupro:{r['row_idx']}", "dataset": "mmlu_pro", "kind": "mc",
                      "prompt": f"Question: {row['question']}\n{body}\nAnswer with the single letter only.",
                      "gold": str(row["answer"]).strip().upper(), "max_tokens": 2048})  # 10-way MC; reasoning room
    return items

def arc(n, offset=0):
    items, letters = [], ["A", "B", "C", "D"]
    # fetch extra to allow filtering to clean 4-option questions
    raw = _fetch("allenai/ai2_arc", "ARC-Challenge", "test", n * 2, offset)
    for r in raw:
        if len(items) >= n:
            break
        row = r["row"]; texts = row["choices"]["text"]; labs = row["choices"]["label"]
        if len(texts) != 4:
            continue
        try:
            idx = labs.index(row["answerKey"])
        except ValueError:
            continue
        body = "\n".join(f"{letters[i]}. {t}" for i, t in enumerate(texts))
        items.append({"qid": f"arc:{r['row_idx']}", "dataset": "arc", "kind": "mc",
                      "prompt": f"Question: {row['question']}\n{body}\nAnswer with the single letter (A, B, C, or D) only.",
                      "gold": letters[idx], "max_tokens": 1024})
    return items

def _aime_item(qid, problem, answer, ds):
    # AIME answers are integers 0-999. Normalize (strip leading zeros) for exact-match grading.
    try:
        gold = str(int(str(answer).strip()))
    except ValueError:
        gold = str(answer).strip()
    return {"qid": qid, "dataset": ds, "kind": "number",
            "prompt": f"Solve the problem. Show your reasoning, then write the final integer answer after '####'.\n\n{problem}",
            "gold": gold, "max_tokens": 4096}  # competition math: needs room to reason

def math_hard(n, offset=0):
    # MATH Level-5 (lighteval/MATH-Hard): a harder, larger open-ended math distribution than MATH-500;
    # boxed answers, graded by the same sympy-equivalence grader. A second co-failure benchmark.
    import re
    items = []
    for r in _fetch("lighteval/MATH-Hard", "default", "test", n, offset):
        row = r["row"]
        bs = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", row["solution"])
        if not bs:
            continue
        items.append({"qid": f"mathhard:{r['row_idx']}", "dataset": "math_hard", "kind": "math",
                      "prompt": f"Solve the problem. Show brief reasoning, then give the final answer in \\boxed{{}}.\n\n{row['problem']}",
                      "gold": bs[-1].strip(), "max_tokens": 4096})
    return items

def gpqa(n, offset=0):
    # GPQA-Diamond (graduate Physics/Chem/Bio), MC; options are inline in the problem and the
    # gold answer is a \boxed{letter}. A genuinely different (science) domain from MATH-500.
    import re
    items = []
    for r in _fetch("hendrydong/gpqa_diamond_mc", "default", "test", n, offset):
        row = r["row"]
        m = re.search(r"\\boxed\{\s*([A-D])\s*\}", row["solution"])
        if not m:
            continue
        items.append({"qid": f"gpqa:{r['row_idx']}", "dataset": "gpqa", "kind": "mc",
                      "prompt": row["problem"],  # already includes options + boxed-answer instruction
                      "gold": m.group(1), "max_tokens": 1536})  # MC boxed-letter; truncated reasoners de-truncated later
    return items

def gpqa_open(n, offset=0):
    # the SAME GPQA-Diamond questions as gpqa(), but OPEN-ENDED (options stripped) so the
    # answer is free text graded by an LLM-judge panel (judge_open.py), NOT multiple choice. Content is held
    # constant vs the MC version (which shows beta~0), so any co-failure here isolates OPEN-ENDEDNESS as the
    # driver -- the cleanest test of "co-failure tracks open-ended generation, not verifiability". kind="open"
    # => no programmatic grade (grade() returns 0 stub); the real grading is the judge panel over cached answers.
    import re
    items = []
    for r in _fetch("hendrydong/gpqa_diamond_mc", "default", "test", n, offset):
        row = r["row"]; prob = row["problem"]
        m = re.search(r"\\boxed\{\s*([A-D])\s*\}", row["solution"])
        if not m:
            continue
        gold_letter = m.group(1)
        opts = dict(re.findall(r"\(([A-D])\)\s*(.+)", prob))  # letter -> option text
        gold_text = (opts.get(gold_letter) or "").strip()
        if not gold_text:
            continue
        stem = prob.split("(A)")[0].strip()  # question without the options/boxed instruction
        items.append({"qid": f"gpqaopen:{r['row_idx']}", "dataset": "gpqa_open", "kind": "open",
                      "prompt": (f"Answer this graduate-level science question. Reason briefly, then give your "
                                 f"final answer concisely on the last line as 'Answer: <answer>'.\n\n{stem}"),
                      "gold": gold_text, "max_tokens": 2048})
    return items

def codegen(n, offset=0):
    # Third independent open-ended domain: competitive programming (deepmind/code_contests, hard CF band),
    # pure stdin->stdout, execution-graded. Reads the locally-trimmed problem set built by codegen_build.py.
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "codegen_problems.json")
    probs = json.load(open(path))[offset:offset + n]
    items = []
    for i, p in enumerate(probs):
        prompt = ("Solve this competitive-programming problem. Write a COMPLETE Python 3 program that reads "
                  "from standard input and writes the answer to standard output, matching the I/O format "
                  "exactly. Put ONLY the final program in a single ```python code block.\n\n" + p["description"])
        items.append({"qid": f"codegen:{i}", "dataset": "codegen", "kind": "codegen",
                      "prompt": prompt,
                      "gold": json.dumps({"tests": p["tests"], "time_limit": p.get("time_limit", 4)}),
                      "max_tokens": 24000})  # hard problems: reasoning models burn >12k thinking before emitting code;
                                             # 24k avoids truncating the program to empty (a false co-failure). Residual
                                             # truncations in all-wrong columns are de-truncated post-hoc.
    return items

def aime24(n, offset=0):
    items = []
    for r in _fetch("Maxwell-Jia/AIME_2024", "default", "train", n, offset):
        row = r["row"]
        items.append(_aime_item(f"aime24:{r['row_idx']}", row["Problem"], row["Answer"], "aime24"))
    return items

def aime25(n, offset=0):
    items = []
    for r in _fetch("math-ai/aime25", "default", "test", n, offset):
        row = r["row"]
        items.append(_aime_item(f"aime25:{r['row_idx']}", row["problem"], row["answer"], "aime25"))
    return items

LOADERS = {"gsm8k": gsm8k, "mmlu": mmlu, "math500": math500, "mbpp": mbpp, "arc": arc,
           "mmlu_pro": mmlu_pro, "aime24": aime24, "aime25": aime25, "gpqa": gpqa, "math_hard": math_hard,
           "codegen": codegen, "gpqa_open": gpqa_open}

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "items_snapshot.json")
_SNAP = None


def _snapshot():
    global _SNAP
    if _SNAP is None:
        _SNAP = json.load(open(SNAPSHOT)) if os.path.exists(SNAPSHOT) else {}
    return _SNAP


def load(dataset, n, offset=0):
    """Pinned loader (2026-09-25): if runs/items_snapshot.json covers the requested row window, return exactly the
    items the paper used (same prompts, gold and token budgets), offline. The live HF datasets-server is only
    consulted for windows the snapshot does not cover. Row-window semantics match the live loaders: items whose
    source row index lies in [offset, offset + n), in row order (a loader may skip unparseable rows)."""
    snap = _snapshot(); rows = snap.get("__windows__", {}).get(dataset)
    if rows is not None and offset + n <= rows:
        out = [it for q, it in snap.items() if q != "__windows__" and it["dataset"] == dataset
               and offset <= int(q.rsplit(":", 1)[1]) < offset + n]
        return sorted(out, key=lambda it: int(it["qid"].rsplit(":", 1)[1]))
    return LOADERS[dataset](n, offset)
