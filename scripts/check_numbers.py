"""Integrity gate: recompute every paper number from the released data and compare with the committed
paper/numbers.tex and paper/numbers_pillars.tex. Exits non-zero (and prints each differing macro) if anything differs.
Run after scripts/fetch_data.py:  python3 scripts/check_numbers.py
"""
import os, re, sys, subprocess, tempfile, shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H = os.path.join(ROOT, "harness")
FILES = [os.path.join(ROOT, "paper", f) for f in ("numbers.tex", "numbers_pillars.tex")]


def macros(path):
    return dict(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", open(path).read()))


def facts():
    """numbers the paper states in prose rather than through a macro, recomputed from the released data"""
    import csv, json, statistics
    R = os.path.join(ROOT, "runs")
    reg = list(csv.DictReader(l for l in open(os.path.join(ROOT, "market_registry.csv")) if not l.startswith("#")))
    pool = list(csv.DictReader(l for l in open(os.path.join(ROOT, "model_registry.csv")) if not l.startswith("#")))
    mix = {m for v in json.load(open(os.path.join(R, "matrix_stageA2v3_final.json"))).values() for m in v["models"]}
    code = json.load(open(os.path.join(R, "codegen_problems.json")))
    E3 = json.load(open(os.path.join(R, "matrix_marketE3_final.json")))
    items = json.load(open(os.path.join(ROOT, "gpqa_open_v2", "items_v2.json")))
    lab = lambda x: sum(1 for i in items if i["label"] == x)
    why = lambda x: sum(1 for i in items if i.get("exclude_reason") == x)
    return {
        "67 market models": (67, len(reg)), "abstract: 7 named models and 60 more": (60, len(reg) - 7), "21 provider families": (21, len({r["provider_family"] for r in reg})),
        "15-model pool": (15, len(mix)), "9 families in the 15-model pool":
            (9, len({r["provider_family"] for r in pool if r["model_id"] in mix})),
        "63 code problems": (63, len(code)), "140 code problems fetched":
            (140, len(json.load(open(os.path.join(R, "codegen_problems_raw.json"))))), "code rating from 1900": (1900, min(int(p["cf_rating"]) for p in code)),
        "code rating to 3500": (3500, max(int(p["cf_rating"]) for p in code)),
        "median 20 tests per code problem": (20, statistics.median(len(__import__("ast").literal_eval(p["tests"])) if isinstance(p["tests"], str) else len(p["tests"]) for p in code)),
        "60 AIME questions": (60, sum(1 for v in E3.values() if v["dataset"] in ("aime24", "aime25"))),
        "12 AIME-2024 questions never run": (12, sum(1 for v in E3.values() if v["dataset"] == "aime24" and not v["models"])),
        "AIME-2025 q27 answered by 51 models": (51, sum(1 for c in E3["aime25:27"]["models"].values() if not c.get("missing"))),
        "AIME-2025 q27: 30 answers at the token budget": (30, sum(1 for c in E3["aime25:27"]["models"].values()
            if c.get("max_tokens") == 32768 or c.get("tok_out", 0) >= (c.get("max_tokens") or 1e9) - 8)),
        "AIME-2025 q27 solved by six models at 32,768 tokens": (6, sum(c["correct"] for c in E3["aime25:27"]["models"].values()
                                                                   if c.get("max_tokens") == 32768)),
        "130 GPQA questions screened": (130, len(items)), "80 KEEP": (80, lab("KEEP")), "5 REWRITE": (5, lab("REWRITE")),
        "45 EXCLUDE": (45, lab("EXCLUDE")), "22 option-dependent": (22, why("option-dependent")),
        "23 doubtful reference": (23, why("non-unique or doubtful gold")),
    }


if __name__ == "__main__":
    bad_facts = {k: v for k, v in facts().items() if v[0] != v[1]}
    for k, (stated, data) in bad_facts.items():
        print(f"  FACT MISMATCH {k}: paper states {stated}, data gives {data}")
    print(f"[check] {len(facts())} prose facts checked; {len(bad_facts)} differ.")
    keep = {}
    for f in FILES:
        keep[f] = tempfile.mktemp(suffix=".tex"); shutil.copy(f, keep[f])
    try:
        subprocess.run([sys.executable, "canonical.py"], cwd=H, check=True)
        subprocess.run([sys.executable, "pillars.py"], cwd=H, check=True)
        new, old = {}, {}
        for f in FILES:
            new.update(macros(f)); old.update(macros(keep[f]))
    finally:
        for f in FILES:
            shutil.copy(keep[f], f)          # never silently overwrite the committed files
    diff = {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
    for k, (a, b) in diff.items():
        print(f"  MISMATCH \\{k}: paper has {a!r}, recomputed {b!r}")
    print(f"[check] {len(new)} numbers recomputed; {len(diff)} differ from the paper.")
    sys.exit(1 if diff or bad_facts else 0)
