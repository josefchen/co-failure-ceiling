"""Fill the numbers in README.md, DATA_AUDIT.md and the Space page from the generated paper macros, so that no
document states a number the pipeline did not produce. Templates use {{MacroName}} (the macro without its leading \\n).
Usage: python3 scripts/render_docs.py        (after harness/canonical.py and harness/pillars.py)
"""
import os, re, sys, json

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAIRS = [("docs/README.template.md", "README.md"), ("docs/DATA_AUDIT.template.md", "DATA_AUDIT.md"),
         ("docs/space_index.template.html", "space/index.html"), ("docs/dataset_card.template.md", "release/dataset_card.md"),
         ("docs/arxiv_payload_readme.template.md", "arxiv_payload/PAYLOAD_README.md")]


def macros():
    m = {}
    for f in ("paper/numbers.tex", "paper/numbers_pillars.tex"):
        for k, v in re.findall(r"\\newcommand\{\\n([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", open(os.path.join(ROOT, f)).read()):
            m[k] = re.sub(r"\\texttt\{([^}]*)\}", r"\1", v)
    rev = re.search(r'REVISION = "([^"]+)"', open(os.path.join(ROOT, "scripts", "fetch_data.py")).read()).group(1)
    m["HfRevision"] = rev                                   # the pinned dataset commit the numbers were computed from
    m["PaperAbstract"] = plain_abstract(m)                   # the arXiv form's abstract, generated from paper/main.tex
    return m


def plain_abstract(m):
    """paper/main.tex's abstract as plain text (arXiv's abstract field), macros filled in, wrapped at 120 characters"""
    import textwrap
    t = open(os.path.join(ROOT, "paper", "main.tex")).read()
    t = t[t.index("\\begin{abstract}") + len("\\begin{abstract}"):t.index("\\end{abstract}")]
    t = re.sub(r"\\n([A-Za-z]+)", lambda mo: m.get(mo.group(1), mo.group(0)), t)
    t = re.sub(r"\s*\((?:see )?App\.~\\ref\{[^}]*\}\)", "", t)
    for a, b in [("\\beta", "beta"), ("\\rho", "rho"), ("{<}\\,", "<"), ("\\$", "\x00"), ("--", "-"), ("\\%", "%"), ("~", " "),
                 ("\\emph{", "{"), ("\\texttt{", "{"), ("$", ""), ("\x00", "$")]:
        t = t.replace(a, b)
    t = re.sub(r"[{}]", "", t); t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 1920:
        print(f"[docs] WARNING: abstract is {len(t)} characters; arXiv accepts at most 1920")
    return "\n".join(textwrap.wrap(t, 120, break_on_hyphens=False))


if __name__ == "__main__":
    M = macros(); missing = set()
    for src, dst in PAIRS:
        p = os.path.join(ROOT, src)
        if not os.path.exists(p):
            continue
        text = open(p).read()
        def sub(mo):
            k = mo.group(1)
            if k not in M:
                missing.add(k); return mo.group(0)
            return M[k]
        C = json.load(open(os.path.join(ROOT, "runs", "canonical.json")))
        def jsub(mo):
            node = C
            for part in mo.group(1).split("."):
                node = node[part]
            return json.dumps(node, separators=(",", ":"))
        text = re.sub(r"\{\{json:([A-Za-z0-9_.]+)\}\}", jsub, text)     # {{json:dotted.path}} -> canonical.json value
        out = re.sub(r"\{\{([A-Za-z]+)\}\}", sub, text)
        os.makedirs(os.path.dirname(os.path.join(ROOT, dst)) or ".", exist_ok=True)
        open(os.path.join(ROOT, dst), "w").write(out)
        print(f"[docs] {src} -> {dst}")
    if missing:
        print("[docs] UNKNOWN placeholders:", sorted(missing)); sys.exit(1)
