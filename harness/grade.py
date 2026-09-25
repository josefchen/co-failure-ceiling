"""Grader used by every script (version 2, 2026-09-25). The version-1 grader is kept unchanged in grade_v1.py for
audit; this module fixes its math and multiple-choice defects and re-exports everything else from it.

v1 defects found in the audit:
  * math extraction: the \\boxed{...} regex handled one level of nested braces, so \\boxed{\\frac{3\\sqrt{3}}{4}} fell
    through to the last line of the response;
  * math equivalence: no answer normalization, so a correct "5" failed against the reference "x=5", "10080" against
    "10,\\!080", "864" against "864 \\mbox{ inches}^2", "5.4" against "5.4 \\text{ cents}", "\\frac{4}{3}" against
    "\\frac43", and "1+\\sqrt{19}, 1-\\sqrt{19}" against "1 \\pm \\sqrt{19}". A reference that no answer can match makes
    every model wrong on that question, which looks exactly like a common-mode co-failure;
  * multiple choice: "**Answer:** A" and "The final answer is: I" returned no letter.
v2 uses balanced-brace extraction and the answer normalization of the MATH benchmark's reference evaluation code
(Hendrycks et al., 2021: strip_string), extended with unit/text removal, thousands separators, +- expansion and
order-free comparison of comma-separated answer lists; sympy equivalence is kept as a fallback.
"""
import re
import grade_v1 as v1
from grade_v1 import grade_codegen, grade_code, _norm_out, _CODE_BAD, _boxed, _to_expr, _last_number  # noqa: F401 (re-exported)


# ---------------- extraction ----------------
def last_boxed(s):
    """content of the last \\boxed{...} (or \\fbox{...}) with balanced braces; None if absent."""
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def extract_math(output):
    b = last_boxed(output or "")
    if b is not None:
        return b.strip()
    return v1._boxed(output or "")        # v1 fallbacks: "answer is ...", then last line


# ---------------- normalization (MATH reference strip_string, extended) ----------------
def _fix_fracs(s):
    s = re.sub(r"\\frac\{([^{}]*)\}([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)   # \frac{270}7 -> \frac{270}{7}
    subs = s.split("\\frac"); new = subs[0]
    for sub in subs[1:]:
        new += "\\frac"
        if sub and sub[0] == "{":
            new += sub
        elif len(sub) >= 2:
            a, b = sub[0], sub[1]
            if b != "{":
                new += "{" + a + "}{" + b + "}" + sub[2:]
            else:
                new += "{" + a + "}" + sub[1:]
        else:
            new += sub
    return new


def _fix_sqrt(s):
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def _fix_a_slash_b(s):
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return "\\frac{%s}{%s}" % (m.group(1), m.group(2)) if m else s


def strip_string(s):
    s = str(s).replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"\\(?:text|mbox|mathrm|textbf)\{\s*(?:inches|inch|cm|meters?|feet|foot|units?|square|sq|dollars?|cents?|"
               r"degrees?|hours?|minutes?|seconds?|days?|miles?|mph|km|pounds?|ounces?|years?|people|ways|students)[^}]*\}"
               r"(\^\{?\d\}?)?", "", s)
    s = re.sub(r"\\(?:text|mbox|mathrm|textbf)\{\s*([^}]*)\}", r"\1", s)     # keep other \text{...} content
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\s*(dollars?|cents?|degrees?|inches|units?)\s*$", "", s.strip())
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    parts = s.split("=")
    if len(parts) == 2 and re.fullmatch(r"\s*[a-zA-Z]\s*(\(\s*[a-zA-Z]\s*\))?\s*", parts[0]):
        s = parts[1]                                  # "x=5", "k(x)=..." -> right-hand side
    s = _fix_sqrt(s).replace(" ", "")
    s = _fix_fracs(s)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")                        # thousands separators
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    s = s.rstrip(".")
    return s


def _alternatives(s):
    """a \\pm b -> {a+b, a-b}; comma/semicolon lists (outside brackets) -> sorted tuple of items."""
    if "\\pm" in s and s.count("\\pm") == 1:
        a, b = s.split("\\pm")
        return tuple(sorted({a + "+" + b, a + "-" + b}))
    depth = 0; items = []; cur = ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch in ",;" and depth == 0:
            items.append(cur); cur = ""
        else:
            cur += ch
    items.append(cur)
    items = [i for i in (x.replace("\\text{and}", "").replace("and", "") for x in items) if i]
    return tuple(sorted(items)) if len(items) > 1 else (s,)


def _sym_equal(a, b):
    try:
        from sympy import simplify
        return simplify(v1._to_expr(a) - v1._to_expr(b)) == 0
    except Exception:
        return False


def math_equiv(cand, gold):
    if cand is None:
        return False
    c, g = strip_string(cand), strip_string(gold)
    if c == g:
        return True
    ca, ga = _alternatives(c), _alternatives(g)
    if len(ca) == len(ga) and (ca == ga or all(any(x == y or _sym_equal(x, y) for y in ga) for x in ca)
                               and all(any(x == y or _sym_equal(x, y) for x in ca) for y in ga)):
        return True
    return _sym_equal(c, g)


def grade_math(output, gold):
    return int(math_equiv(extract_math(output), gold) or v1.grade_math(output, gold) == 1)


# ---------------- multiple choice ----------------
def mc_letter(output):
    """grade_v1._mc_letter with the answer-marker pattern widened (same precedence as v1: a lone letter, then \\boxed,
    then answer markers, ...). v1 missed "**Answer:** A" (markdown between the marker and the letter) and
    "The final answer is: I" ("is" followed by a colon)."""
    up = (output or "").upper()
    _LONE = r"^\(?\*{0,2}\s*([A-J])\s*\*{0,2}\s*[).:]?\.?$"
    m0 = re.match(_LONE, up.strip())
    if m0:
        return m0.group(1)
    for pat in (r"\\?BOXED\{?\s*([A-J])\b",
                r"ANSWER\s*(?:IS)?\s*[:=]?\s*[*_]*\s*\(?\s*([A-J])\b",
                r"\bOPTION\s*\(?\s*([A-J])\b",
                r"\*\*\s*([A-J])\s*\*\*"):
        m = re.findall(pat, up)
        if m:
            return m[-1]
    return v1._mc_letter(output or "")


def grade_mc(output, gold):
    c = mc_letter(output)
    return int(c is not None and c == str(gold).strip().upper())


def num_answer(output):
    out = output or ""
    if "####" in out:
        seg = out.split("####")[-1]
        line = next((ln for ln in seg.splitlines() if ln.strip()), "")
        nums = re.findall(r"-?\d[\d,]*\.?\d*", line)
        if nums:
            return nums[-1].replace(",", "").rstrip(".")
    return v1._num(out)


def grade_number(output, gold):
    a = num_answer(output)
    try:
        return int(a is not None and abs(float(a) - float(str(gold).replace(",", ""))) < 1e-6)
    except (TypeError, ValueError):
        return 0


def grade(kind, output, gold):
    if kind == "number":
        return grade_number(output, gold)
    if kind == "math":
        return grade_math(output, gold)
    if kind == "mc":
        return grade_mc(output, gold)
    return v1.grade(kind, output, gold)


def extract(kind, output):
    if kind == "number":
        return num_answer(output)
    if kind == "mc":
        return mc_letter(output)
    if kind == "math":
        b = extract_math(output)
        if b is None:
            return None
        n = strip_string(b)
        try:                                   # canonical sympy form so equivalent answers vote together (as in v1)
            return str(v1._to_expr(n))
        except Exception:
            return n or None
    return v1.extract(kind, output)


def check(kind, cand, gold):
    if cand is None:
        return 0
    if kind == "math":
        return int(math_equiv(cand, gold))
    return v1.check(kind, cand, gold)
