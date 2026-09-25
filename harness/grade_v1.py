"""Programmatic graders -> correctness in {0,1}. Answer-ANCHORED extraction (v2):
 - MC: prefer an explicit answer marker; else the LAST standalone letter (not the first),
   so verbose/reasoning models that reason-then-answer are not penalized.
 - Number: prefer the post-'####' segment, then answer markers, then last number.
 - Math: sympy-based symbolic equivalence (fractions, sqrt, pi, powers), with numeric fallback.
 - Code: sandboxed execution (deferred from the live runs).
No LLM judge anywhere. v2 fixes the verbosity-correlated extraction bias flagged in review.
"""
import re, subprocess, sys, tempfile, os, json

def _last_number(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return nums[-1] if nums else None

# ---------- multiple choice ----------
def _mc_letter(output):
    up = output.upper()
    _LONE = r"^\(?\*{0,2}\s*([A-J])\s*\*{0,2}\s*[).:]?\.?$"  # a string that IS one option letter
    # 0) the whole answer is a single option letter, e.g. "I", "(I)", "**A**", "C." -- credit it
    #    even if it is I/A (those are excluded only as prose pronoun/article in stage 3).
    m0 = re.match(_LONE, up.strip())
    if m0:
        return m0.group(1)
    # 1) explicit answer markers (take the LAST such, i.e. the final declared answer)
    for pat in (r"\\?BOXED\{?\s*([A-J])\b",
                r"ANSWER\s*(?:IS|:|=)?\s*\(?\s*([A-J])\b",
                r"\bOPTION\s*\(?\s*([A-J])\b",
                r"\*\*\s*([A-J])\s*\*\*"):
        m = re.findall(pat, up)
        if m:
            return m[-1]
    # 1b) the LAST non-empty line is a lone option letter -- reasoning models routinely end
    #     with the bare answer on its own line; credit I/A here too.
    lines = [ln.strip() for ln in up.splitlines() if ln.strip()]
    if lines:
        ml = re.match(_LONE, lines[-1])
        if ml:
            return ml.group(1)
    # 2) letters with an option cue: "(C)", "C)", "C." -- take the last
    m = re.findall(r"\(?\b([A-J])\b\s*[).:]", up)
    if m:
        return m[-1]
    # 3) last standalone letter, EXCLUDING the pronoun 'I' and the article 'A' to avoid
    #    prose false-positives; also drop enumerations like "(A, B, C, D)" by removing
    #    comma/space-separated runs of single letters before scanning.
    cleaned = re.sub(r"\(?\s*(?:[A-J]\s*,\s*){2,}[A-J]\s*\)?", " ", up)  # kill "A, B, C, D" runs
    m = [x for x in re.findall(r"\b([A-J])\b", cleaned) if x not in ("I", "A")]
    return m[-1] if m else None

def grade_mc(output, gold):
    c = _mc_letter(output)
    return int(c is not None and c == str(gold).strip().upper())

# ---------- numeric (gsm8k) ----------
def _num(output):
    if "####" in output:
        seg = output.split("####")[-1].split("\n")[0]  # the GSM8K answer is right after ####, same line
        m = re.findall(r"-?\d[\d,]*\.?\d*", seg)
        if m:
            return m[0].replace(",", "")               # FIRST number in the segment, not last
    m = re.findall(r"(?:answer|result|total|equals?)\s*(?:is|:|=)?\s*\$?\s*(-?\d[\d,]*\.?\d*)", output, re.I)
    if m:
        return m[-1].replace(",", "")
    return _last_number(output)

def grade_number(output, gold):
    cand = _num(output)
    if cand is None:
        return 0
    try:
        return int(abs(float(cand) - float(gold)) < 1e-6)
    except ValueError:
        return int(str(cand).strip() == str(gold).strip())

# ---------- competition math (sympy equivalence) ----------
_MATH_SAFE = re.compile(r"^[0-9A-Za-z+\-*/^().,{}\\\s_=\[\].!|]*$")  # conservative whitelist
_MATH_BAD = ("import", "__", "lambda", "eval", "exec", "open", "system", "subprocess", "os.", "sys.",
             "print", "input", "compile", "globals", "locals", "getattr", "setattr")

def _to_expr(s):
    """Parse a LaTeX-ish math string to a sympy expr WITHOUT eval/builtins (security: model
    output is untrusted -- a prior version allowed code execution via sympify)."""
    if any(b in s for b in _MATH_BAD):
        raise ValueError("rejected: unsafe token")
    s = s.strip().strip("$").replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for _ in range(6):
        s2 = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    s = s.replace("\\%", "/100").replace("%", "/100")
    s = s.replace("{", "(").replace("}", ")").replace("\\", "")
    if not _MATH_SAFE.match(s):
        raise ValueError("rejected: non-whitelisted chars")
    from sympy.parsing.sympy_parser import parse_expr
    # parse_expr (unlike sympify) does NOT eval Python -- it builds sympy objects -- so it is
    # not a code-exec vector; the _MATH_BAD/_MATH_SAFE guards are defense in depth.
    return parse_expr(s, evaluate=True)

def _boxed(output):
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", output)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer|result)\s*(?:is|:|=)?\s*\$?([^\n.]+)", output, re.I)
    if m:
        return m[-1].strip()
    # last non-empty line (let _to_expr parse structure); do NOT collapse to a bare number
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _is_pure_number(x):
    return bool(re.fullmatch(r"\s*-?\$?\d[\d,]*\.?\d*\s*", str(x)))

def grade_math(output, gold):
    cand = _boxed(output)
    g = str(gold).strip()
    if cand.strip() == g:
        return 1
    try:
        from sympy import simplify
        if simplify(_to_expr(cand) - _to_expr(g)) == 0:
            return 1
    except Exception:
        pass
    # numeric fallback ONLY when BOTH sides are purely numeric (avoid x=3 vs y=3 false-accepts)
    if _is_pure_number(cand) and _is_pure_number(g):
        try:
            return int(abs(float(_last_number(cand)) - float(_last_number(g))) < 1e-6)
        except (ValueError, TypeError):
            return 0
    return 0

# ---------- code (sandboxed; deferred from live runs) ----------
def grade_code(output, gold_tests_json):
    m = re.search(r"```(?:python)?\s*(.*?)```", output, re.DOTALL)
    code = m.group(1) if m else output
    tests = json.loads(gold_tests_json)
    prog = code + "\n" + "\n".join(tests) + "\nprint('PASS_OK')\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        ok = int("PASS_OK" in r.stdout and r.returncode == 0)
    except Exception:
        ok = 0
    finally:
        try: os.unlink(path)
        except Exception: pass
    return ok

# ---------- competitive code: stdin->stdout, sandboxed (no fs/network), exact-token match ----------
_CODE_BAD = ("subprocess", "socket", "urllib", "requests", "shutil", "os.system", "os.popen",
             "os.remove", "os.rmdir", "ctypes", "import pty", "__import__", "eval(", "exec(",
             "open(", "Path(", "pathlib")  # algorithmic solutions need none of these

def _norm_out(s):
    # whitespace-normalized token sequence: robust to trailing spaces / blank lines
    return [tok for tok in s.split()]

def _sandbox(mem_bytes=1536 * 1024 * 1024, cpu_s=12):
    """preexec for the grading subprocess: cap address space + CPU so a generated solution that
    over-allocates (mis-read constraints on a hard problem) dies with MemoryError -> graded wrong,
    instead of OOM-killing the whole run's process group. Unix only."""
    import resource
    def _set():
        for res, lim in ((resource.RLIMIT_AS, mem_bytes), (resource.RLIMIT_CPU, cpu_s)):
            try:
                resource.setrlimit(res, (lim, lim))
            except Exception:
                pass
    return _set

def grade_codegen(output, gold_json):
    g = json.loads(gold_json)
    tests = g.get("tests", [])
    if not tests:
        return 0
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", output, re.DOTALL)
    code = blocks[-1] if blocks else output
    low = code.lower()
    if any(b in low for b in _CODE_BAD):   # refuse to execute anything beyond pure computation
        return 0
    tl = float(g.get("time_limit", 4)) or 4
    timeout = min(max(tl * 3.0, 4.0), 12.0)  # STRICT but Python-fair: 3x the (C++-calibrated) official limit,
                                             # so correct-but-slower Python is not failed -- avoids manufacturing
                                             # co-failure via a language-unfair TLE; capped at 12s for bounded runtime.
    import tempfile
    d = tempfile.mkdtemp(prefix="cg_")
    path = os.path.join(d, "sol.py")
    try:
        with open(path, "w") as f:
            f.write(code)
        for t in tests:
            try:
                r = subprocess.run([sys.executable, path], input=t["input"], capture_output=True,
                                   text=True, timeout=timeout, cwd=d,
                                   preexec_fn=_sandbox(cpu_s=int(timeout) + 2))
            except Exception:
                return 0
            if r.returncode != 0:
                return 0
            if _norm_out(r.stdout) != _norm_out(t["output"]):
                return 0
        return 1   # passed every checked test
    finally:
        try:
            os.unlink(path); os.rmdir(d)
        except Exception:
            pass

def grade(kind, output, gold):
    # "open" => LLM-judge-graded post-hoc (judge_open.py over cached answers); 0 stub here so the matrix
    # run just collects answers.
    if kind == "open":
        return 0
    return {"number": grade_number, "mc": grade_mc, "math": grade_math, "code": grade_code,
            "codegen": grade_codegen}[kind](output, gold)

# ---------- extraction for majority vote ----------
def extract(kind, output):
    if kind == "number":
        return _num(output)
    if kind == "mc":
        return _mc_letter(output)
    if kind == "math":
        b = _boxed(output)
        try:
            return str(_to_expr(b))  # canonical sympy form so equal answers vote together
        except Exception:
            return b.strip() or None
    return None

def check(kind, cand, gold):
    if cand is None:
        return 0
    if kind == "number":
        try:
            return int(abs(float(cand) - float(gold)) < 1e-6)
        except ValueError:
            return int(str(cand).strip() == str(gold).strip())
    if kind == "mc":
        return int(str(cand).strip().upper() == str(gold).strip().upper())
    if kind == "math":
        return grade_math(str(cand), gold)
    return 0
