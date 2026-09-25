"""Special checkers for the five code_contests problems that accept more than one correct output (the all-wrong code
problems of the first release). The exact-match grader scores a different valid answer as wrong; these checkers accept
any output that satisfies the problem statement, as the official Codeforces checkers do. Each checker gets the test
input, the reference output (used only for the feasibility verdict or the optimal value) and the candidate output.

`regrade()` re-runs every model's released program on the same tests, in the same sandbox and time limit as
grade_v1.grade_codegen, and judges each output with the checker. Writes runs/code_checker_regrade.json.
Usage: python3 code_checkers.py   (needs responses/responses.jsonl.gz: scripts/fetch_data.py --responses)
"""
import os, re, sys, json, gzip, tempfile, subprocess
import grade_v1

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); RUNS = os.path.join(ROOT, "runs")


class Tok:
    def __init__(self, s):
        self.t = s.split(); self.i = 0
    def next(self):
        if self.i >= len(self.t):
            raise ValueError("output too short")
        self.i += 1
        return self.t[self.i - 1]
    def int(self):
        return int(self.next())
    def done(self):
        return self.i == len(self.t)


def xor_of_3(inp, ref, out):                      # 1572B
    I, E, O = Tok(inp), Tok(ref), Tok(out)
    for _ in range(I.int()):
        n = I.int(); a = [I.int() for _ in range(n)]
        e = E.next().upper()
        if e == "YES":
            k = E.int(); [E.next() for _ in range(k)]
        c = O.next().upper()
        if c != e:
            return False
        if c == "NO":
            continue
        k = O.int()
        if not 0 <= k <= n:
            return False
        for _ in range(k):
            b = O.int()
            if not 1 <= b <= n - 2:
                return False
            x = a[b - 1] ^ a[b] ^ a[b + 1]; a[b - 1] = a[b] = a[b + 1] = x
        if any(a):
            return False
    return O.done()


def expression_error(inp, ref, out):              # 1567D
    I, E, O = Tok(inp), Tok(ref), Tok(out)
    b11 = lambda x: int(str(x), 11)
    for _ in range(I.int()):
        s, n = I.int(), I.int()
        best = sum(b11(E.int()) for _ in range(n))
        xs = [O.int() for _ in range(n)]
        if any(x <= 0 for x in xs) or sum(xs) != s or sum(b11(x) for x in xs) != best:
            return False
    return O.done()


def bipartite_array(inp, ref, out):               # 1620F
    I, E, O = Tok(inp), Tok(ref), Tok(out)
    for _ in range(I.int()):
        n = I.int(); p = [I.int() for _ in range(n)]
        e = E.next().upper()
        if e == "YES":
            [E.next() for _ in range(n)]
        c = O.next().upper()
        if c != e:
            return False
        if c == "NO":
            continue
        a = [O.int() for _ in range(n)]
        if any(abs(x) != y for x, y in zip(a, p)):
            return False
        # distinct values: the inversion graph is a permutation graph, bipartite iff no decreasing subsequence of length 3
        pre, suf = [float("-inf")] * n, [float("inf")] * n
        for i in range(1, n):
            pre[i] = max(pre[i - 1], a[i - 1])
        for i in range(n - 2, -1, -1):
            suf[i] = min(suf[i + 1], a[i + 1])
        if any(pre[j] > a[j] > suf[j] for j in range(n)):
            return False
    return O.done()


def weights(inp, ref, out):                       # 1599A
    I, O = Tok(inp), Tok(out)
    n = I.int(); A = [I.int() for _ in range(n)]; S = I.next()
    if ref.split() == ["-1"]:
        return out.split() == ["-1"]
    left = right = 0; used = []
    for i in range(n):
        w, side = O.int(), O.next().upper()
        used.append(w)
        if side == "L":
            left += w
        elif side == "R":
            right += w
        else:
            return False
        if (S[i] == "L" and not left > right) or (S[i] == "R" and not right > left):
            return False
    return sorted(used) == sorted(A) and O.done()


def moment_of_bloom(inp, ref, out):               # 1586E
    I, E, O = Tok(inp), Tok(ref), Tok(out)
    n, m = I.int(), I.int()
    edges = set()
    for _ in range(m):
        x, y = I.int(), I.int(); edges.add((min(x, y), max(x, y)))
    q = I.int(); Q = [(I.int(), I.int()) for _ in range(q)]
    e = E.next().upper(); c = O.next().upper()
    if c != e:
        return False
    if c == "NO":
        return O.int() == E.int() and O.done()
    par = {}
    for a, b in Q:
        x = O.int(); path = [O.int() for _ in range(x)]
        if x < 2 or path[0] != a or path[-1] != b or len(set(path)) != x:
            return False
        for u, v in zip(path, path[1:]):
            k = (min(u, v), max(u, v))
            if k not in edges:
                return False
            par[k] = par.get(k, 0) ^ 1
    return not any(par.values()) and O.done()


CHECKERS = {"codegen:61": xor_of_3, "codegen:58": expression_error, "codegen:35": bipartite_array,
            "codegen:14": weights, "codegen:5": moment_of_bloom}


def run_program(code, tests, time_limit):
    """run one program on every test exactly as grade_v1.grade_codegen does; returns the outputs or None"""
    if any(b in code.lower() for b in grade_v1._CODE_BAD):
        return None
    timeout = min(max(float(time_limit or 4) * 3.0, 4.0), 12.0)
    d = tempfile.mkdtemp(prefix="cc_"); path = os.path.join(d, "sol.py")
    try:
        open(path, "w").write(code); outs = []
        for t in tests:
            try:
                r = subprocess.run([sys.executable, path], input=t["input"], capture_output=True, text=True,
                                   timeout=timeout, cwd=d, preexec_fn=grade_v1._sandbox(cpu_s=int(timeout) + 2))
            except Exception:
                return None
            if r.returncode != 0:
                return None
            outs.append(r.stdout)
        return outs
    finally:
        try:
            os.unlink(path); os.rmdir(d)
        except Exception:
            pass


def check(q, tests, outs):
    try:
        return all(CHECKERS[q](t["input"], t["output"], o) for t, o in zip(tests, outs))
    except Exception:
        return False


def regrade():
    S = json.load(open(os.path.join(RUNS, "items_snapshot.json")))
    progs = {}
    for line in gzip.open(os.path.join(ROOT, "responses", "responses.jsonl.gz"), "rt"):
        r = json.loads(line)
        if r["matrix"] == "matrix_marketCG_final.json" and r["qid"] in CHECKERS:
            progs[(r["qid"], r["model"])] = r["content"] or ""
    out = {}
    for q in CHECKERS:
        g = json.loads(S[q]["gold"]); tests = g["tests"]
        assert all(CHECKERS[q](t["input"], t["output"], t["output"]) for t in tests), f"checker rejects the reference on {q}"
        res = {}
        for (qq, m), content in sorted(progs.items()):
            if qq != q:
                continue
            blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", content, re.DOTALL)
            outs = run_program(blocks[-1] if blocks else content, tests, g.get("time_limit", 4))
            res[m] = {"correct": int(outs is not None and check(q, tests, outs)), "ran": outs is not None}
        out[q] = {"tests": len(tests), "models": res, "n_correct": sum(v["correct"] for v in res.values())}
        print(f"[checker] {q}: {out[q]['n_correct']} of {len(res)} models produce a valid answer on all {len(tests)} tests", flush=True)
    json.dump(out, open(os.path.join(RUNS, "code_checker_regrade.json"), "w"), indent=1)
    return out


def apply(res):
    """write the checker verdicts into runs/matrix_marketCG_final.json (flag "checker") and log every changed cell to
    runs/rerun_log.csv (step "code-checker")"""
    import csv, datetime
    f = os.path.join(RUNS, "matrix_marketCG_final.json"); R = json.load(open(f))
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(os.path.join(RUNS, "rerun_log.csv"), "a", newline="") as fh:
        w = csv.writer(fh)
        for q, v in res.items():
            for m, r in v["models"].items():
                c = R[q]["models"][m]; old = c["correct"]
                c["correct"] = r["correct"]; c["checker"] = True
                if old != r["correct"]:
                    w.writerow([stamp, "code-checker", "matrix_marketCG_final.json", q, m, old, r["correct"], False,
                                c.get("tok_in"), c.get("tok_out"), "0.000000"])
    json.dump(R, open(f, "w"))


if __name__ == "__main__":
    apply(regrade())
