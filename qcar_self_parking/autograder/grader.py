#!/usr/bin/env python3
"""
Auto-grader for the 5 short Python problems (strings/lists, sets, dicts,
try-except, zip/sort).  Each problem is worth 20 points (100 total).

USAGE
  python grader.py SUBMISSION            # one student
  python grader.py SUBMISSIONS_DIR --batch --csv grades.csv   # many students
  add -v to also list the checks that passed.

SUBMISSION can be
  * a single .py file whose problems are separated by marker comments such as
        # Problem 1        (also "# Q2", "# P3", "# Question 4" ...)
  * a folder with one file per problem; the digit 1-5 in the file name picks
    the problem (p1.py, problem2.py, q_3.py, jane_p4.py, 5.py ...)
With --batch every child of SUBMISSIONS_DIR (file or folder) is one student.

HOW IT GRADES
  Static checks (AST) for required constructs  - e.g. set.add/update, the `&`
  operator, no loop for the split, try/except/else/finally structure.
  Dynamic checks: each student's code is executed in a throw-away subprocess
  (timeout, `python -I`, temp working dir).  Results are read back from the
  student's variables / stdout.  Hidden tests re-run the code with the input
  data swapped for different values, which catches hard-coded answers.

SECURITY: student code is really executed.  There is a timeout and a temp cwd
but NO true sandbox -- run this in a VM/container for untrusted submissions.
Requires Python 3.9+ (ast.unparse).
"""
import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
import io
from pathlib import Path

TIMEOUT = 10  # seconds per student execution
POINTS_PER_PROBLEM = 20

# --------------------------------------------------------------------------
# Subprocess runner (executed with `python -I runner.py`)
# --------------------------------------------------------------------------
RUNNER = r'''
import sys, io, json, builtins, traceback, contextlib, os, random
cfg = json.loads(os.environ["GRADER_CFG"])
_open, _float = builtins.open, builtins.float
src = _open(cfg["src"], encoding="utf-8").read()
b = dict(vars(builtins))
tracked = []
pre = None
def t_open(*a, **k):
    f = _open(*a, **k)
    tracked.append(f)
    return f
if cfg.get("track_files"):
    b["open"] = t_open
if cfg.get("float_boom"):          # float("...BOOM...") raises RuntimeError
    def f_float(x=0.0):
        if isinstance(x, str) and "BOOM" in x:
            raise RuntimeError("boom")
        return _float(x)
    b["float"] = f_float
ns = {"__builtins__": b, "__name__": "__main__"}
if cfg.get("inf_path"):
    pre = _open(cfg["inf_path"], "r")   # used only if the student never opens a file
    ns["inf"] = pre
if cfg.get("seed") is not None:
    random.seed(cfg["seed"])
buf = io.StringIO()
err = None
try:
    code = compile(src, "<student>", "exec")
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
except BaseException:
    err = traceback.format_exc(limit=3)

def enc(o, d=0):
    if d > 6:
        return {"__repr__": repr(o)[:200]}
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if type(o) is list:
        return [enc(x, d + 1) for x in o]
    if type(o) is tuple:
        return {"__tuple__": [enc(x, d + 1) for x in o]}
    if type(o) in (set, frozenset):
        return {"__set__": [enc(x, d + 1) for x in sorted(o, key=repr)]}
    if type(o) is dict:
        return {"__dict__": [[enc(k, d + 1), enc(v, d + 1)] for k, v in o.items()]}
    return {"__repr__": repr(o)[:200]}

res = {
    "stdout": buf.getvalue(),
    "error": err,
    "vars": {k: enc(v) for k, v in ns.items() if not k.startswith("__")},
    "files_closed": (all(f.closed for f in tracked) if tracked
                     else (pre.closed if pre is not None else None)),
    "n_open": len(tracked),
}
with _open(cfg["out"], "w", encoding="utf-8") as fh:
    json.dump(res, fh)
'''


class Opaque(str):
    """Stand-in for a student object that is not a plain builtin value."""


def dec(o):
    if isinstance(o, list):
        return [dec(x) for x in o]
    if isinstance(o, dict):
        if "__tuple__" in o:
            return tuple(dec(x) for x in o["__tuple__"])
        if "__set__" in o:
            return set(dec(x) for x in o["__set__"])
        if "__dict__" in o:
            return {dec(k): dec(v) for k, v in o["__dict__"]}
        if "__repr__" in o:
            return Opaque(o["__repr__"])
    return o


class Result:
    def __init__(self, d):
        self.stdout = d.get("stdout", "")
        self.error = d.get("error")
        self.vars = {k: dec(v) for k, v in d.get("vars", {}).items()}
        self.files_closed = d.get("files_closed")
        self.n_open = d.get("n_open", 0)

    @property
    def lines(self):
        return [l.strip() for l in self.stdout.splitlines() if l.strip()]


def run_code(src, files=None, timeout=None, **cfg):
    timeout = timeout or TIMEOUT
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "student_src.py").write_text(src, encoding="utf-8")
        (d / "runner.py").write_text(RUNNER, encoding="utf-8")
        wd = d / "work"
        wd.mkdir()
        for name, content in (files or {}).items():
            (wd / name).write_text(content, encoding="utf-8")
        cfg.update(src=str(d / "student_src.py"), out=str(d / "out.json"))
        env = dict(os.environ, GRADER_CFG=json.dumps(cfg))
        try:
            p = subprocess.run([sys.executable, "-I", str(d / "runner.py")],
                               cwd=wd, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result({"error": f"Timed out after {timeout}s (infinite loop?)"})
        out = d / "out.json"
        if not out.exists():
            return Result({"error": "Runner crashed: " + (p.stderr or "")[-300:]})
        return Result(json.loads(out.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------
def try_parse(src):
    try:
        return ast.parse(src), None
    except SyntaxError as e:
        return None, f"SyntaxError line {e.lineno}: {e.msg}"


def assigns(tree, name):
    return [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def call_name(c):
    f = c.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def calls_in(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def substitute(src, name, expr):
    """Replace the value of top-level assignment(s) to `name` by python expr."""
    tree = ast.parse(src)
    hit = False
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            n.value = ast.parse(expr, mode="eval").body
            hit = True
    return (ast.unparse(tree), True) if hit else (src, False)


def norm(s):
    return re.sub(r"\s+", "", s).replace('"', "'")


def prints_text(nodes, text):
    for node in nodes:
        for c in calls_in(node):
            if call_name(c) == "print":
                for a in c.args:
                    for k in ast.walk(a):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                                and text in k.value:
                            return True
    return False


# --------------------------------------------------------------------------
# Rubric bookkeeping
# --------------------------------------------------------------------------
class Problem:
    def __init__(self, n, title):
        self.n, self.title = n, title
        self.items = []   # (desc, pts, earned, note)
        self.infos = []

    def check(self, desc, pts, ok, note=""):
        frac = 1.0 if ok is True else 0.0 if ok is False else float(ok)
        self.items.append((desc, pts, pts * frac, note if frac < 1 else ""))

    def info(self, msg):
        self.infos.append(msg)

    @property
    def earned(self):
        return round(sum(i[2] for i in self.items), 2)

    @property
    def total(self):
        return sum(i[1] for i in self.items)


def syntax_fail(P, err):
    P.check("Code parses and runs", POINTS_PER_PROBLEM, False, err)
    return P


def report_run_error(P, r, label="run"):
    if r.error:
        last = r.error.strip().splitlines()[-1] if r.error.strip() else r.error
        P.info(f"{label}: runtime error -> {last}")


# --------------------------------------------------------------------------
# Problem 1 - split string / word sizes
# --------------------------------------------------------------------------
def grade_p1(src):
    P = Problem(1, "String -> words -> wordSize")
    tree, err = try_parse(src)
    if err:
        return syntax_fail(P, err)
    r = run_code(src)
    report_run_error(P, r)
    v = r.vars
    line, words, sizes = v.get("line"), v.get("words"), v.get("wordSize")

    P.check("(a) `line` is a string with commas and white space", 4,
            isinstance(line, str) and "," in line and re.search(r"\s", line) is not None,
            "need a string variable `line` with comma/space separated words")
    exp = [w.strip() for w in line.split(",")] if isinstance(line, str) else None
    P.check("(b) `words` correct for the student's own line", 4,
            exp is not None and words == exp,
            f"got {words!r}, expected {exp!r} (no white space inside words)")

    # (b) no loop -----------------------------------------------------------
    a = assigns(tree, "words")
    if not a:
        P.check("(b) no loop used for the split", 3, False, "no top-level `words = ...`")
    else:
        idx = tree.body.index(a[0])
        loops = any(isinstance(n, (ast.For, ast.While))
                    for s in tree.body[:idx + 1] for n in ast.walk(s))
        implicit = any(isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp))
                       or (isinstance(n, ast.Call) and call_name(n) in ("map", "filter"))
                       for n in ast.walk(a[0].value))
        if loops:
            P.check("(b) no loop used for the split", 3, False, "for/while loop found")
        elif implicit:
            P.check("(b) no loop used for the split", 3, 0.5,
                    "comprehension/map is an implicit loop - instructor to review")
        else:
            P.check("(b) no loop used for the split", 3, True)

    # (c) own data ------------------------------------------------------------
    P.check("(c) `wordSize` = lengths of `words`", 3,
            isinstance(words, list) and sizes == [len(w) for w in words],
            f"got {sizes!r}")

    # hidden test: different line -------------------------------------------
    test_line = "  alpha ,beta,   gamma  ,  d   ,epsilon  "
    src2, hit = substitute(src, "line", repr(test_line))
    if hit:
        r2 = run_code(src2)
        w2, s2 = r2.vars.get("words"), r2.vars.get("wordSize")
        P.check("hidden test: `words` for a different line", 3,
                w2 == ["alpha", "beta", "gamma", "d", "epsilon"],
                f"on {test_line!r} got {w2!r}")
        P.check("hidden test: `wordSize` for a different line", 3,
                s2 == [5, 4, 5, 1, 7], f"got {s2!r}")
    else:
        P.check("hidden test: `words` for a different line", 3, False, "could not find `line = ...`")
        P.check("hidden test: `wordSize` for a different line", 3, False, "could not find `line = ...`")
    return P


# --------------------------------------------------------------------------
# Problem 2 - sets
# --------------------------------------------------------------------------
def grade_p2(src):
    P = Problem(2, "Set operations")
    tree, err = try_parse(src)
    if err:
        return syntax_fail(P, err)
    r = run_code(src)
    report_run_error(P, r)
    v = r.vars

    ta = assigns(tree, "t")
    P.check("(1) t created as empty set via set()", 3,
            bool(ta) and isinstance(ta[0].value, ast.Call) and call_name(ta[0].value) == "set"
            and not ta[0].value.args,
            "t = {} makes a dict; use t = set()")
    used = any(isinstance(c.func, ast.Attribute) and c.func.attr in ("add", "update")
               and isinstance(c.func.value, ast.Name) and c.func.value.id == "t"
               for c in calls_in(tree))
    P.check("(2) uses t.add(...) or t.update(...)", 3, used)
    P.check("(2) t == {'bye','sad','happy'}", 2, v.get("t") == {"bye", "sad", "happy"},
            f"got {v.get('t')!r}")
    sa = assigns(tree, "s")
    P.check("(3) s created with set(...) factory", 2,
            bool(sa) and isinstance(sa[0].value, ast.Call) and call_name(sa[0].value) == "set"
            and len(sa[0].value.args) == 1)
    P.check("(3) s == set of lst", 2, v.get("s") == {"bye", "hi", "good", "okay", "sad"},
            f"got {v.get('s')!r}")
    wa = assigns(tree, "w")
    op = bool(wa) and any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd)
                          for n in ast.walk(wa[0].value))
    method = any(call_name(c) == "intersection" for c in calls_in(tree))
    P.check("(4) intersection with the & operator (not a method)", 3, op and not method,
            "use w = s & t")
    P.check("(4) w == {'bye','sad'}", 2, v.get("w") == {"bye", "sad"}, f"got {v.get('w')!r}")

    lines = r.lines
    ok_items = False
    if len(lines) == 1:
        items = [x.strip() for x in lines[0].split(",")]
        ok_items = len(items) == 2 and set(items) == {"bye", "sad"}
    P.check("(5) prints elements of w on ONE line", 2, ok_items,
            f"stdout was {r.stdout!r}")
    P.check("(5) comma separated, no brackets/quotes", 1,
            len(lines) == 1 and "," in lines[0] and not re.search(r"[\[\]{}'\"]", lines[0]))
    return P


# --------------------------------------------------------------------------
# Problem 3 - dict counting
# --------------------------------------------------------------------------
def find_str_list(tree):
    for n in tree.body:
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.List)
                and len(n.value.elts) >= 3
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                        for e in n.value.elts)
                and isinstance(n.targets[0], ast.Name)):
            return n.targets[0].id
    return None


def sorted_block_printed(stdout, names):
    toks = [w for w in re.findall(r"[A-Za-z]+", stdout) if w in names]
    k = len(names)
    return any(toks[i:i + k] == names for i in range(len(toks) - k + 1))


def has_number(stdout, n):
    return re.search(rf"(?<![\d.]){n}(?![\d.])", stdout) is not None


def grade_p3(src):
    P = Problem(3, "Dictionary of fruit counts")
    tree, err = try_parse(src)
    if err:
        return syntax_fail(P, err)
    r = run_code(src)
    report_run_error(P, r)
    names = ["apple", "banana", "cherry", "melon"]
    P.check("(a) Fruits counts correct", 4,
            r.vars.get("Fruits") == {"apple": 1, "melon": 3, "banana": 2, "cherry": 1},
            f"got {r.vars.get('Fruits')!r}")
    P.check("(b) sorted names printed (ascending)", 3, sorted_block_printed(r.stdout, names),
            "expected apple, banana, cherry, melon in order")
    P.check("(b) uses sorted()", 2, any(call_name(c) == "sorted" for c in calls_in(tree)))
    P.check("(c) total count (7) printed", 3, has_number(r.stdout, 7), "7 not found in output")

    lst = find_str_list(tree)
    alt = ["kiwi", "fig", "kiwi", "kiwi", "fig", "plum"]
    src2, hit = (substitute(src, lst, repr(alt)) if lst else (src, False))
    if hit:
        r2 = run_code(src2)
        P.check("hidden test: Fruits for a different list", 3,
                r2.vars.get("Fruits") == {"kiwi": 3, "fig": 2, "plum": 1},
                f"got {r2.vars.get('Fruits')!r}")
        P.check("hidden test: sorted names for a different list", 2,
                sorted_block_printed(r2.stdout, ["fig", "kiwi", "plum"]))
        P.check("hidden test: total (6) for a different list", 3, has_number(r2.stdout, 6),
                "total seems hard-coded or wrong")
    else:
        for d, p in (("Fruits", 3), ("sorted names", 2), ("total", 3)):
            P.check(f"hidden test: {d} for a different list", p, False,
                    "could not find the fruit list assignment - grade manually")
    return P


# --------------------------------------------------------------------------
# Problem 4 - try / except / else / finally
# --------------------------------------------------------------------------
def handler_names(h):
    if h.type is None:
        return {None}
    if isinstance(h.type, ast.Name):
        return {h.type.id}
    if isinstance(h.type, ast.Tuple):
        return {e.id for e in h.type.elts if isinstance(e, ast.Name)}
    return set()


def grade_p4(src):
    P = Problem(4, "File reading with try/except/else/finally")
    tree, err = try_parse(src)
    if err:
        return syntax_fail(P, err)
    trys = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    T = trys[0] if trys else None
    P.check("try statement present", 2, T is not None)
    if T:
        hv = [h for h in T.handlers if "ValueError" in handler_names(h)]
        hall = [h for h in T.handlers if handler_names(h) & {None, "Exception", "BaseException"}]
        P.check("except ValueError prints 'Wrong data format'", 2,
                any(prints_text(h.body, "Wrong data format") for h in hv))
        P.check("catch-all except prints 'unexpected error'", 2,
                any(prints_text(h.body, "unexpected error") for h in hall))
        P.check("'Good' printed inside the try construct (else / end of try)", 2,
                prints_text(T.orelse, "Good") or prints_text(T.body, "Good"))
        P.check("finally closes the file", 1,
                any(call_name(c) == "close" for s in T.finalbody for c in calls_in(s)))
        P.check("finally prints 'Done'", 1, prints_text(T.finalbody, "Done"))
    else:
        for d, p in (("except ValueError prints 'Wrong data format'", 2),
                     ("catch-all except prints 'unexpected error'", 2),
                     ("'Good' printed inside the try construct", 2),
                     ("finally closes the file", 1), ("finally prints 'Done'", 1)):
            P.check(d, p, False, "no try statement")

    def scenario(data):
        return run_code(src, files={"data.txt": data}, inf_path="data.txt",
                        track_files=True, float_boom=True)

    def done_last(lines):
        return lines.count("Done") == 1 and lines[-1] == "Done"

    # A: clean file
    r = scenario("95.2\n82.5\n91\n")
    report_run_error(P, r, "clean file")
    L = r.lines
    P.check("run: clean data -> 'Good' then 'Done' (no error messages)", 3,
            r.error is None and "Good" in L and "Wrong data format" not in L
            and "unexpected error" not in L and done_last(L),
            f"output was {L!r}")
    P.check("run: clean data -> file closed", 1, r.files_closed is True,
            "file not closed (or never opened)")
    # B: bad format
    r = scenario("95.2\n92#5\n82.5\n")
    report_run_error(P, r, "bad-format file")
    L = r.lines
    P.check("run: bad value -> 'Wrong data format', no 'Good', ends with 'Done'", 3,
            r.error is None and "Wrong data format" in L and "Good" not in L
            and "unexpected error" not in L and done_last(L),
            f"output was {L!r}")
    P.check("run: bad value -> file closed", 1, r.files_closed is True)
    # C: non-ValueError
    r = scenario("95.2\nBOOM\n82.5\n")
    report_run_error(P, r, "other-exception file")
    L = r.lines
    P.check("run: other exception -> 'unexpected error', no 'Good', ends with 'Done'", 2,
            r.error is None and "unexpected error" in L and "Wrong data format" not in L
            and "Good" not in L and done_last(L), f"output was {L!r}")
    return P


# --------------------------------------------------------------------------
# Problem 5 - zip / sort
# --------------------------------------------------------------------------
GIVEN_NAMES = {'Kevin', 'Ada', 'Jenny', 'Ade', 'Betty', 'Sam', 'Kenny', 'Beta',
               'Berry', 'Terry', 'Nathan', 'Jess'}


def comments_of(src):
    try:
        return [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [l for l in src.splitlines() if l.strip().startswith("#")]


def grade_p5(src):
    P = Problem(5, "zip + sorting")
    tree, err = try_parse(src)
    if err:
        return syntax_fail(P, err)
    r = run_code(src)
    report_run_error(P, r)
    v = r.vars
    names, ages, info = v.get("nameLst"), v.get("ageLst"), v.get("infoLst")

    P.check("(a) nameLst is the given 12 unique names", 2,
            isinstance(names, list) and len(names) == len(set(names)) and set(names) == GIVEN_NAMES,
            f"got {names!r}")
    aa = assigns(tree, "ageLst")
    P.check("(b) ageLst built with a list comprehension", 1,
            bool(aa) and isinstance(aa[0].value, ast.ListComp))
    P.check("(b) ageLst: same length as names, ints in [18, 21]", 1,
            isinstance(ages, list) and isinstance(names, list) and len(ages) == len(names)
            and all(type(a) is int and 18 <= a <= 21 for a in ages), f"got {ages!r}")

    # larger list -> check the range is inclusive of 18 and 21
    big = [f"n{i:02d}" for i in range(40)]
    src3, hit3 = substitute(src, "nameLst", repr(big))
    if hit3:
        r3 = run_code(src3)
        a3 = r3.vars.get("ageLst")
        ok = (isinstance(a3, list) and len(a3) == 40 and all(type(a) is int for a in a3)
              and set(a3) == {18, 19, 20, 21})
        P.check("(b) hidden test: 40 names -> ages cover 18..21 inclusive", 2, ok,
                "ages out of range or 18/21 never generated (randint(18,21) is inclusive)")
    else:
        P.check("(b) hidden test: 40 names -> ages cover 18..21 inclusive", 2, False,
                "could not find `nameLst = ...` - grade manually")

    P.check("(c) uses zip", 1, any(call_name(c) == "zip" for c in calls_in(tree)))
    try:
        pairs = list(zip(names, ages))
        ok = (isinstance(info, list) and len(info) == len(pairs)
              and sorted(map(tuple, info)) == sorted(pairs))
    except TypeError:
        pairs, ok = [], False
    P.check("(c) infoLst is a list of (name, age) tuples", 2, ok, f"got {info!r}")
    P.check("(c) infoLst printed (>= 3 lists of tuples in output)", 1,
            r.stdout.count("[(") >= 3, "expected 3 printed lists: zip, by name, by age")

    if ok:
        by_name = sorted(pairs, key=lambda p: p[0])
        by_age = sorted(pairs, key=lambda p: (-p[1], p[0]))
        out = norm(r.stdout)
        P.check("(d) list sorted by name printed", 3, norm(repr(by_name)) in out,
                f"expected {by_name!r}")
        P.check("(e) list sorted by age desc, then name, printed", 3, norm(repr(by_age)) in out,
                f"expected {by_age!r}")
    else:
        P.check("(d) list sorted by name printed", 3, False, "infoLst wrong; cannot verify")
        P.check("(e) list sorted by age desc, then name, printed", 3, False, "infoLst wrong; cannot verify")

    # (f) empty list + comments
    src2, hit2 = substitute(src, "nameLst", "[]")
    if hit2:
        r2 = run_code(src2)
        report_run_error(P, r2, "empty nameLst")
        P.check("(f) empty nameLst runs without error and prints []", 2,
                r2.error is None and "[]" in r2.stdout, r2.error or "")
    else:
        P.check("(f) empty nameLst runs without error and prints []", 2, False,
                "could not find `nameLst = ...` - grade manually")
    cm = comments_of(src)
    n_runs = sum(1 for c in cm if re.search(r"test\s*run", c, re.I))
    n_out = sum(1 for c in cm if "('" in c or '("' in c or "[]" in c)
    P.check("(f) comments label >= 3 test runs", 1, n_runs >= 3, f"found {n_runs}")
    P.check("(f) pasted output present in comments", 1, n_out >= 3,
            "expected >= 3 comment lines showing output")
    return P


GRADERS = {1: grade_p1, 2: grade_p2, 3: grade_p3, 4: grade_p4, 5: grade_p5}
TITLES = {1: "String -> words -> wordSize", 2: "Set operations", 3: "Dictionary of fruit counts",
          4: "try/except/else/finally", 5: "zip + sorting"}

# --------------------------------------------------------------------------
# Loading submissions
# --------------------------------------------------------------------------
MARKER = re.compile(
    r"^[ \t]*#+[ \t]*(?:problem|prob|question|ques|q|p)[ \t]*(?:no\.?|#)?[ \t]*([1-5])\b",
    re.I | re.M)


def split_single(text):
    ms = list(MARKER.finditer(text))
    out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out[int(m.group(1))] = out.get(int(m.group(1)), "") + "\n" + text[m.start():end]
    return out


def read(p):
    return Path(p).read_text(encoding="utf-8", errors="replace")


def load_submission(path):
    path = Path(path)
    probs = {}
    if path.is_dir():
        for f in sorted(path.glob("*.py")):
            m = re.search(r"(?<!\d)([1-5])(?!\d)", f.stem)
            if m:
                probs[int(m.group(1))] = read(f)
    else:
        probs = split_single(read(path))
        if not probs:  # no markers: maybe a single-problem file named ..._p3.py
            m = re.search(r"(?<!\d)([1-5])(?!\d)", path.stem)
            if m:
                probs[int(m.group(1))] = read(path)
    return probs


def grade_student(path):
    probs = load_submission(path)
    results = {}
    for n in range(1, 6):
        if n in probs:
            results[n] = GRADERS[n](probs[n])
        else:
            P = Problem(n, TITLES[n])
            P.check("Submission found", POINTS_PER_PROBLEM, False,
                    "no code found for this problem (missing file or '# Problem N' marker)")
            results[n] = P
    return results


def print_report(name, results, verbose):
    total = sum(p.earned for p in results.values())
    print("=" * 72)
    print(f"{name}: {total:.1f} / {POINTS_PER_PROBLEM * 5}")
    for n, p in results.items():
        print(f"\n  Problem {n} - {p.title}: {p.earned:g} / {p.total:g}")
        for msg in p.infos:
            print(f"      ! {msg}")
        for desc, pts, got, note in p.items:
            if got == pts and not verbose:
                continue
            tag = "PASS" if got == pts else ("PART" if got > 0 else "FAIL")
            print(f"    [{tag}] {desc}  ({got:g}/{pts:g})")
            if note:
                print(f"           {note}")
    print()


def main():
    global TIMEOUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--batch", action="store_true", help="each child of PATH is one student")
    ap.add_argument("--csv", help="write grades to this CSV file")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    a = ap.parse_args()
    TIMEOUT = a.timeout
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root = Path(a.path)
    if a.batch:
        students = sorted(p for p in root.iterdir()
                          if p.is_dir() or p.suffix == ".py")
    else:
        students = [root]
    rows = []
    for s in students:
        res = grade_student(s)
        name = s.stem if s.is_file() else s.name
        print_report(name, res, a.verbose)
        rows.append([name] + [res[n].earned for n in range(1, 6)] +
                    [round(sum(p.earned for p in res.values()), 2)])
    if a.batch:
        print("-" * 72)
        for r in rows:
            print(f"{r[0]:<30} " + " ".join(f"{x:>5g}" for x in r[1:]))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["student", "p1", "p2", "p3", "p4", "p5", "total"])
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
