#!/usr/bin/python3

import sympy, time, os, sys, subprocess, traceback, random, collections, psutil, concurrent.futures, pickle
import sympy.parsing.sympy_parser as parser
import sympy.parsing.latex as latex
import matplotlib.pyplot as plt
import sympy.plotting as plotter
from sympy.plotting.plot import Plot

deque = collections.deque


getattr(latex, "__builtins__", {})["print"] = lambda *void1, **void2: None


def logging(func):
    def call(self, *args, **kwargs):
        try:
            output = func(self, *args, **kwargs)
        except:
            print(traceback.format_exc(), end="")
            raise
        return output
    return call


BF_PREC = 256
BF_ALPHA = "0123456789abcdefghijklmnopqrstuvwxyz"

def TryWrapper(func):
    def __call__(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print(traceback.format_exc(), end="")
    return __call__

# Brainfuck parser below borrowed from:

#!/usr/bin/python
#
# Brainfuck Interpreter
# Copyright 2011 Sebastian Kaspari

def evaluate(code):
    out = deque()
    code = cleanup(list(code))
    bracemap = buildbracemap(code)
    cells, codeptr, cellptr = [0], 0, 0
    while codeptr < len(code):
        command = code[codeptr]
        if command == ">":
            cellptr += 1
            while cellptr >= len(cells):
                cells.append(0)
        if command == "<":
            cellptr = 0 if cellptr <= 0 else cellptr - 1
        if command == "+":
            cells[cellptr] = cells[cellptr] + 1 if cells[cellptr] < 255 else 0
        if command == "-":
            cells[cellptr] = cells[cellptr] - 1 if cells[cellptr] > 0 else 255
        if command == "[" and cells[cellptr] == 0:
            codeptr = bracemap[codeptr]
        if command == "]" and cells[cellptr] != 0:
            codeptr = bracemap[codeptr]
        if command == ".":
            out.append(chr(cells[cellptr]))
        if command == ",":
            cells[cellptr] = ord("")
        codeptr += 1
    return "".join(out)

cleanup = lambda code: "".join(filter(lambda x: x in ".,[]<>+-", code))

def buildbracemap(code):
    temp_bracestack, bracemap = deque(), {}
    for position, command in enumerate(code):
        if command == "[": temp_bracestack.append(position)
        if command == "]":
            start = temp_bracestack.pop()
            bracemap[start] = position
            bracemap[position] = start
    return bracemap


BF_ALIAS = ("bf", "brainfuck")
alphanumeric = frozenset("abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper() + "0123456789" + "_")


# Enclose string in brackets if possible
def bf_parse(s):
    if "'" in s or '"' in s:
        return s
    s = s.replace("\n", " ").replace("\t", " ")
    for a in BF_ALIAS:
        try:
            i = s.index(a) + len(a)
        except ValueError:
            continue
        else:
            while s[i] in " \n":
                i += 1
            if s[i] == "(":
                i += 1
                v, s = s[:i], s[i:]
                e = s.index(")")
                s = v + '"' + s[:e] + '"' + s[e:]
    if " " in s:
        i = len(s)
        while i:
            try:
                i = s[:i].rindex(" ")
            except ValueError:
                break
            else:
                if i > 0 and i < len(s) - 1:
                    if s[i - 1].isalnum() and s[i + 1].isalnum():
                        s = s[:i] + "(" + s[i + 1:] + ")"
    return s

_bf = lambda s: evaluate(s)


# Randomizer
class Random(sympy.Basic):

    def __init__(self, a=None, b=None):
        if a is None:
            self.a = 0
            self.b = 1
            self.isint = False
        elif b is None:
            self.a = 0
            self.b = a
            self.isint = True
        else:
            sgn = sympy.sign(b - a)
            self.a = sgn * a + (1 - sgn) * b
            self.b = sgn * b + (1 - sgn) * a + 1
            self.isint = True
    
    def evalf(self, prec):
        randfloat = sympy.Float(random.random(), dps=prec) / 2.7 ** (prec / 7 - random.random())
        temp = (sympy.Float(random.random(), dps=prec) / (randfloat + time.time() % 1)) % 1
        temp *= self.b - self.a
        temp += self.a
        if self.isint:
            temp = sympy.Integer(temp)
        return temp

    gcd = lambda self, other, *gens, **args: sympy.gcd(self.evalf(BF_PREC), other, *gens, **args)
    lcm = lambda self, other, *gens, **args: sympy.lcm(self.evalf(BF_PREC), other, *gens, **args)
    is_Rational = lambda self: True
    expand = lambda self, **void: self.evalf(BF_PREC)
    nsimplify = lambda self, **void: self.evalf(BF_PREC)
    as_coeff_Add = lambda self, *void: (0, self.evalf(BF_PREC))
    as_coeff_Mul = lambda self, *void: (0, self.evalf(BF_PREC))
    _eval_power = lambda *void: None
    _eval_evalf = evalf
    __abs__ = lambda self: abs(self.evalf(BF_PREC))
    __neg__ = lambda self: -self.evalf(BF_PREC)
    __repr__ = lambda self, *void: str(self.evalf(BF_PREC))
    __str__ = __repr__


# Sympy plotting functions
def plotArgs(args):
    if not args:
        return
    if type(args[0]) in (tuple, list):
        args = list(args)
        args[0] = sympy.interpolating_spline(
            3,
            sympy.Symbol("x"),
            list(range(len(args[0]))),
            args[0],
        )
    return args

colours = [c + "-H" for c in "bgrymc"]

def plot(*args, **kwargs):
    kwargs.pop("show", None)
    return plotter.plot(*plotArgs(args), show=False, **kwargs)

def plot_parametric(*args, **kwargs):
    kwargs.pop("show", None)
    return plotter.plot_parametric(*plotArgs(args), show=False, **kwargs)

def plot_implicit(*args, **kwargs):
    kwargs.pop("show", None)
    return plotter.plot_implicit(*plotArgs(args), show=False, **kwargs)

def plot_array(*args, **kwargs):
    kwargs.pop("show", None)
    for arr, c in zip(args, colours):
        plt.plot(list(range(len(arr))), arr, c, **kwargs)
    return plt

def plot3d(*args, **kwargs):
    kwargs.pop("show", None)
    return plotter.plot3d(*plotArgs(args), show=False, **kwargs)

def plot3d_parametric_line(*args, **kwargs):
    kwargs.pop("show", None)
    return plotter.plot3d_parametric_line(*plotArgs(args), show=False, **kwargs)

def plot3d_parametric_surface(*args, **kwargs):
    kwargs.pop("show", None)
    return plotter.plot3d_parametric_surface(*plotArgs(args), show=False, **kwargs)

# Multiple variable limit
def lim(f, **kwargs):
    for i in kwargs:
        f = sympy.limit(f, i, kwargs[i])
    return f

# May integrate a spline
def integrate(*args, **kwargs):
    try:
        return sympy.integrate(*args, **kwargs)
    except ValueError:
        return sympy.integrate(*plotArgs(args), sympy.Symbol("x"))

if os.name == "nt":
    def _factorint(n, **kwargs):
        try:
            s = str(n)
            if "." in s:
                raise TypeError
            if abs(int(s)) < 1 << 64:
                raise ValueError
        except (TypeError, ValueError):
            return sympy.factorint(n, **kwargs)
        data = subprocess.check_output("misc/ecm.exe " + s).decode("utf-8").replace(" ", "")
        if "<li>" not in data:
            if not data:
                raise RuntimeError("no output found.")
            raise RuntimeError(data)
        data = data[data.index("<li>") + 4:]
        data = data[:data.index("</li>")]
        while "(" in data:
            i = data.index("(")
            try:
                j = data.index(")")
            except ValueError:
                break
            data = data[:i] + data[j + 1:]
        if data.endswith("isprime"):
            data = data[:-7]
        else:
            data = data[data.rindex("=") + 1:]
        factors = {}
        for factor in data.split("*"):
            if "^" in factor:
                k, v = factor.split("^")
            else:
                k, v = factor, 1
            factors[int(k)] = int(v)
        return factors
else:
    _factorint = sympy.factorint

def factorize(*args, **kwargs):
    temp = _factorint(*args, **kwargs)
    output = []
    for k in sorted(temp):
        output.extend([k] * temp[k])
    return output

def rounder(x):
    try:
        if type(x) is int or x == int(x):
            return int(x)
        f = int(round(x))
        if x == f:
            return f
    except:
        pass
    return x


# Allowed functions for ~math
_globals = dict(sympy.__dict__)
pop = (
    "init_printing",
    "init_session",
    "seterr",
    "factorint",
)
for i in pop:
    _globals.pop(i)
plots = (
    "plot",
    "plot_parametric",
    "plot_implicit",
    "plot_array",
    "plot3d",
    "plot3d_parametric_line",
    "plot3d_parametric_surface",
)
for i in plots:
    _globals[i] = globals()[i]
_globals.update({
    "bf": _bf,
    "brainfuck": _bf,
    "random": Random,
    "rand": Random,
    "dice": Random,
    "plt": plot,
    "lim": lim,
    "factorint": _factorint,
    "factors": _factorint,
    "factorize": factorize,
    "factor": factorize,
    "abs": sympy.Abs,
    "solve": sympy.solve,
    "simplify": sympy.simplify,
    "intg": integrate,
    "integral": integrate,
    "integrate": integrate,
    "differentiate": sympy.diff,
    "derivative": sympy.diff,
    "derive": sympy.diff,
    "phase": sympy.arg,
    "ceil": sympy.ceiling,
    "min": sympy.Min,
    "max": sympy.Max,
    "fac": sympy.factorial,
    "phi": sympy.GoldenRatio,
    "tau": sympy.pi * 2,
    "deg": sympy.pi / 180,
    "degrees": sympy.pi / 180,
    "degree": sympy.pi / 180,
    "rad": sympy.Integer(1),
    "radians": sympy.Integer(1),
    "radian": sympy.Integer(1),
    "inf": sympy.oo,
    "nan": sympy.nan,
    "i": sympy.I,
    "j": sympy.I,
    "e": sympy.E,
    "k": sympy.Integer(1 << 10),
    "M": sympy.Integer(1 << 20),
    "G": sympy.Integer(1 << 30),
    "T": sympy.Integer(1 << 40),
    "P": sympy.Integer(1 << 50),
    "E": sympy.Integer(1 << 60),
    "Z": sympy.Integer(1 << 70),
    "Y": sympy.Integer(1 << 80),
    "c": sympy.Integer(299792458),
})
pop = (
    "input",
)
for i in pop:
    _globals["__builtins__"].pop(i)

sym_tr = parser.standard_transformations
sym_tr += (
    parser.convert_xor,
    parser.implicit_multiplication_application,
    parser.rationalize,
)

# Mathematical symbols
translators = {
    "√": "sqrt 0+",
    "°": " deg",
    "÷": "/",
    "–": "-",
    "−": "-",
    "×": "*",
    "·": "*",
    "᛫": "*",
    "•": "*",
    "‧": "*",
    "∙": "*",
    "⋅": "*",
    "⸱": "*",
    "・": "*",
    "ꞏ": "*",
    "･": "*",
    "Σ": "Sum 0+",
    "∑": "Sum 0+",
    "∫": "intg 0+",
    "Γ": "gamma 0+",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "ς": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "∞": "oo",
    "ℯ": "e",
}

replacers = {
    "INF": "oo",
    "NAN": "nan",
    "NaN": "nan",
    "TRUE": "True",
    "FALSE": "False",
    "coo": "zoo",
    "cinf": "zoo",
    "Dₓ": "diff 0+",
}

ftrans = "".maketrans(translators)


# Use more conventional names for non-finite outputs

def convAns(f):
    return str(f).replace("zoo", "nan").replace("oo", "inf")

def prettyAns(f):
    return sympy.pretty(
        f,
        use_unicode=True,
        num_columns=2147483647,
        mat_symbol_style="bold",
    ).replace("zoo", "ℂ∞").replace("nan", "NaN").replace("⋅", "∙")


# Main math equation solver
@logging
def evalSym(f, prec=64, r=False, variables=None):
    if variables is None:
        env = _globals
    else:
        env = dict(_globals)
        envd = eval(variables, {}, {})
        envs = {k: sympy.sympify(v) for k, v in envd.items()}
        env.update(envs)
    if f.lower() == "help":
        lines = deque()
        line = deque()
        lines.append("Available functions/variables:")
        items = set(_globals)
        while items:
            temp = items.pop()
            if not temp or temp.startswith("__"):
                continue
            line.append(temp)
            if len(line) >= 8:
                lines.append("\t".join(line))
                line.clear()
        return ["\n".join(lines)]
    global BF_PREC
    BF_PREC = sympy.ceiling(int(prec) * 1.25)
    r = int(r)
    prec = int(prec)
    f, y = f.translate(ftrans), f
    for i in replacers:
        f = f.replace(i, replacers[i])
    # Attempt to parse as SymPy expression, then as LaTeX if possible
    try:
        if "\\" in y:
            raise SyntaxError
        f = parser.parse_expr(
            bf_parse(f),
            local_dict=None,
            global_dict=env,
            transformations=sym_tr,
            evaluate=True,
        )
    except SyntaxError:
        try:
            f = latex.parse_latex(y)
        except:
            f = latex.parse_latex(f)
    # Solve any sums and round off floats when possible
    for i in sympy.preorder_traversal(f):
        if issubclass(type(i), sympy.Number):
            try:
                f = f.subs(i, rounder(i))
            except:
                pass
        elif hasattr(i, "doit"):
            try:
                f = f.subs(i, i.doit())
            except:
                pass
    # If the requested expression evaluates to a plot, return it
    if isinstance(f, Plot) or f == plt or type(f) is str:
        return (f,)
    try:
        f = sympy.simplify(f)
    except:
        pass
    # Solve any sums and round off floats when possible
    for i in sympy.preorder_traversal(f):
        try:
            f = f.subs(i, rounder(i))
        except:
            pass
        if hasattr(i, "doit"):
            try:
                f = f.subs(i, i.doit())
            except:
                pass
    # Select list of answers to return based on the desired float precision level
    if prec:
        if type(f) in (tuple, list, dict):
            return [f]
        try:
            y = f.evalf(prec, chop=True)
        except:
            y = [f]
        try:
            e = rounder(y)
        except TypeError:
            e = y
            for i in sympy.preorder_traversal(e):
                if isinstance(i, sympy.Float):
                    e = e.subs(i, rounder(i))
        if r:
            p = prettyAns(f)
            if p == convAns(e):
                p = ""
            return [f, p]
        p = prettyAns(f)
        if p == convAns(e):
            p = ""
        if "." in str(e):
            e = str(e).rstrip("0")
        return [e, p]
    else:
        p = prettyAns(f)
        if p == convAns(f):
            p = ""
        return [f, p]


@logging
def procResp(resp):
    # Return file path if necessary
    if isinstance(resp[0], Plot):
        ts = time.time_ns() // 1000
        name = f"{ts}.png"
        fn = "cache/" + name
        try:
            resp[0].save(fn)
        except FileNotFoundError:
            fn = name
            resp[0].save(fn)
        plt.clf()
        s = "{'file':'" + fn + "'}\n"
    elif resp[0] == plt:
        ts = time.time_ns() // 1000
        name = f"{ts}.png"
        fn = "cache/" + name
        try:
            plt.savefig(fn)
        except FileNotFoundError:
            fn = name
            plt.savefig(fn)
        plt.clf()
        s = "{'file':'" + fn + "'}\n"
    elif type(resp) is tuple:
        s = list(resp)
    else:
        s = [convAns(i) for i in resp]
    return s


def evaluate(ts, args):
    try:
        resp = evalSym(*eval(eval(args)))
        out = procResp(resp)
        if len(out) > 8388608:
            raise OverflowError("Output data too large.")
    except Exception as ex:
        sys.stdout.write(f"~PROC_RESP[{ts}].set_exception(pickle.loads({repr(pickle.dumps(ex))}))\n")
    else:
        sys.stdout.write(f"~PROC_RESP[{ts}].set_result({repr(out)})\n")
    sys.stdout.flush()


def ensure_parent(proc, parent):
    while True:
        if not parent.is_running():
            proc.kill()
        time.sleep(12)

if __name__ == "__main__":
    pid = os.getpid()
    ppid = os.getppid()
    proc = psutil.Process(pid)
    parent = psutil.Process(ppid)
    exc = concurrent.futures.ThreadPoolExecutor(max_workers=9)
    exc.submit(ensure_parent)
    while True:
        argv = sys.stdin.readline().rstrip()
        if argv:
            if argv[0] == "~":
                ts, args = argv[1:].split("~", 1)
                exc.submit(evaluate, ts, args)