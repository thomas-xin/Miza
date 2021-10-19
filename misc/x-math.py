#!/usr/bin/python3

import sympy, mpmath, math, time, os, sys, subprocess, psutil, traceback, random
import collections, itertools, pickle, ast, re
import sympy.stats
import numpy as np
import sympy.parsing.sympy_parser as parser
import sympy.parsing.latex as latex
import matplotlib.pyplot as plt
import sympy.plotting as plotter
from sympy.plotting.plot import Plot

if not hasattr(time, "time_ns"):
    time.time_ns = lambda: int(time.time() * 1e9)

deque = collections.deque

getattr(latex, "__builtins__", {})["print"] = lambda *void1, **void2: None


def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)

literal_eval = lambda s: ast.literal_eval(as_str(s).lstrip())


BF_PREC = 256
BF_ALPHA = "0123456789abcdefghijklmnopqrstuvwxyz"

mp = mpmath.mp
mp.dps = BF_PREC
mpf = mpmath.mpf
mpf.__floordiv__ = lambda x, y: int(x / y)
mpf.__rfloordiv__ = lambda y, x: int(x / y)
mpf.__lshift__ = lambda x, y: x * (1 << y if type(y) is int else 2 ** y)
mpf.__rshift__ = lambda x, y: x // (1 << y if type(y) is int else 2 ** y)
mpf.__rlshift__ = lambda y, x: x * (1 << y if type(y) is int else 2 ** y)
mpf.__rrshift__ = lambda y, x: x * (1 << y if type(y) is int else 2 ** y)
mpc = mpmath.mpc
Mat = mat = matrix = mpmath.matrix

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

def bf_evaluate(code):
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

_bf = lambda s: bf_evaluate(s)


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


def astype(obj, t, *args, **kwargs):
    try:
        if not isinstance(obj, t):
            if callable(t):
                return t(obj, *args, **kwargs)
            return t
    except TypeError:
        if callable(t):
            return t(obj, *args, **kwargs)
        return t
    return obj


RI = 1 << 256

def iand(a, b):
    if a == b:
        return a
    if hasattr(a, "p") and getattr(a, "q", 1) == 1 and hasattr(b, "p") and getattr(b, "q", 1) == 1:
        return sympy.Integer(a.p & b.p)
    x = round(a * RI)
    y = round(b * RI)
    return sympy.Integer(x & y) / RI

def ior(a, b):
    if a == b:
        return a
    if hasattr(a, "p") and getattr(a, "q", 1) == 1 and hasattr(b, "p") and getattr(b, "q", 1) == 1:
        return sympy.Integer(a.p | b.p)
    x = round(a * RI)
    y = round(b * RI)
    return sympy.Integer(x | y) / RI

def ixor(a, b):
    if a == b:
        return sympy.Integer(0)
    if hasattr(a, "p") and getattr(a, "q", 1) == 1 and hasattr(b, "p") and getattr(b, "q", 1) == 1:
        return sympy.Integer(a.p ^ b.p)
    x = round(a * RI)
    y = round(b * RI)
    return sympy.Integer(x ^ y) / RI

_pow = sympy.Float.__pow__

def pow(a, b):
    if hasattr(a, "p") and getattr(a, "q", 1) == 1 and hasattr(b, "p") and getattr(b, "q", 1) == 1:
        exponent = mpmath.log(a.p, 2) * b.p
        if exponent > 256:
            return sympy.Float(2 ** exponent)
        return a.p ** b.p
    a = astype(a, sympy.Float)
    return _pow(a, b)

sympy.Basic.__and__ = lambda self, other: iand(self, other)
sympy.Basic.__or__ = lambda self, other: ior(self, other)
sympy.Basic.__xor__ = lambda self, other: ixor(self, other)
sympy.Basic.__pow__ = lambda self, other: pow(self, other)
sympy.Basic.__rpow__ = lambda self, other: pow(other, self)
sympy.core.numbers.Infinity.__str__ = lambda self: "inf"
sympy.core.numbers.NegativeInfinity.__str__ = lambda self: "-inf"
sympy.core.numbers.ComplexInfinity.__str__ = lambda self: "ℂ∞"
sympy.erfcinv = lambda z: sympy.erfinv(1 - z)
sympy.erfcinv.inverse = sympy.erfc
sympy.erfc.inverse = sympy.erfcinv


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

def array(*args):
    if len(args) == 1:
        arr = args[0]
        if type(arr) is str:
            arr = re.split("[^0-9\\-+e./]+", arr)
            arr = list(map(sympy.Rational, arr))
        return np.asanyarray(arr, dtype=object)
    return np.array(args, dtype=object)

def _predict_next(seq):
    if len(seq) < 2:
        return
    if np.min(seq) == np.max(seq):
        return round_min(seq[0])
    if len(seq) < 3:
        return
    if len(seq) > 4 and all(seq[2:] - seq[1:-1] == seq[:-2]):
        return round_min(seq[-1] + seq[-2])
    a = _predict_next(seq[1:] - seq[:-1])
    if a is not None:
        return round_min(seq[-1] + a)
    if len(seq) < 4 or 0 in seq[:-1]:
        return
    b = _predict_next(seq[1:] / seq[:-1])
    if b is not None:
        return round_min(seq[-1] * b)

def predict_next(seq, limit=8):
    seq = np.array(deque(astype(x, mpf) for x in seq), dtype=object)
    for i in range(min(5, limit), 1 + max(5, min(len(seq), limit))):
        temp = _predict_next(seq[-i:])
        if temp is not None:
            return temp

# Multiple variable limit
def lim(f, **kwargs):
    if hasattr(f, "subs"):
        g = f.subs(kwargs)
        try:
            if not math.isnan(g):
                return g
        except TypeError:
            return g
    for i in kwargs:
        g = sympy.limit(f, i, kwargs[i], "+")
        h = sympy.limit(f, i, kwargs[i], "-")
        if g != h:
            try:
                if not math.isfinite(g) and not math.isfinite(h) and g == -h:
                    f = sympy.zoo
                    continue
            except TypeError:
                pass
            f = (g + h) / 2
        else:
            f = g
    return f

# May integrate a spline
def integrate(*args, **kwargs):
    try:
        return sympy.integrate(*args, **kwargs)
    except ValueError:
        return sympy.integrate(*plotArgs(args), sympy.Symbol("x"))

fac = sympy.factorial
ncr = lambda n, k: 0 if k > n else fac(n) / fac(k) / fac(n - k)
npr = lambda n, k: 0 if k > n else fac(n) / fac(n - k)
normcdf = lambda x: 0.5 * sympy.erfc(-x / sympy.sqrt(2))
norminv = lambda x: -sympy.sqrt(2) * sympy.erfcinv(2 * x)

def lcm(*nums):
    while len(nums) > 1:
        if len(nums) & 1:
            x = nums[-1]
        else:
            x = None
        nums = [sympy.lcm(*t) for t in zip(nums[::2], nums[1::2])]
        if x is not None:
            nums.append(x)
    return nums[0]

def gcd(*nums):
    while len(nums) > 1:
        if len(nums) & 1:
            x = nums[-1]
        else:
            x = None
        nums = [sympy.gcd(*t) for t in zip(nums[::2], nums[1::2])]
        if x is not None:
            nums.append(x)
    return nums[0]

if os.name == "nt":
    if not os.path.exists("misc/ecm.exe"):
        import requests
        with requests.get("https://cdn.discordapp.com/attachments/731709481863479436/899561574145081364/ecm.exe") as resp:
            b = resp.content
        with open("misc/ecm.exe", "wb") as f:
            f.write(b)
else:
    if not os.path.exists("misc/ecm"):
        import requests
        with requests.get("https://cdn.discordapp.com/attachments/731709481863479436/899561549881032734/ecm") as resp:
            b = resp.content
        with open("misc/ecm", "wb") as f:
            f.write(b)
        subprocess.run(("chmod", "777", "misc/ecm"))
_fcache = {}
def _factorint(n, **kwargs):
    try:
        s = str(n)
        if "." in s:
            raise TypeError
        if abs(int(s)) < 1 << 64:
            raise ValueError
    except (TypeError, ValueError):
        return sympy.factorint(n, **kwargs)
    try:
        return _fcache[s]
    except KeyError:
        pass
    args = ["misc/ecm", s]
    proc = subprocess.run(args, stdout=subprocess.PIPE)
    data = proc.stdout.decode("utf-8", "replace").replace(" ", "")
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
    _fcache[s] = factors
    return factors

def factorize(*args, **kwargs):
    temp = _factorint(*args, **kwargs)
    return list(itertools.chain(*((k,) * v for k, v in sorted(temp.items()))))

def sort(*args):
    if len(args) != 1:
        return sorted(args)
    return sorted(args[0])

def round_random(x):
    try:
        y = int(x)
    except (ValueError, TypeError):
        return x
    if y == x:
        return y
    x -= y
    if random.random() <= x:
        y += 1
    return y

def rounder(x):
    try:
        if type(x) is int:
            return x
        y = int(x)
        if x == y:
            return y
    except:
        pass
    return x


# Allowed functions for ~math
_globals = dict(sympy.stats.__dict__)
_globals.update(sympy.__dict__)
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
    "round_random": round_random,
    "plt": plot,
    "array": array,
    "predict": predict_next,
    "predict_next": predict_next,
    "rollaxis": np.rollaxis,
    "swapaxes": np.swapaxes,
    "transpose": np.transpose,
    "expand_dims": np.expand_dims,
    "squeeze": np.squeeze,
    "concatenate": np.concatenate,
    "concat": np.concatenate,
    "stack": np.stack,
    "block": np.block,
    "split": np.split,
    "tile": np.tile,
    "repeat": np.repeat,
    "delete": np.delete,
    "insert": np.insert,
    "append": np.append,
    "resize": np.resize,
    "unique": np.unique,
    "flip": np.flip,
    "reshape": np.reshape,
    "roll": np.roll,
    "rot90": np.rot90,
    "sum": np.sum,
    "max": np.nanmax,
    "min": np.nanmin,
    "argmax": np.argmax,
    "argmin": np.argmin,
    "ptp": np.ptp,
    "mean": np.mean,
    "median": np.median,
    "std": lambda a: sympy.sqrt(np.var(a)),
    "var": np.var,
    "corrcoef": np.corrcoef,
    "correlate": np.correlate,
    "cov": np.cov,
    "covariance": np.cov,
    "dot": np.dot,
    "dotproduct": np.dot,
    "inner": np.inner,
    "outer": np.outer,
    "matmul": np.matmul,
    "inv": np.linalg.inv,
    "matinv": np.linalg.inv,
    "pinv": np.linalg.pinv,
    "matpwr": np.linalg.matrix_power,
    "matrix_power": np.linalg.matrix_power,
    "einsum": np.einsum,
    "eig": np.linalg.eig,
    "eigvals": np.linalg.eigvals,
    "svd": np.linalg.svd,
    "norm": np.linalg.norm,
    "cond": np.linalg.cond,
    "det": np.linalg.det,
    "trace": np.trace,
    "histogram": np.histogram,
    "average": np.average,
    "percentile": np.percentile,
    "quantile": np.quantile,
    "digitize": np.digitize,
    "digitise": np.digitize,
    "fft": np.fft.fft,
    "rfft": np.fft.rfft,
    "ifft": np.fft.ifft,
    "irfft": np.fft.irfft,
    "fftfreq": np.fft.fftfreq,
    "lim": lim,
    "sub": lim,
    "subs": lim,
    "factorint": _factorint,
    "factors": _factorint,
    "factorise": factorize,
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
    "sort": sort,
    "sorted": sort,
    "fac": fac,
    "ncr": ncr,
    "nCr": ncr,
    "npr": npr,
    "nPr": npr,
    "normcdf": normcdf,
    "norminv": norminv,
    "lcm": lcm,
    "gcd": gcd,
    "hcf": gcd,
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
supported = set((
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "memoryview",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "tuple",
    "type",
    "zip",
))
builtins = getattr(__builtins__, "__dict__", __builtins__)
for k, v in list(builtins.items()):
    if k in supported:
        _globals[k] = v

sym_tr = parser.standard_transformations
sym_tr += (
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
    "^": "**",
    "\x7f": "^",
}

replacers = {
    "<<": "*2**",
    ">>": "//2**",
    "INF": "oo",
    "NAN": "nan",
    "NaN": "nan",
    "TRUE": "True",
    "FALSE": "False",
    "coo": "zoo",
    "cinf": "zoo",
    "ℂ∞": "zoo",
    "Dₓ": "diff 0+",
    "^^": "\x7f",
}

ftrans = "".maketrans(translators)


# Use more conventional names for non-finite outputs

def prettyAns(f):
    return sympy.pretty(
        f,
        use_unicode=True,
        num_columns=2147483647,
        mat_symbol_style="bold",
    ).replace("zoo", "ℂ∞").replace("nan", "NaN").replace("⋅", "∙")


# Main math equation solver
def evalSym(f, prec=64, r=False, variables=None):
    if variables is None:
        env = _globals
    else:
        env = dict(_globals)
        envs = {k: sympy.sympify(v) for k, v in variables.items()}
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
    y = f
    for k, v in replacers.items():
        f = f.replace(k, v)
    f = f.translate(ftrans)
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
            f = None
            try:
                f = latex.parse_latex(f)
            except:
                pass
        if not f:
            raise
    # Solve any sums and round off floats when possible
    for i in sympy.preorder_traversal(f):
        if isinstance(i, (sympy.Number, float, np.floating)):
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
    if isinstance(f, Plot) or f is plt or type(f) is str:
        return (f,)
    try:
        f = sympy.simplify(f)
    except:
        pass
    # Solve any sums and round off floats when possible
    for i in sympy.preorder_traversal(f):
        if isinstance(i, (sympy.Number, float, np.floating)):
            try:
                f = f.subs(i, rounder(i))
            except:
                pass
        elif hasattr(i, "doit"):
            try:
                f = f.subs(i, i.doit())
            except:
                pass
    # Select list of answers to return based on the desired float precision level
    if type(f) in (str, bool, tuple, list, dict, np.ndarray):
        return [f]
    if prec:
        try:
            if isinstance(f, sympy.Integer):
                return [f]
            y = f.evalf(prec, chop=True)
        except:
            y = f
        try:
            e = rounder(y)
        except TypeError:
            e = y
            for i in sympy.preorder_traversal(e):
                if isinstance(i, sympy.Float):
                    e = e.subs(i, rounder(i))
        if r:
            p = prettyAns(f)
            if p == repr(e):
                p = ""
            return [f, p]
        p = prettyAns(f)
        f = repr(e)
        if p == f:
            p = ""
        if "." in f:
            e = f.rstrip("0")
        return [e, p]
    else:
        p = prettyAns(f)
        if p == repr(f):
            p = ""
        return [f, p]


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
        s = dict(file=fn)
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
        s = dict(file=fn)
    elif type(resp) is tuple:
        s = list(resp)
    else:
        s = [repr(i) if type(i) is not str else i for i in resp]
    return s


def evaluate(ts, args):
    try:
        resp = evalSym(*literal_eval(literal_eval(args)))
        out = procResp(resp)
        if len(out) > 8388608:
            raise OverflowError("Output data too large.")
        sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_result({repr(out)})\n".encode("utf-8"))
    except Exception as ex:
        sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_exception(pickle.loads({repr(pickle.dumps(ex))}))\n".encode("utf-8"))
        sys.stdout.buffer.write(f"~print({args},{repr(traceback.format_exc())},sep='\\n',end='')\n".encode("utf-8"))
    sys.stdout.flush()


if __name__ == "__main__":
    def ensure_parent():
        parent = psutil.Process(os.getppid())
        while True:
            if not parent.is_running():
                p = psutil.Process()
                for c in p.children(True):
                    c.terminate()
                    try:
                        c.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        c.kill()
                p.terminate()
                p.wait()
            time.sleep(12)
    import threading
    threading.Thread(target=ensure_parent, daemon=True).start()
    while True:
        argv = sys.stdin.readline().rstrip()
        if argv:
            if argv[0] == "~":
                ts, args = argv[1:].split("~", 1)
                evaluate(ts, args)