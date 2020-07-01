#!/usr/bin/python3

import sympy, time, sys, traceback, random, numpy
import sympy.parsing.sympy_parser as parser
import sympy.parsing.latex as latex
import matplotlib.pyplot as plt
import sympy.plotting as plotter
from sympy.plotting.plot import Plot

plt.rcParams["figure.figsize"] = (6.4, 4.8)

getattr(latex, "__builtins__", {})["print"] = lambda *void1, **void2: None


def logging(func):
    def call(self, *args, file="log.txt", **kwargs):
        try:
            output = func(self, *args, **kwargs)
        except:
            f = open(file, "ab")
            f.write(traceback.format_exc().encode("utf-8"))
            f.close()
            raise
        return output
    return call


BF_PREC = 256
BF_ALPHA = "0123456789abcdefghijklmnopqrstuvwxyz"

def tryWrapper(func):
    def __call__(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print(traceback.format_exc())
    return __call__


class dice(sympy.Basic):

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


special_colours = {
    "message": (0, 0, 1),
    "typing": (0, 1, 0),
    "command": (0, 1, 1),
    "reaction": (1, 1, 0),
    "misc": (1, 0, 0),
}

def plt_special(d, user, **void):
    plt.rcParams["figure.figsize"] = (16, 5)
    temp = None
    for k, v in reversed(d.items()):
        if temp is None:
            temp = numpy.array(v)
        else:
            temp += numpy.array(v)
        plt.bar(list(range(-len(v) + 1, 1)), v, color=special_colours.get(k, "k"), label=k)
    plt.bar(list(range(-len(temp) + 1, 1)), (temp <= 0) * max(temp) / 512, color=(0, 0, 0))
    plt.title("Recent Discord Activity for " + user)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Action Count")
    plt.legend(loc="upper left")
    return plt

def plotArgs(args):
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
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot(*plotArgs(args), show=False, **kwargs)

def plot_parametric(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot_parametric(*plotArgs(args), show=False, **kwargs)

def plot_implicit(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot_implicit(*plotArgs(args), show=False, **kwargs)

def plot_array(*args, **kwargs):
    plt.rcParams["figure.figsize"] = (6.4, 4.8)
    for arr, c in zip(args, colours):
        plt.plot(list(range(len(arr))), arr, c, **kwargs)
    return plt

def plot3d(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot3d(*plotArgs(args), show=False, **kwargs)

def plot3d_parametric_line(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot3d_parametric_line(*plotArgs(args), show=False, **kwargs)

def plot3d_parametric_surface(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot3d_parametric_surface(*plotArgs(args), show=False, **kwargs)

def lim(f, **kwargs):
    for i in kwargs:
        f = sympy.limit(f, i, kwargs[i])
    return f

def integrate(*args, **kwargs):
    try:
        return sympy.integrate(*args, **kwargs)
    except ValueError:
        return sympy.integrate(*plotArgs(args), sympy.Symbol("x"))

def factorize(*args, **kwargs):
    temp = sympy.factorint(*args, **kwargs)
    output = []
    for k in temp:
        for _ in range(temp[k]):
            output.append(k)
    return output

def rounder(x):
    try:
        if x == int(x):
            return int(x)
        f = int(round(x))
        if x == f:
            return f
    except:
        pass
    return x

locked = True

def _eval(func, glob=None, loc=None, key=None, **void):
    if glob is None:
        glob = globals()
    if locked and key != globals()["key"]:
        raise PermissionError("Nice try, but this is locked behind a randomized SHA256 key :3")
    try:
        return eval(func, glob, loc)
    except SyntaxError:
        pass
    return exec(func, glob, loc)


_globals = dict(sympy.__dict__)
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
    "eval": _eval,
    "random": dice,
    "rand": dice,
    "dice": dice,
    "plt": plot,
    "lim": lim,
    "factors": sympy.factorint,
    "factorize": factorize,
    "factor": factorize,
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
    "rad": 1,
    "radians": 1,
    "radian": 1,
    "inf": sympy.oo,
    "nan": sympy.nan,
    "i": sympy.I,
    "j": sympy.I,
    "e": sympy.E,
    "C": 299792458,
    "G": 6.6743015e-11,
})
pop = (
    "init_printing",
    "init_session",
    "seterr",
)
for i in pop:
    _globals.pop(i)
pop = (
    "open",
    "input",
)
for i in pop:
    _globals["__builtins__"].pop

sym_tr = parser.standard_transformations
sym_tr += (
    parser.convert_xor,
    parser.implicit_multiplication_application,
    parser.rationalize,
)

translators = {
    "√": "sqrt ",
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
    "Σ": "Sum ",
    "∑": "Sum ",
    "∫": "intg ",
    "Γ": "gamma ",
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
}

ftrans = "".maketrans(translators)


def convAns(f):
    return str(f).replace("zoo", "nan").replace("oo", "inf")

def prettyAns(f):
    return sympy.pretty(
        f,
        use_unicode=True,
        num_columns=2147483647,
        mat_symbol_style="bold",
    ).replace("zoo", "ℂ∞").replace("nan", "NaN").replace("⋅", "∙")


@logging
def evalSym(f, prec=64, r=False):
    global BF_PREC
    BF_PREC = sympy.ceiling(int(prec) * 1.25)
    r = int(r)
    prec = int(prec)
    f, y = f.translate(ftrans), f
    for i in replacers:
        f = f.replace(i, replacers[i])
    try:
        if "\\" in y:
            raise SyntaxError
        f = parser.parse_expr(
            f,
            local_dict=None,
            global_dict=_globals,
            transformations=sym_tr,
            evaluate=True,
        )
    except SyntaxError:
        try:
            f = latex.parse_latex(y)
        except:
            f = latex.parse_latex(f)
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
    if isinstance(f, Plot) or f == plt:
        return [f]
    try:
        f = sympy.simplify(f)
    except:
        pass
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
    if prec:
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
        return [e, p]
    else:
        p = prettyAns(f)
        if p == convAns(f):
            p = ""
        return [f, p]


key = eval(sys.stdin.readline()).decode("utf-8", "replace").strip()


while True:
    try:
        locked = True
        args = eval(sys.stdin.readline()).decode("utf-8", "replace").strip().split("`")
        if len(args) > 3:
            args, key_in = args[:3], args[-1]
            if key_in == key:
                locked = False
        resp = evalSym(*args)
        if isinstance(resp[0], Plot):
            plt.rcParams["figure.figsize"] = (6.4, 4.8)
            ts = round(time.time() * 1000)
            name = str(ts) + ".png"
            fn = "cache/" + name
            try:
                resp[0].save(fn)
            except FileNotFoundError:
                fn = name
                resp[0].save(fn)
            plt.clf()
            s = "{'file':'" + fn + "'}\n"
        elif resp[0] == plt:
            ts = round(time.time() * 1000)
            name = str(ts) + ".png"
            fn = "cache/" + name
            try:
                plt.savefig(fn)
            except FileNotFoundError:
                fn = name
                plt.savefig(fn)
            plt.clf()
            plt.rcParams["figure.figsize"] = (6.4, 4.8)
            s = "{'file':'" + fn + "'}\n"
        else:
            s = repr([convAns(i) for i in resp])
        b = s.encode("utf-8")
        if len(b) > 8388608:
            raise OverflowError("Output data too large.")
        sys.stdout.write(repr(b) + "\n")
        sys.stdout.flush()
    except Exception as ex:
        sys.stdout.write(repr(ex) + "\n")
        sys.stdout.flush()
    time.sleep(0.01)
