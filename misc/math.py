import sympy, time, sys, traceback, random, numpy
import sympy.parsing.sympy_parser as parser
import sympy.parsing.latex as latex
import sympy.plotting as plotter
from sympy.plotting.plot import Plot
#latex.__builtins__["print"] = lambda *void1, **void2: None

key = "0"
BF_PREC = 256
BF_ALPHA = "0123456789abcdefghijklmnopqrstuvwxyz"

def tryWrapper(func):
    def __call__(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print(traceback.format_exc())
    return __call__


def printFile(s):
    f = open("log.txt", "ab")
    f.write(str(s).encode("utf-8"))
    f.close()


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


class baseFloat(sympy.Float):

    def __base__(self, b):
        self.base = b
        return self

    def __repr__(self):
        if not hasattr(self, "base"):
            self.base = 10
        return base(self.__add__(0).__str__(), self.base, sympy.ceiling(self._prec * sympy.log(2, 10)))

    @tryWrapper
    def evalf(self, prec):
        temp = baseFloat(self, prec * 1.25)
        temp.__base__(self.base)
        s = repr(temp)
        d = sympy.ceiling(prec / sympy.log(self.base, 10))
        try:
            d += s.index(".")
        except ValueError:
            pass
        if len(s) >= d:
            f = s.lower()
            up = f != s
            s = f
            x = s[d].lower()
            s = s[:d]
            if self.base != 64:
                i = BF_ALPHA.index(x)
                if i >= self.base / 2:
                    s = s[:-1] + BF_ALPHA[(1 + BF_ALPHA.index(s[-1])) % len(BF_ALPHA)]
            if up:
                s = s.upper()
        return s

    def nsimplify(self, **void):
        return self
    
    __str__ = __repr__


def base(x, b, p, alphabet=None, upper=True):
    """Converts a number from decimal to another base."""
    if not alphabet:
        alphabet = BF_ALPHA
    x = sympy.Float(str(x), dps=p)
    b = int(round(float(b)))
    if b == 64:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    elif b > len(alphabet):
        raise ValueError("Invalid number base.")
    if b < 2:
        raise ValueError("Invalid number base.")
    if x <= 0:
        if x == 0:
            return alphabet[0]
        else:
            return  "-" + base(-x, b, p, alphabet)
    r = sympy.ceiling(p / sympy.log(b, 10))
    for i in range(r):
        if x == round(x):
            break
        x *= b
        i += 1
    x = int(round(x))
    dp = bool(i)
    s = ""
    for j in range(i):
        x, d = divmod(x, b)
        if not j and r >= b / 2:
            d += 1
        s = alphabet[d] + s
    s = "." * dp + s
    while x:
        x, d = divmod(x, b)
        s = alphabet[d] + s
    if upper:
        s = s.upper()
    return s

def debase(x, b=10):
    """Converts a number from a base to decimal."""
    b = int(round(float(b)))
    i = str(x).lower()
    try:
        i = str(sympy.Number(i).evalf(BF_PREC))
    except:
        pass
    print(i)
    try:
        ind = i.index(".")
        f = i[ind + 1:]
        i = i[:ind]
    except ValueError:
        f = ""
    temp = str(int(i, b)) + "."
    fp = sympy.Rational(0)
    m = 1
    while f:
        m *= b
        fp += sympy.Rational(int(f[0], b)) / m
        f = f[1:]
    s = temp + str(fp.evalf(BF_PREC)).replace("0.", "")
    print(s)
    return rounder(sympy.Rational(s))

def h2d(x):
    return debase(x, 16)

def o2d(x):
    return debase(x, 8)

def b2d(x):
    return debase(x, 2)

def arbFloat(x, b):
    f = baseFloat(x, BF_PREC)
    return f.__base__(b)

def hexFloat(x):
    f = baseFloat(x, BF_PREC)
    return f.__base__(16)

def octFloat(x):
    f = baseFloat(x, BF_PREC)
    return f.__base__(8)

def binFloat(x):
    f = baseFloat(x, BF_PREC)
    return f.__base__(2)


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

def factorize(*args):
    temp = sympy.factorint(*args)
    output = []
    for k in temp:
        for i in range(temp[k]):
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


_globals = dict(sympy.__dict__)
plots = (
    "plot",
    "plot_parametric",
    "plot_implicit",
    "plot3d",
    "plot3d_parametric_line",
    "plot3d_parametric_surface",
    "lim",
)
for i in plots:
    _globals[i] = globals()[i]
_globals.update({
    "random": dice,
    "rand": dice,
    "dice": dice,
    "base": arbFloat,
    "h2d": h2d,
    "o2d": o2d,
    "b2d": b2d,
    "hex": hexFloat,
    "dec": debase,
    "oct": octFloat,
    "bin": binFloat,
    "plt": plot,
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
    "×": "*",
    "⋅": "*",
    "·": "*",
    "∑": "sum ",
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
    "0x": "h2d ",
    "0o": "o2d ",
    "0b": "b2d ",
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
        num_columns=2000,
        mat_symbol_style="bold",
    ).replace("zoo", "ℂ∞").replace("nan", "NaN")

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
        )
    except SyntaxError:
        try:
            f = latex.parse_latex(y)
        except:
            f = latex.parse_latex(f)
    try:
        if hasattr(f, "__class__") and issubclass(f.__class__, baseFloat):
            a = str(f.evalf(prec))
            try:
                b = str(prettyAns(sympy.Rational(str(f.num))))
            except:
                b = str(prettyAns(sympy.Number(str(f.num))))
            if b == a:
                b = ""
            return [a, b]
        f = sympy.simplify(f)
    except:
        p = prettyAns(f)
        if p == convAns(f):
            p = ""
        return [f, p]
    for i in sympy.preorder_traversal(f):
        try:
            f = f.subs(i, rounder(i))
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
            for i in preorder_traversal(e):
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

def readline(stream):
    output = ""
    t = time.time()
    while not "\n" in output:
        c = stream.read(1)
        if c:
            output += c
        else:
            time.sleep(0.003)
            if time.time() - t > 900:
                time.sleep(0.5)
    return output


while True:
    try:
        i = eval(readline(sys.stdin)).decode("utf-8").replace("\n", "").split("`")
        if len(i) <= 1:
            i.append("0")
        key = i[-1]
        resp = evalSym(*i[:-1])
        if isinstance(resp[0], Plot):
            resp[0].margin = 0.5
            fn = "cache/" + key + ".png"
            try:
                resp[0].save(fn)
            except FileNotFoundError:
                resp[0].save(key + ".png")
            s = "{'file':'" + fn + "'}\n"
        else:
            s = repr([convAns(i) for i in resp])
        sys.stdout.write(repr(s.encode("utf-8")) + "\n")
        sys.stdout.flush()
    except Exception as ex:
        raise
        sys.stdout.write(repr(ex) + "\n")
        sys.stdout.flush()
    time.sleep(0.01)
