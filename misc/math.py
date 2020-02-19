import sympy, math, time, sys
import sympy.parsing.sympy_parser as parser
import sympy.plotting as plotter
from sympy.plotting.plot import Plot
key = "0"

def plot(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot(*args, show=False, **kwargs)

def plot_parametric(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot_parametric(*args, show=False, **kwargs)

def plot_implicit(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot_implicit(*args, show=False, **kwargs)

def plot3d(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot3d(*args, show=False, **kwargs)

def plot3d_parametric_line(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot3d_parametric_line(*args, show=False, **kwargs)

def plot3d_parametric_surface(*args, **kwargs):
    if "show" in kwargs:
        kwargs.pop("show")
    return plotter.plot3d_parametric_surface(*args, show=False, **kwargs)

def lim(f, **kwargs):
    for i in kwargs:
        f = sympy.limit(f, i, kwargs[i])
    return f

def integrate(*args, **kwargs):
    try:
        return sympy.integrate(*args, **kwargs)
    except ValueError:
        return sympy.integrate(*args, sympy.Symbol("x"))

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
    "plt": plot,
    "factors": sympy.factorint,
    "factorize": sympy.primefactors,
    "factor": sympy.primefactors,
    "intg": integrate,
    "integral": integrate,
    "differentiate": sympy.diff,
    "derivative": sympy.diff,
    "derive": sympy.diff,
    "phi": sympy.GoldenRatio,
    "tau": sympy.pi * 2,
    "deg": sympy.pi / 180,
    "degrees": sympy.pi / 180,
    "degree": sympy.pi / 180,
    "rad": 1,
    "radians": 1,
    "radian": 1,
    "inf": math.inf,
    "nan": math.nan,
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
    parser.implicit_multiplication_application,
    parser.rationalize,
)


def rounder(x):
    try:
        if x == int(x):
            return int(x)
        elif x == round(x):
            return round(x)
    except:
        pass
    return x


translators = {
    "√": "sqrt ",
    "°": " deg",
    "÷": "/",
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
}

replacers = {
    "inf": "oo",
    "nan": "(oo-oo)",
    "INF": "oo",
    "NAN": "(oo-oo)",
    "TRUE": "True",
    "FALSE": "False",
}

ftrans = "".maketrans(translators)


def evalSym(f, prec=None, r=False):
    for i in replacers:
        f = f.replace(i, replacers[i])
    f = parser.parse_expr(
        f.translate(ftrans),
        local_dict=None,
        global_dict=_globals,
        transformations=sym_tr,
    )
    try:
        f = simplify(f)
    except:
        return [f, ""]
    for i in preorder_traversal(f):
        if isinstance(i, Float):
            f = f.subs(i, rounder(i))
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
                if isinstance(i, Float):
                    e = e.subs(i, rounder(i))
        if r:
            p = pretty(f)
            f = str(e)
            if p == f:
                p = ""
            return [f, p]
        return [e, ""]
    else:
        f = str(f)
        p = pretty(f)
        if p == f:
            p = ""
        return [f, p]


def readline(stream):
    output = ""
    t = time.time()
    while not "\n" in output:
        if time.time() - t > 900:
            sys.exit(1)
        c = stream.read(1)
        if c:
            output += c
        else:
            time.sleep(0.002)
    return output


while True:
    try:
        i = readline(sys.stdin).replace("\n", "").split("`")
        key = i[0]
        resp = evalSym(*i[1:])
        if isinstance(resp[0], Plot):
            resp[0].margin = 0
            fn = "cache/" + key + ".png"
            resp[0].save(fn)
            s = "{'file':'" + fn + "'}\n"
        else:
            s = repr([str(i) for i in resp]).replace("oo", "inf") + "\n"
        sys.stdout.write(s)
        sys.stdout.flush()
    except Exception as ex:
        sys.stderr.write(repr(ex) + "\n")
        sys.stderr.flush()
    time.sleep(0.01)
##    f = open("temp.txt", "a")
##    f.write(s)
##    f.close()
