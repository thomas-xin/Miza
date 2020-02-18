from sys import argv, stdout, stdin
from sympy import *
import sympy.parsing.sympy_parser as parser

sym_tr = parser.standard_transformations
sym_tr += (
    #parser.auto_symbol,
    parser.implicit_multiplication_application,
    parser.rationalize,
)


def roundMin(x):
    try:
        if x == int(x):
            return int(x)
        elif x == round(x):
            return round(x)
    except:
        pass
    return x


def evalSym(f, prec=None, r=False):
    f = parser.parse_expr(
        f,
        local_dict=None,
        global_dict=None,
        transformations=sym_tr,
    )
    try:
        f = simplify(f)
    except:
        return [f, ""]
    for i in preorder_traversal(f):
        if isinstance(i, Float):
            f = f.subs(i, roundMin(i))
    if prec:
        y = f.evalf(prec, chop=True)
        try:
            e = roundMin(y)
        except TypeError:
            e = y
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


while True:
    i = stdin.readline().replace("\n", "").split("`")
    s = repr(evalSym(*i)) + "\n"
    stdout.write(s)
    stdout.flush()
##    f = open("temp.txt", "a")
##    f.write(s)
##    f.close()
