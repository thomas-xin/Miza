"""
Adds many useful math-related functions.
"""

import traceback, time, datetime
import collections, ast, copy, pickle, io
import random, math, cmath, fractions, mpmath, sympy, shlex, numpy, colorsys, re, hashlib, base64

from scipy import interpolate, special, signal
from dateutil import parser as tparser
from sympy.parsing.sympy_parser import parse_expr
from itertools import repeat
from colormath import color_objects, color_conversions

loop = lambda x: repeat(None, x)

np = numpy
array = np.array
deque = collections.deque

random.seed(random.random() + time.time() % 1)
mp = mpmath.mp
mp.dps = 64

math.round = round

mpf = mpmath.mpf
mpc = mpmath.mpc
Mat = mat = matrix = mpmath.matrix

inf = Infinity = math.inf
nan = math.nan
eval_const = {
    "none": None,
    "null": None,
    "NULL": None,
    "true": True,
    "false": False,
    "TRUE": True,
    "FALSE": False,
    "inf": inf,
    "nan": nan,
    "Infinity": inf,
}
null = None
i = I = j = J = 1j
pi = mp.pi
E = e = mp.e
c = 299792458
lP = 1.61625518e-35
mP = 2.17643524e-8
tP = 5.39124760e-44
h = 6.62607015e-34
G = 6.6743015e-11
g = 9.80665
tau = pi * 2
d2r = mp.degree
phi = mp.phi
euler = mp.euler
twinprime = mp.twinprime

Function = sympy.Function
Symbol = sympy.Symbol
factorize = factorint = primeFactors = sympy.ntheory.factorint
mobius = sympy.ntheory.mobius

TRUE, FALSE = True, False
true, false = True, False


nop = lambda *void1, **void2: None


def shuffle(it):
    if type(it) is list:
        random.shuffle(it)
        return it
    elif type(it) is tuple:
        it = list(it)
        random.shuffle(it)
        return it
    elif type(it) is dict:
        ir = shuffle(list(it))
        new = {}
        for i in ir:
            new[i] = it[i]
        it.clear()
        it.update(new)
        return it
    elif type(it) is deque:
        it = list(it)
        random.shuffle(it)
        return deque(it)
    elif isinstance(it, hlist):
        temp = it.shuffle()
        it.data = temp.data
        it.offs = temp.offs
        it.size = temp.size
        del temp
        return it
    else:
        try:
            it = list(it)
            random.shuffle(it)
            return it
        except TypeError:
            raise TypeError("Shuffling " + type(it) + " is not supported.")

def reverse(it):
    if type(it) is list:
        return list(reversed(it))
    elif type(it) is tuple:
        return list(reversed(it))
    elif type(it) is dict:
        ir = tuple(reversed(it))
        new = {}
        for i in ir:
            new[i] = it[i]
        it.clear()
        it.update(new)
        return it
    elif type(it) is deque:
        return deque(reversed(it))
    elif isinstance(it, hlist):
        temp = it.reverse()
        it.data = temp.data
        it.offs = temp.offs
        it.chash = None
        del temp
        return it
    else:
        try:
            return list(reversed(it))
        except TypeError:
            raise TypeError("Reversing " + type(it) + " is not supported.")

def sort(it, key=None, reverse=False):
    if type(it) is list:
        it.sort(key=key, reverse=reverse)
        return it
    elif type(it) is tuple:
        it = sorted(it, key=key, reverse=reverse)
        return it
    elif issubclass(type(it), collections.abc.Mapping):
        keys = sorted(it, key=it.get if key is None else lambda x: key(it.get(x)))
        if reverse:
            keys = reversed(keys)
        items = tuple((i, it[i]) for i in keys)
        it.clear()
        it.__init__(items)
        return it
    elif type(it) is deque:
        it = sorted(it, key=key, reverse=reverse)
        return deque(it)
    elif isinstance(it, hlist):
        it.__init__(sorted(it, key=key, reverse=reverse))
        it.chash = None
        return it
    else:
        try:
            it = list(it)
            it.sort(key=key, reverse=reverse)
            return it
        except TypeError:
            raise TypeError("Sorting " + type(it) + " is not supported.")


phase = cmath.phase
sin = mpmath.sin
cos = mpmath.cos
tan = mpmath.tan
sec = mpmath.sec
csc = mpmath.csc
cot = mpmath.cot
sinh = mpmath.sinh
cosh = mpmath.cosh
tanh = mpmath.tanh
sech = mpmath.sech
csch = mpmath.csch
coth = mpmath.coth
asin = mpmath.asin
acos = mpmath.acos
atan = mpmath.atan
asec = mpmath.asec
acsc = mpmath.acsc
acot = mpmath.acot
asinh = mpmath.asinh
acosh = mpmath.acosh
atanh = mpmath.atanh
asech = mpmath.asech
acsch = mpmath.acsch
acoth = mpmath.acoth
sinc = mpmath.sinc
atan2 = mpmath.atan2
ei = mpmath.ei
e1 = mpmath.e1
en = mpmath.expint
li = mpmath.li
si = mpmath.si
ci = mpmath.ci
shi = mpmath.shi
chi = mpmath.chi
erf = mpmath.erf
erfc = mpmath.erfc
erfi = mpmath.erfi
aerf = mpmath.erfinv
npdf = mpmath.npdf
ncdf = mpmath.ncdf
fac = factorial = mpmath.fac
fib = fibonacci = mpmath.fib
trib = tribonacci = sympy.tribonacci
luc = lucas = sympy.lucas
harm = harmonic = sympy.harmonic
ber = bernoulli = mpmath.bernoulli
eul = eulernum = mpmath.eulernum
sqrt = mpmath.sqrt
hypot = mpmath.hypot
cbrt = mpmath.cbrt
root = mpmath.root
exp = mpmath.exp
expi = expj = mpmath.expj
log = mpmath.log
ln = mpmath.ln
frac = sympy.frac


bits = lambda x: (i for i in range((x if type(x) is int else int(x)).bit_length()) if x & (1 << i))


def isqrt(x):
    x = int(x)
    y = (x << 2) // 3
    b = y.bit_length()
    a = b >> 1
    if b & 1:
        c = 1 << a
        d = (c + (x >> a)) >> 1
    else:
        c = (3 << a) >> 2
        d = (c + (y >> a)) >> 1
    if c != d:
        c = d
        d = (c + x // c) >> 1
        while d < c:
            c = d
            d = (c + x // c) >> 1
    return c


def round(x, y=None):
    try:
        if isValid(x):
            try:
                if x == int(x):
                    return int(x)
                if y is None:
                    return int(math.round(x))
            except:
                pass
            return roundMin(math.round(x, y))
        else:
            return x
    except:
        if type(x) is complex:
            return round(x.real, y) + round(x.imag, y) * 1j
    try:
        return math.round(x)
    except:
        return x

def round_random(x):
    if x == round(x):
        return round(x)
    x, y = divmod(x, 1)
    if random.random() <= y:
        x += 1
    return int(x)

def ceil(x):
    try:
        return math.ceil(x)
    except:
        if type(x) is complex:
            return ceil(x.real) + ceil(x.imag) * 1j
    try:
        return math.ceil(x)
    except:
        return x

def floor(x):
    try:
        return math.floor(x)
    except:
        if type(x) is complex:
            return floor(x.real) + floor(x.imag) * 1j
    try:
        return math.floor(x)
    except:
        return x

def trunc(x):
    try:
        return math.trunc(x)
    except:
        if type(x) is complex:
            return trunc(x.real) + trunc(x.imag) * 1j
    try:
        return math.trunc(x)
    except:
        return x


sqr = lambda x: ((sin(x) >= 0) << 1) - 1

saw = lambda x: (x / pi + 1) % 2 - 1

tri = lambda x: (abs((0.5 - x / pi) % 2 - 1)) * 2 - 1

sgn = lambda x: (((x > 0) << 1) - 1) * (x != 0)


frand = lambda x=1, y=0: (random.random() * max(x, y) / mpf(random.random())) % x + y

def xrand(x, y=None, z=0):
    if y == None:
        y = 0
    if x == y:
        return x
    return random.randint(floor(min(x, y)), ceil(max(x, y)) - 1) + z

rrand = lambda x=1, y=0: frand(x) ** (1 - y)


def modularInv(a, b):
    if b == 0:
        return (1, a)
    a %= b
    x = 0
    y = 1
    while a:
        d = divmod(b, a)
        a, b = d[1], a
        x, y = y, x - (d[0]) * y
    return (x, 1)


def pisanoPeriod(x):
    a, b = 0, 1
    for i in range(0, x * x):
        a, b = b, (a + b) % x
        if a == 0 and b == 1:
            return i + 1


def jacobi(a, n):
    if a == 0 or n < 0:
        return 0
    x = 1
    if a < 0:
        a = -a
        if n & 3 == 3:
            x = -x
    if a == 1:
        return x
    while a:
        if a < 0:
            a = -a
            if n & 3 == 3:
                x = -x
        while not a & 1:
            a >>= 1
            if n & 7 == 3 or n & 7 == 5:
                x = -x
        a, n = n, a
        if a & 3 == 3 and n & 3 == 3:
            x = -x
        a %= n
        if a > n >> 1:
            a -= n
    if n == 1:
        return x
    return 0


def next6np(start=0):
    if start <= 2:
        yield 2
    if start <= 3:
        yield 3
    x = start - start % 6 + 6
    if x > 6 and x - start >= 5:
        yield x - 5
    while True:
        yield x - 1
        yield x + 1
        x += 6


def isPrime(n):
    
    def divisibility(n):
        t = min(n, 2 + ceil(log(n) ** 2))
        g = next6np()
        while True:
            p = next(g)
            if p >= t:
                break
            if n % p == 0:
                return False
        return True

    def fermat(n):
        t = min(n, 2 + ceil(log(n)))
        g = next6np()
        while True:
            p = next(g)
            if p >= t:
                break
            if pow(p, n - 1, n) != 1:
                return False
        return True

    def miller(n):
        d = n - 1
        while d & 1 == 0:
            d >>= 1
        t = min(n, 2 + ceil(log(n)))
        g = next6np()
        while True:
            p = next(g)
            if p >= t:
                break
            x = pow(p, d, n)
            if x == 1 or x == n - 1:
                continue
            while n != d + 1:
                x = (x * x) % n
                d <<= 1
                if x == 1:
                    return False
                if x == n - 1:
                    break
            if n == d + 1:
                return False
        return True

    def solovoyStrassen(n):
        t = min(n, 2 + ceil(log(n)))
        g = next6np()
        while True:
            p = next(g)
            if p >= t:
                break
            j = (n + jacobi(p, n)) % n
            if j == 0:
                return False
            m = pow(p, (n - 1) >> 1, n)
            if m != j:
                return False
        return True

    i = int(n)
    if n == i:
        n = i
        if n < 2:
            return False
        if n <= 3:
            return True
        t = n % 6
        if t != 1 and t != 5:
            return False
        if not divisibility(n):
            return False
        if not fermat(n):
            return False
        if not miller(n):
            return False
        if not solovoyStrassen(n):
            return False
        return True
    return None

def generatePrimes(a=2, b=inf, c=1):
    primes = hlist()
    a = round(a)
    b = round(b)
    if b is None:
        a, b = 0, a
    if a > b:
        a, b = b, a
    a = max(1, a)
    g = next6np(a)
    while c:
        p = next(g)
        if p >= b:
            break
        if isPrime(p):
            c -= 1
            primes.append(p)
    return primes


def iterSum(it):
    if issubclass(type(it), collections.abc.Mapping):
        return sum(tuple(it.values()))
    try:
        return sum(iter(it))
    except TypeError:
        return it

def iterMax(it):
    if issubclass(type(it), collections.abc.Mapping):
        keys, values = tuple(it.keys()), tuple(it.values())
        m = max(values)
        for i in keys:
            if it[i] >= m:
                return i
    try:
        return max(iter(it))
    except TypeError:
        return it

def setDict(d, k, v, ignore=False):
    try:
        v = d.__getitem__(k)
        if v is None and ignore:
            raise LookupError
    except LookupError:
        d.__setitem__(k, v)
    return v

def addDict(a, b, replace=True, insert=None):
    if type(a) is not dict:
        if replace:
            r = b
        else:
            r = copy.copy(b)
        try:
            r[insert] += a
        except KeyError:
            r[insert] = a
        return r
    elif type(b) is not dict:
        if replace:
            r = a
        else:
            r = copy.copy(a)
        try:
            r[insert] += b
        except KeyError:
            r[insert] = b
        return r
    else:
        if replace:
            r = a
        else:
            r = copy.copy(a)
        for k in b:
            try:
                temp = a[k]
            except KeyError:
                r[k] = b[k]
                continue
            if type(temp) is dict or type(b[k]) is dict:
                r[k] = addDict(b[k], temp, replace)
                continue
            r[k] = b[k] + temp
    return r

def subDict(d, key):
    output = dict(d)
    try:
        key[0]
    except TypeError:
        key = [key]
    for k in key:
        try:
            output.pop(k)
        except KeyError:
            pass
    return output


def roundMin(x):
    if type(x) is not complex:
        if isValid(x) and x == int(x):
            return int(x)
        else:
            return x
    else:
        x = complex(x)
        if x.imag == 0:
            return roundMin(x.real)
        else:
            return roundMin(complex(x).real) + roundMin(complex(x).imag) * (1j)


def closeRound(n):
    rounds = [0.125, 0.375, 0.625, 0.875, 0.25, 0.5, 0.75, 1 / 3, 2 / 3]
    a = math.floor(n)
    b = n % 1
    c = round(b, 1)
    for i in range(0, len(rounds)):
        if abs(b - rounds[i]) < 0.02:
            c = rounds[i]
    return mpf(a + c)


def toFrac(num, limit=2147483647):
    if num >= limit:
        return [limit, 1]
    if num <= 0:
        return [1, limit]
    num = mpf(num)
    f = fractions.Fraction(num).limit_denominator(limit)
    frac = [f.numerator, f.denominator]
    if frac[0] == 0:
        return [1, limit]
    return frac


def gcd(x, y=1):
    if y != 1:
        while y > 0:
            x, y = y, x % y
        return x
    return x

def lcm2(x, y=1):
    if x != y:
        x = abs(x)
        y = abs(y)
        i = True
        if x != int(x):
            i = False
            x = toFrac(x)[0]
        if y != int(y):
            i = False
            y = toFrac(y)[0]
        if i:
            return x * y // gcd(x, y)
        else:
            return toFrac(x / y)[0]
    return x

def lcm(*x):
    try:
        while True:
            x = [i for j in x for i in j]
    except:
        if 0 in x:
            raise ValueError("Cannot find LCM of zero.")
        while len(x) > 1:
            x = [lcm2(x[i], x[-i - 1]) for i in range(ceil(len(x) / 2))]
    return x[-1]

def lcmRange(x):
    primes = generatePrimes(1, x, -1)
    y = 1
    for p in primes:
        y *= p ** floor(log(x, p))
    return y


mean = lambda *nums: roundMin(np.mean(np.array(nums)))


def pwr(x, power=2):
    if x.real >= 0:
        return roundMin(x ** power)
    else:
        return roundMin(-((-x) ** power))


def pulse(x, y=0.5):
    p = y * tau
    x *= 0.5 / len(x) * (x < p) + 0.5 / (1 - len(x)) * (x >= p)
    return x


isnan = cmath.isnan


def isValid(x):
    if type(x) is complex:
        return not (cmath.isinf(x) or cmath.isnan(x))
    try:
        if type(x) is int:
            return True
        return x.is_finite()
    except:
        return math.isfinite(x)


def approach(x, y, z, threshold=0.125):
    if z <= 1:
        x = y
    else:
        x = (x * (z - 1) + y) / z
        if abs(x - y) <= threshold / z:
            x = y
    return x


def scaleRatio(x, y):
    try:
        return x * (x - y) / (x + y)
    except ZeroDivisionError:
        return 0


def xrange(a, b=None, c=None):
    if b == None:
        b = ceil(a.real)
        a = 0
    if c == None:
        if a > b:
            c = -1
        else:
            c = 1
    return range(floor(a.real), ceil(b.real), c)


def romanNumerals(num, order=0):
    num = int(num)
    carry = 0
    over = ""
    sym = ""
    output = ""
    if num >= 4000:
        carry = num // 1000
        num %= 1000
        over = romanNumerals(carry, order + 1)
    while num >= 1000:
        num -= 1000
        output += "M"
    if num >= 900:
        num -= 900
        output += "CM"
    elif num >= 500:
        num -= 500
        output += "D"
    elif num >= 400:
        num -= 400
        output += "CD"
    while num >= 100:
        num -= 100
        output += "C"
    if num >= 90:
        num -= 90
        output += "XC"
    elif num >= 50:
        num -= 50
        output += "L"
    elif num >= 40:
        num -= 40
        output += "XL"
    while num >= 10:
        num -= 10
        output += "X"
    if num >= 9:
        num -= 9
        output += "IX"
    elif num >= 5:
        num -= 5
        output += "V"
    elif num >= 4:
        num -= 4
        output += "IV"
    while num >= 1:
        num -= 1
        output += "I"
    if output != "":
        if order == 1:
            sym = "ᴍ"
        elif order == 2:
            sym = "ᴍᴹ"
    return over + output + sym


def limStr(s, maxlen=10):
    if maxlen is None:
        return s
    if type(s) is not str:
        s = str(s)
    over = (len(s) - maxlen) / 2
    if over > 0:
        half = len(s) / 2
        s = s[: ceil(half - over - 1)] + ".." + s[ceil(half + over + 1) :]
    return s


def expNum(num, maxlen=10, decimals=0):
    if not isValid(num):
        if num.real > 0:
            return "inf"
        elif num.real < 0:
            return "-inf"
        else:
            return "NaN"
    if type(num) is complex:
        i = expNum(num.imag, maxlen // 2 - 1, decimals)
        p = "+" if num.imag > 0 else ""
        return expNum(num.real, ceil(maxlen / 2) - 1, decimals) + p + i + "i"
    if num < 0:
        n = "-"
        num = -num
    else:
        n = ""
    try:
        numlen = floor(num.log10())
    except:
        numlen = floor(math.log10(max(0.001, num)))
    if log(max(0.001, num), 10) <= maxlen - decimals:
        return n + roundX(num, min(maxlen - numlen - 2 - len(n), decimals))
    else:
        if numlen > 0:
            try:
                loglen = floor(numlen.log10())
            except:
                loglen = floor(math.log10(numlen)) + len(n)
        else:
            loglen = 0
        s = roundX(num / 10 ** numlen, maxlen - loglen - 5)[: max(1, maxlen - loglen - 2)]
        if s[:3] == "10.":
            s = "9." + "9" * (maxlen - loglen - 4)
        return n + s + "e+" + str(numlen)


def roundX(num, prec):
    if prec > 0:
        s = str(round(num.real, round(prec)))
        if "." in s:
            while len(s) - s.index(".") <= prec:
                s += "0"
        else:
            s += "." + "0" * prec
        return s
    else:
        return str(round(num.real))


def verifyString(string):
    if type(string) is list or type(string) is tuple:
        return "".join(str(c) for c in string)
    else:
        if type(string) is not str:
            string = str(string)
        return string


colourCalculation = lambda a, offset=0: adjColour(colorsys.hsv_to_rgb((a / 1536) % 1, 1, 1), offset, 255)

def colour2Raw(c):
    if len(c) == 3:
        return (c[0] << 16) + (c[1] << 8) + c[2]
    else:
        return (c[0] << 16) + (c[1] << 8) + c[2] + (c[3] << 24)

def raw2Colour(x):
    if x > 1 << 24:
        return verifyColour(((x >> 16) & 255, (x >> 8) & 255, x & 255, (x >> 24) & 255))
    else:
        return verifyColour(((x >> 16) & 255, (x >> 8) & 255, x & 255))

rgb_to_hsv = lambda c: list(colorsys.rgb_to_hsv(*c[:3])) + c[3:]
hsv_to_rgb = lambda c: list(colorsys.hsv_to_rgb(*c[:3])) + c[3:]
rgb_to_cmy = lambda c: [1 - x for x in c[:3]] + c[3:]
cmy_to_rgb = rgb_to_cmy
rgb_to_lab = lambda c: list(color_conversions.convert_color(color_objects.sRGBColor(*c[:3]), color_objects.LabColor).get_value_tuple()) + c[3:]
lab_to_rgb = lambda c: list(color_conversions.convert_color(color_objects.LabColor(*c[:3]), color_objects.sRGBColor).get_value_tuple()) + c[3:]
rgb_to_luv = lambda c: list(color_conversions.convert_color(color_objects.sRGBColor(*c[:3]), color_objects.LuvColor).get_value_tuple()) + c[3:]
luv_to_rgb = lambda c: list(color_conversions.convert_color(color_objects.LuvColor(*c[:3]), color_objects.sRGBColor).get_value_tuple()) + c[3:]
rgb_to_xyz = lambda c: list(color_conversions.convert_color(color_objects.sRGBColor(*c[:3]), color_objects.XYZColor).get_value_tuple()) + c[3:]
xyz_to_rgb = lambda c: list(color_conversions.convert_color(color_objects.XYZColor(*c[:3]), color_objects.sRGBColor).get_value_tuple()) + c[3:]

hex2Colour = lambda h: verifyColour(hex2Bytes(h))

luma = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def verifyColour(c):
    c = list(c)
    for i in range(len(c)):
        if c[i] > 255:
            c[i] = 255
        elif c[i] < 0:
            c[i] = 0
        c[i] = int(abs(c[i]))
    return c

def fillColour(a):
    if type(a) is complex:
        a = a.real
    if a > 255:
        a = 255
    elif a < 0:
        a = 0
    a = round(a)
    return verifyColour([a, a, a])

def negColour(c, t=127):
    i = luma(c)
    if i > t:
        return fillColour(0)
    else:
        return fillColour(255)

invColour = lambda c: [255 - i for i in c]

def adjColour(colour, brightness=0, intensity=1, hue=0, bits=0, scale=False):
    if hue != 0:
        h = list(colorsys.rgb_to_hsv(*(array(colour) / 255)))
        c = adjColour(colorsys.hsv_to_rgb((h[0] + hue) % 1, h[1], h[2]), intensity=255)
    else:
        c = list(colour)
    for i in range(len(c)):
        c[i] = round(c[i] * intensity + brightness)
    if scale:
        for i in range(len(c)):
            if c[i] > 255:
                for j in range(len(c)):
                    if i != j:
                        c[j] += c[i] - 255
                c[i] = 255
    c = bitCrush(c, bits)
    return verifyColour(c)


def bitCrush(dest, b=0, f=round):
    try:
        a = 1 << b
    except:
        a = 2 ** b
    try:
        len(dest)
        dest = list(dest)
        for i in range(len(dest)):
            dest[i] = f(dest[i] / a) * a
    except TypeError:
        try:
            dest = f(dest / a) * a
        except:
            raise
    return dest


def listPermutation(dest):
    order = [0 for i in range(len(dest))]
    for i in range(len(dest)):
        for j in range(i, len(dest)):
            if dest[i] > dest[j]:
                order[i] += 1
            elif dest[i] < dest[j]:
                order[j] += 1
    return order


def multiVectorScalarOp(dest, operator):
    expression = "a" + operator + "b"
    function = eval("lambda a,b: " + expression)
    output = []
    for i in range(len(dest[0])):
        s = 0
        for j in range(len(dest)):
            s = function(s, dest[j][i])
        output.append(s)
    return output

def vectorVectorOp(dest, source, operator):
    expression = "dest[i]" + operator + "source[i]"
    function = eval("lambda dest,source,i: " + expression)
    for i in range(len(source)):
        dest[i] = function(dest, source, i)
    return dest

def vectorScalarOp(dest, source, operator):
    expression = "dest[i]" + operator + str(source)
    function = eval("lambda dest,i: " + expression)
    for i in range(len(dest)):
        dest[i] = function(dest, i)
    return dest


def resizeVector(v, length, mode=5):
    size = len(v)
    new = round(length)
    if new == size:
        resized = v
    elif mode == 0:
        resized = np.array([v[round(i / new * size) % size] for i in range(new)])
    elif mode <= 5 and mode == int(mode):
        spl = interpolate.splrep(np.arange(1 + size), np.append(v, v[0]), k=int(min(size, mode)))
        resized = np.array([interpolate.splev((i / new * size) % size, spl) for i in range(new)])
    elif mode <= 5:
        if math.floor(mode) == 0:
            resized1 = resizeVector(v, new, 0)
        else:
            spl1 = interpolate.splrep(np.arange(1 + size), np.append(v, v[0]), k=floor(min(size, mode)))
            resized1 = np.array([interpolate.splev((i / new * size) % size, spl1) for i in range(new)])
        spl2 = interpolate.splrep(np.arange(1 + size), np.append(v, v[0]), k=ceil(min(size, mode)))
        resized2 = np.array([interpolate.splev((i / new * size) % size, spl2) for i in range(new)])
        resized = resized1 * (1 - mode % 1) + (mode % 1) * resized2
    else:
        resizing = []
        for i in range(1, floor(mode)):
            resizing.append(resizeVector(v, new, i / floor(mode) * 5))
        resized = np.mean(resizing, 0)
    return resized

def get(v, i, mode=5):
    size = len(v)
    i = i.real + i.imag * size
    if i == int(i) or mode == 0:
        return v[round(i) % size]
    elif mode > 0 and mode < 1:
        return get(v, i, 0) * (1 - mode) + mode * get(v, i, 1)
    elif mode == 1:
        return v[floor(i) % size] * (1 - i % 1) + v[ceil(i) % size] * (i % 1)
    elif mode == int(mode):
        return roundMin(interpolate.splev(i, interpolate.splrep(np.arange(1 + size), np.append(v, v[0]), k=int(min(size, mode)))))
    else:
        return get(v, i, floor(mode)) * (1 - mode % 1) + (mode % 1) * get(v, i, ceil(mode))


def product(*nums):
    p = 1
    for i in nums:
        p *= i
    return p

def dotProduct(*vects):
    if len(vects) > 1:
        return sum(product(*(array(v) for v in vects)))
    else:
        return sum((i ** 2 for i in vects[-1]))


def limitList(source, dest, direction=False):
    for i in range(len(source)):
        if direction:
            if source[i] < dest[i]:
                source[i] = dest[i]
        else:
            if source[i] > dest[i]:
                source[i] = dest[i]
    return source


randomPolarCoord = lambda x=1: polarCoords(frand(x), frand(tau))

def polarCoords(dist, angle, pos=None):
    p = dist * array([math.cos(angle), math.sin(angle)])
    if pos is None:
        return p
    return p + pos

def cartesianCoords(x, y, pos=None):
    if pos is None:
        d = array(x, y)
    else:
        d = array(x, y) - array(pos)
    return array([hypot(*d), atan2(*reversed(d))])


def convertRect(rect, edge=0):
    dest_rect = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
    if dest_rect[0] > dest_rect[2]:
        dest_rect[0], dest_rect[2] = dest_rect[2], dest_rect[0]
    if dest_rect[1] > dest_rect[3]:
        dest_rect[1], dest_rect[3] = dest_rect[3], dest_rect[1]
    dest_rect[0] += edge
    dest_rect[1] += edge
    dest_rect[2] -= edge
    dest_rect[3] -= edge
    return dest_rect

def inRect(pos, rect, edge=0):
    dest_rect = convertRect(rect, edge)
    if pos[0] - dest_rect[0] <= 0:
        return False
    if pos[1] - dest_rect[1] <= 0:
        return False
    if pos[0] - dest_rect[2] > 0:
        return False
    if pos[1] - dest_rect[3] > 0:
        return False
    return True

def toRect(pos, rect, edge=0):
    p = list(pos)
    if not all(isValid(i) for i in pos):
        return p, True, True
    dest_rect = convertRect(rect, 0)
    lr, ud = False, False
    for _ in loop(4):
        diff = p[0] - dest_rect[0] - edge
        if diff <= 0:
            p[0] = dest_rect[0] - diff + edge
            lr = True
            continue
        diff = p[1] - dest_rect[1] - edge
        if diff <= 0:
            p[1] = dest_rect[1] - diff + edge
            ud = True
            continue
        diff = p[0] - dest_rect[2] + edge
        if diff > 0:
            p[0] = dest_rect[2] - diff - edge
            lr = True
            continue
        diff = p[1] - dest_rect[3] + edge
        if diff > 0:
            p[1] = dest_rect[3] - diff - edge
            ud = True
            continue
    return p, lr, ud

def rdRect(pos, rect, edge=0):
    dest_rect = convertRect(rect, edge)
    if not inRect(pos, rect, edge):
        s = array(dest_rect[:2])
        t = array(pos)
        p = array(dest_rect[2:]) - s
        m = p / 2
        diff = t - s - m
        angle = atan2(*reversed(diff))
        vel = polarCoords(hypot(*m), angle)
        pos = vel + s + m
    return pos


def diffExpD(r, s, t):
    if r == 1:
        return s * t
    else:
        return log(s * (r ** t - 1), r)

def diffExpT(r, s, d):
    coeff = d * log(r) / s + 1
    if coeff < 0:
        return inf
    else:
        return log(coeff, r)

def predictTrajectory(src, dest, vel, spd, dec=1, boundary=None, edge=0):
    pos = array(dest)
    dist = hypot(*(src - dest))
    for _ in loop(64):
        time = diffExpT(dec, spd, dist)
        new_pos = dest + vel * min(time, 1 << 32)
        if boundary:
            new_pos = array(toRect(new_pos, boundary, edge)[0])
        new_dist = hypot(*(new_pos - pos))
        pos = new_pos
        dist = hypot(*(src - pos))
        if new_dist < 0.0625:
            break
    return pos


def collisionCheck(pos1, pos2, vel1, vel2, mass1, mass2, radius1, radius2):
    diff = pos1 - pos2
    dist = frameDistance(pos1, pos2, -vel1, -vel2)
    mindist = radius1 + radius2
    if dist < mindist:
        pos1, pos2 = array(pos1), array(pos2)
        vel1, vel2 = array(vel1), array(vel2)
        dist -= 1
        angle = atan2(*reversed(diff))
        mov = polarCoords(mindist - dist + 1, angle)
        p1 = mass1 * hypot(*vel1)
        p2 = mass2 * hypot(*vel2)
        r = p1 / max((p1 + p2), 0.1)
        v1 = mov * (1 - r)
        v2 = mov * -r
        totalmass = mass1 + mass2
        coeff1 = mass2 / totalmass * 2
        coeff2 = mass1 / totalmass * 2
        vect1 = diff
        vect2 = -vect1
        pos1 += v1
        pos2 += v2
        veld1 = vel1 - vel2
        veld2 = -veld1
        arg1 = dotProduct(veld1, vect1) / dotProduct(vect1)
        arg2 = dotProduct(veld2, vect2) / dotProduct(vect2)
        vect1 *= coeff1 * arg1
        vect2 *= coeff2 * arg2
        vel1 -= vect1
        vel2 -= vect2
        hit = True
    else:
        hit = False
    return hit, pos1, pos2, vel1, vel2


def angleDifference(angle1, angle2, unit=tau):
    angle1 %= unit
    angle2 %= unit
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1
    a = abs(angle2 - angle1)
    b = abs(angle2 - unit - angle1)
    return min(a, b)

def angleDistance(angle1, angle2, unit=tau):
    angle1 %= unit
    angle2 %= unit
    a = angle2 - angle1
    b = angle2 - unit - angle1
    c = angle2 + unit - angle1
    return sorted((a, b, c), key=lambda x: abs(x))[0]


def frameDistance(pos1, pos2, vel1, vel2):
    line1 = [pos1 - vel1, pos1]
    line2 = [pos2 - vel2, pos2]
    return intervalIntervalDist(line1, line2)

def intervalIntervalDist(line1, line2):
    if intervalsIntersect(line1, line2):
        return 0
    distances = [
        pointIntervalDist(line1[0], line2),
        pointIntervalDist(line1[1], line2),
        pointIntervalDist(line2[0], line1),
        pointIntervalDist(line2[1], line1)]
    return min(distances)

def pointIntervalDist(point, line):
    px, py = point
    x1, x2 = line[0][0], line[1][0]
    y1, y2 = line[0][1], line[1][1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        dx = px - x1 - t * dx
        dy = py - y1 - t * dy
    return hypot(dx, dy)

def intervalsIntersect(line1, line2):
    x11, y11 = line1[0]
    x12, y12 = line1[1]
    x21, y21 = line2[0]
    x22, y22 = line2[1]
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21
    delta = dx2 * dy1 - dy2 * dx1
    if delta == 0:
        return False
    s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
    t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
    return (0 <= s <= 1) and (0 <= t <= 1)


def func2Array(func, size=4096):
    function = eval("lambda x: " + str(func))
    period = 2 * pi
    array = function(np.arange(0, period, 1 / (size + 1) * period))
    return array

def array2Harmonics(data, precision=1024):
    output = []
    T = len(data)
    t = np.arange(T)
    for n in range(precision + 1):
        if n > T / 2 + 1:
            output.append(np.array((0, 0)))
        else:
            bn = 2 / T * (data * np.cos(2 * pi * n * t / T)).sum()
            an = 2 / T * (data * np.sin(2 * pi * n * t / T)).sum()
            R = np.sqrt(an ** 2 + bn ** 2)
            p = np.arctan2(bn, an)
            if R == 0:
                p = 0
            output.append(np.array((R, p)))
    return np.array(output[1 : precision + 1])

def harmonics2Array(period, harmonics, func="sin(x)"):
    expression = func
    function = eval("lambda x: " + expression)
    result = 0
    t = np.arange(period)
    for n, (a, b) in enumerate(harmonics):
        result += a * function((n + 1) * t * 2 * pi / period + b)
    return result


def limLine(s, lim):
    curr = s
    if len(curr) > lim:
        temp = curr.split(" ")
        final = ""
        string = ""
        for t in temp:
            if len(string) + len(t) > lim:
                final += string[:-1] + "\n"
                string = ""
            string += t + " "
        s = final + string[:-1]
    return s


def strGetRem(s, arg):
    if arg + " " in s:
        s = s.replace(arg + " ", "")
        return s, True
    elif " " + arg in s:
        s = s.replace(" " + arg, "")
        return s, True
    else:
        return s, False


def strIter(it, key=None, limit=1728, offset=0, left="[", right="]"):
    try:
        try:
            len(it)
        except TypeError:
            it = hlist(i for i in it)
    except:
        it = hlist(it)
    if issubclass(type(it), collections.abc.Mapping):
        keys = it.keys()
        add = None
    else:
        keys = range(len(it))
        add = offset
    s = ""
    i = offset
    for k in keys:
        s += "\n" + left
        if type(k) is not str:
            s += " " * (int(math.log10(len(it))) - int(math.log10(max(1, i))))
            if add is not None:
                s += str(k + add)
            else:
                s += str(k)
        else:
            s += k
        s += right + " "
        if key is None:
            s += str(it[k])
        else:
            s += str(key(it[k]))
        i += 1
    return limStr(s, limit)


def intKey(d):
    c = {}
    for k in tuple(d):
        try:
            t = d[k]
        except KeyError:
            continue
        try:
            k = int(k)
        except (TypeError, ValueError):
            pass
        if type(t) is dict:
            t = intKey(t)
        c[k] = t
    return c


utc = time.time
utc_dt = datetime.datetime.utcnow
ep = datetime.datetime(1970, 1, 1)
utc_ts = lambda dt: (dt - ep).total_seconds()
# utc_ts = lambda dt: dt.replace(tzinfo=datetime.timezone.utc).timestamp()

TIMEUNITS = {
    "galactic year": 7157540528801820.28133333333333,
    "millenium": [31556925216., "millenia"],
    "century": [3155692521.6, "centuries"],
    "decade": 315569252.16,
    "year": 31556925.216,
    "month": 2629743.768,
    "week": 604800.,
    "day": 86400.,
    "hour": 3600.,
    "minute": 60.,
    "second": 1,
}

def timeConv(s):
    if not isValid(s):
        high = "galactic years"
        return [str(s) + " " + high]
    r = s < 0
    s = abs(s)
    taken = []
    for i in TIMEUNITS:
        a = None
        t = m = TIMEUNITS[i]
        if type(t) is list:
            t = t[0]
        if type(t) is int:
            a = round(s, 3)
        elif s >= t:
            a = int(s // t)
            s = s % t
        if a:
            if a != 1:
                if type(m) is list:
                    i = m[1]
                else:
                    i += "s"
            taken.append("-" * r + str(roundMin(a)) + " " + str(i))
    if not len(taken):
        return [str(roundMin(s)) + " seconds"]
    return taken

sec2Time = lambda s: " ".join(timeConv(s))

def dhms(s):
    s = float(s)
    if not isValid(s):
        return str(s)
    s = round(s)
    output = str(s % 60)
    if len(output) < 2:
        output = "0" + output
    if s >= 60:
        temp = str((s // 60) % 60)
        if len(temp) < 2 and s >= 3600:
            temp = "0" + temp
        output = temp + ":" + output
        if s >= 3600:
            temp = str((s // 3600) % 24)
            if len(temp) < 2 and s >= 86400:
                temp = "0" + temp
            output = temp + ":" + output
            if s >= 86400:
                output = str(s // 86400) + ":" + output
    else:
        output = "0:" + output
    return output


def rdhms(ts):
    data = ts.split(":")
    t = 0
    mult = 1
    while len(data):
        t += float(data[-1]) * mult
        data = data[:-1]
        if mult <= 60:
            mult *= 60
        elif mult <= 3600:
            mult *= 24
        elif len(data):
            raise TypeError("Too many time arguments.")
    return t


UNIFMTS = [
    "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙",
    "𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝓪𝓫𝓬𝓭𝓮𝓯𝓰𝓱𝓲𝓳𝓴𝓵𝓶𝓷𝓸𝓹𝓺𝓻𝓼𝓽𝓾𝓿𝔀𝔁𝔂𝔃𝓐𝓑𝓒𝓓𝓔𝓕𝓖𝓗𝓘𝓙𝓚𝓛𝓜𝓝𝓞𝓟𝓠𝓡𝓢𝓣𝓤𝓥𝓦𝓧𝓨𝓩",
    "𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝒶𝒷𝒸𝒹𝑒𝒻𝑔𝒽𝒾𝒿𝓀𝓁𝓂𝓃𝑜𝓅𝓆𝓇𝓈𝓉𝓊𝓋𝓌𝓍𝓎𝓏𝒜𝐵𝒞𝒟𝐸𝐹𝒢𝐻𝐼𝒥𝒦𝐿𝑀𝒩𝒪𝒫𝒬𝑅𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵",
    "𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡𝕒𝕓𝕔𝕕𝕖𝕗𝕘𝕙𝕚𝕛𝕜𝕝𝕞𝕟𝕠𝕡𝕢𝕣𝕤𝕥𝕦𝕧𝕨𝕩𝕪𝕫𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ",
    "0123456789𝔞𝔟𝔠𝔡𝔢𝔣𝔤𝔥𝔦𝔧𝔨𝔩𝔪𝔫𝔬𝔭𝔮𝔯𝔰𝔱𝔲𝔳𝔴𝔵𝔶𝔷𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ",
    "0123456789𝖆𝖇𝖈𝖉𝖊𝖋𝖌𝖍𝖎𝖏𝖐𝖑𝖒𝖓𝖔𝖕𝖖𝖗𝖘𝖙𝖚𝖛𝖜𝖝𝖞𝖟𝕬𝕭𝕮𝕯𝕰𝕱𝕲𝕳𝕴𝕵𝕶𝕷𝕸𝕹𝕺𝕻𝕼𝕽𝕾𝕿𝖀𝖁𝖂𝖃𝖄𝖅",
    "０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
    #"0123456789ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘQʀꜱᴛᴜᴠᴡxʏᴢᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘQʀꜱᴛᴜᴠᴡxʏᴢ",
    "0123456789🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉",
    "0123456789🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉",
    "⓪①②③④⑤⑥⑦⑧⑨ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ",
    "0123456789𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡",
    "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕",
    "𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉",
    "₀₁₂₃₄₅₆₇₈₉ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖqʳˢᵗᵘᵛʷˣʸᶻ🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿",
    "0123456789ᗩᗷᑢᕲᘿᖴᘜᕼᓰᒚҠᒪᘻᘉᓍᕵᕴᖇSᖶᑘᐺᘺ᙭ᖻᗱᗩᗷᑕᗪᗴᖴǤᕼIᒍKᒪᗰᑎOᑭᑫᖇᔕTᑌᐯᗯ᙭Yᘔ",
    "0ƖᘔƐᔭ59Ɫ86ɐqɔpǝɟɓɥᴉſʞןɯuodbɹsʇnʌʍxʎzꓯᗺƆᗡƎℲ⅁HIſꓘ⅂WNOԀΌᴚS⊥∩ΛMX⅄Z",
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
]
__map = {UNIFMTS[k][i]: UNIFMTS[-1][i] for k in range(len(UNIFMTS) - 1) for i in range(len(UNIFMTS[k]))}
for c in tuple(__map):
    if c in UNIFMTS[-1]:
        __map.pop(c)
__trans = "".maketrans(__map)
__unitrans = ["".maketrans({UNIFMTS[-1][x]: UNIFMTS[i][x] for x in range(len(UNIFMTS[-1]))}) for i in range(len(UNIFMTS) - 1)]

def uniStr(s, fmt=0):
    if type(s) is not str:
        s = str(s)
    return s.translate(__unitrans[fmt])

def reconstitute(s):
    if type(s) is not str:
        s = str(s)
    return s.translate(__trans)


class hlist(collections.abc.MutableSequence, collections.abc.Callable):

    """
custom list-like data structure that incorporates the functionality of np arrays but allocates more space on the ends in order to have faster insertion."""

    maxoff = (1 << 24) - 1
    minsize = 256
    __slots__ = ("chash", "block", "offs", "size", "data")

    def waiting(func):
        def call(self, *args, force=False, **kwargs):
            if not force:
                t = time.time()
                while self.block:
                    time.sleep(0.001)
                    if time.time() - t > 1:
                        raise TimeoutError("Request timed out.")
            return func(self, *args, **kwargs)
        return call

    def blocking(func):
        def call(self, *args, force=False, **kwargs):
            if not force:
                t = time.time()
                while self.block:
                    time.sleep(0.001)
                    if time.time() - t > 1:
                        raise TimeoutError("Request timed out.")
            self.block = True
            self.chash = None
            try:
                output = func(self, *args, **kwargs)
                self.block = False
            except:
                self.block = False
                raise
            return output
        return call

    def __init__(self, *args, **void):
        self.block = True
        if not args:
            iterable = ()
        elif len(args) == 1:
            iterable = args[0]
        else:
            iterable = args
        self.chash = None
        if issubclass(type(iterable), self.__class__) and iterable:
            self.offs = iterable.offs
            self.size = iterable.size
            self.data = iterable.data.copy()
        else:
            if not issubclass(type(iterable), collections.abc.Sequence):
                try:
                    iterable = deque(iterable)
                except TypeError:
                    iterable = [iterable]
            self.size = len(iterable)
            size = max(self.minsize, self.size * 3)
            self.offs = size // 3
            self.data = np.empty(size, dtype=object)
            self.view()[:] = iterable
        self.block = False

    view = lambda self: self.data[self.offs:self.offs + self.size]

    @waiting
    def __call__(self, arg=1, *void1, **void2):
        if arg == 1:
            return self.copy()
        return self * arg

    def __hash__(self):
        if self.chash is None:
            self.chash = hash(self.view().tobytes())
        return self.chash

    __str__ = lambda self: "⟨" + ", ".join(repr(i) for i in iter(self)) + "⟩"
    __repr__ = lambda self: "hlist(" + str(tuple(self)) + ")"
    __bool__ = lambda self: bool(self.size)

    @blocking
    def __iadd__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.add(arr, iterable, out=arr)
        return self

    @blocking
    def __isub__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.subtract(arr, iterable, out=arr)
        return self

    @blocking
    def __imul__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.multiply(arr, iterable, out=arr)
        return self

    @blocking
    def __imatmul__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        temp = np.matmul(arr, iterable)
        self.size = len(temp)
        arr[:self.size] = temp
        return self

    @blocking
    def __itruediv__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.true_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __ifloordiv__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.floor_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __imod__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.mod(arr, iterable, out=arr)
        return self

    @blocking
    def __ipow__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.power(arr, iterable, out=arr)
        return self

    @blocking
    def __ilshift__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        try:
            np.left_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.multiply(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __irshift__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        try:
            np.right_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.divide(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __iand__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.logical_and(arr, iterable, out=arr)
        return self

    @blocking
    def __ixor__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.logical_xor(arr, iterable, out=arr)
        return self

    @blocking
    def __ior__(self, other):
        iterable = self.createIterator(other)
        arr = self.view()
        np.logical_or(arr, iterable, out=arr)
        return self

    @waiting
    def __neg__(self):
        return self.__class__(-self.view())

    @waiting
    def __pos__(self):
        return self

    @waiting
    def __abs__(self):
        d = self.data
        return self.__class__(np.abs(self.view()))

    @waiting
    def __invert__(self):
        return self.__class__(np.invert(self.view()))

    @waiting
    def __add__(self, other):
        temp = self.copy()
        temp += other
        return temp

    @waiting
    def __sub__(self, other):
        temp = self.copy()
        temp -= other
        return temp

    @waiting
    def __mul__(self, other):
        temp = self.copy()
        temp *= other
        return temp

    @waiting
    def __matmul__(self, other):
        temp1 = self.view()
        temp2 = self.forceTuple(other)
        result = temp1 @ temp2
        return self.__class__(result)

    @waiting
    def __truediv__(self, other):
        temp = self.copy()
        temp /= other
        return temp

    @waiting
    def __floordiv__(self, other):
        temp = self.copy()
        temp //= other
        return temp

    @waiting
    def __mod__(self, other):
        temp = self.copy()
        temp %= other
        return temp

    @waiting
    def __pow__(self, other):
        temp = self.copy()
        temp **= other
        return temp

    @waiting
    def __lshift__(self, other):
        temp = self.copy()
        temp <<= other
        return temp

    @waiting
    def __rshift__(self, other):
        temp = self.copy()
        temp >>= other
        return temp

    @waiting
    def __and__(self, other):
        temp = self.copy()
        temp &= other
        return temp

    @waiting
    def __xor__(self, other):
        temp = self.copy()
        temp ^= other
        return temp

    @waiting
    def __or__(self, other):
        temp = self.copy()
        temp |= other
        return temp

    @waiting
    def __round__(self, prec=0):
        temp = np.round(self.view(), prec)
        return self.__class__(temp)

    @waiting
    def __trunc__(self):
        temp = np.trunc(self.view())
        return self.__class__(temp)

    @waiting
    def __floor__(self):
        temp = np.floor(self.view())
        return self.__class__(temp)

    @waiting
    def __ceil__(self):
        temp = np.ceil(self.view())
        return self.__class__(temp)

    __index__ = lambda self: self.view()
    __radd__ = __add__
    __rsub__ = lambda self, other: -self + other
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @waiting
    def __rtruediv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.createIterator(other)
        arr = temp.view()
        np.true_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rfloordiv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.createIterator(other)
        arr = temp.view()
        np.floor_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rmod__(self, other):
        temp = self.__class__(self.data)
        iterable = self.createIterator(other)
        arr = temp.view()
        np.mod(iterable, arr, out=arr)
        return temp

    @waiting
    def __rpow__(self, other):
        temp = self.__class__(self.data)
        iterable = self.createIterator(other)
        arr = temp.view()
        np.power(iterable, arr, out=arr)
        return temp

    @waiting
    def __rlshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.createIterator(other)
        arr = temp.view()
        try:
            np.left_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.multiply(iterable, np.power(2, arr), out=arr)
        return temp

    @waiting
    def __rrshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.createIterator(other)
        arr = temp.view()
        try:
            np.right_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.divide(iterable, np.power(2, arr), out=arr)
        return temp
    
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    @waiting
    def __lt__(self, other):
        it = self.createIterator(other)
        return self.view() < other

    @waiting
    def __le__(self, other):
        it = self.createIterator(other)
        return self.view() <= other

    @waiting
    def __eq__(self, other):
        try:
            it = self.createIterator(other)
            return self.view() == other
        except (TypeError, IndexError):
            return

    @waiting
    def __ne__(self, other):
        try:
            it = self.createIterator(other)
            return self.view() != other
        except (TypeError, IndexError):
            return

    @waiting
    def __gt__(self, other):
        it = self.createIterator(other)
        return self.view() > other

    @waiting
    def __ge__(self, other):
        it = self.createIterator(other)
        return self.view() >= other

    @waiting
    def __getitem__(self, *args):
        if len(args) == 1:
            key = args[0]
            if type(key) in (float, complex):
                return get(self.view(), key, 1)
            if type(key) is int:
                key = key % self.size
                return self.view().__getitem__(key)
            return self.__class__(self.view().__getitem__(key))
        return self.__class__(self.view().__getitem__(*args))

    @blocking
    def __setitem__(self, *args):
        if len(args) == 2:
            key = args[0]
            if type(key) is int:
                key = key % self.size
            return self.view().__setitem__(key, args[1])
        return self.view().__setitem__(*args)

    @blocking
    def __delitem__(self, key):
        if type(key) is slice:
            s = key.indices(self.size)
            return self.pops(xrange(*s))
        try:
            len(key)
        except TypeError:
            return self.pop(key, force=True)
        return self.pops(key)

    __len__ = lambda self: self.size
    __length_hint__ = __len__
    __iter__ = lambda self: iter(self.view())
    __reversed__ = lambda self: iter(np.flip(self.view()))

    @waiting
    def __bytes__(self):
        return bytes(round(i) & 255 for i in self.view())

    def __contains__(self, item):
        return item in self.view()

    __copy__ = lambda self: self.copy()

    def forceTuple(self, value):
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    def constantIterator(self, other):
        while True:
            yield other

    def createIterator(self, other, force=False):
        try:
            len(other)
        except TypeError:
            try:
                other = list(other)
            except TypeError:
                other = [other]
        if len(other) not in (1, self.size) and not force:
            raise IndexError(
                "Unable to perform operation on objects with size "
                + str(self.size) + " and " + str(len(other)) + "."
            )
        return other

    @blocking
    def clear(self):
        self.size = 0
        self.offs = self.size >> 1
        return self

    @waiting
    def copy(self):
        return self.__class__(self.view())

    @waiting
    def sort(self, *args, **kwargs):
        return self.__class__(sorted(self.view(), *args, **kwargs))

    @waiting
    def shuffle(self, *args, **kwargs):
        return self.__class__(shuffle(self.view(), *args, **kwargs))

    @waiting
    def reverse(self):
        return self.__class__(np.flip(self.view()))

    @blocking
    def rotate(self, steps):
        s = self.size
        if not s:
            return self
        steps %= s
        if steps > s >> 1:
            steps -= s
        if abs(steps) < self.minsize:
            while steps > 0:
                self.appendleft(self.popright(force=True), force=True)
                steps -= 1
            while steps < 0:
                self.appendright(self.popleft(force=True), force=True)
                steps += 1
            return self
        self.view()[:] = np.roll(self.view(), steps)
        return self

    @blocking
    def rotateleft(self, steps):
        return self.rotate(-steps, force=True)

    rotateright = rotate

    @blocking
    def isempty(self):
        if self.size:
            if abs(len(self.data) // 3 - self.offs) > self.maxoff:
                self.reconstitute(force=True)
            return False
        self.offs = self.size // 3
        return True

    @waiting
    def get(self, key, default=None):
        try:
            return self[key]
        except LookupError:
            return default

    @blocking
    def popleft(self):
        temp = self.data[self.offs]
        self.offs += 1
        self.size -= 1
        self.isempty(force=True)
        return temp

    @blocking
    def popright(self):
        temp = self.data[self.offs + self.size - 1]
        self.size -= 1
        self.isempty(force=True)
        return temp

    @blocking
    def pop(self, index=None):
        if index is None:
            return self.popright(force=True)
        if index >= len(self.data):
            return self.popright(force=True)
        elif index == 0:
            return self.popleft(force=True)
        index %= self.size
        temp = self.data[index + self.offs]
        if index > self.size >> 1:
            self.view()[index:-1] = self.data[self.offs + index + 1:self.offs + self.size]
        else:
            self.view()[1:index + 1] = self.data[self.offs:self.offs + index]
            self.offs += 1
        self.size -= 1
        return temp

    @blocking
    def insert(self, index, value):
        if index >= len(self.data):
            return self.append(value, force=True)
        elif index == 0:
            return self.appendleft(value, force=True)
        index %= self.size
        if index > self.size >> 1:
            if self.size + self.offs + 1 >= len(self.data):
                self.reconstitute(force=True)
            self.size += 1
            self.view()[index + 1:] = self.view()[index:-1]
        else:
            if self.offs < 1:
                self.reconstitute(force=True)
            self.size += 1
            self.offs -= 1
            self.view()[:index] = self.view()[1:index + 1]
        self.view()[index] = value
        return self

    @blocking
    def insort(self, value, key=None, sorted=True):
        if not sorted:
            self.__init__(sorted(self, key=key))
        if key is None:
            return self.insert(np.searchsorted(self.view(), value), value, force=True)
        v = key(value)
        x = self.size
        index = (x >> 1) + self.offs
        gap = 3 + x >> 2
        seen = {}
        d = self.data
        while index not in seen and index >= self.offs and index < self.offs + self.size:
            check = key(d[index])
            if check < v:
                seen[index] = True
                index += gap
            else:
                seen[index] = False
                index -= gap
            gap = 1 + gap >> 1
        index -= self.offs - seen.get(index, 0)
        if index <= 0:
            return self.appendleft(value, force=True)
        return self.insert(index, value, force=True)

    @blocking
    def remove(self, value, key=None, sorted=False):
        pops = self.search(value, key, sorted, force=True)
        if pops:
            self.pops(pops, force=True)
        return self

    @blocking
    def removedups(self):
        temp = np.unique(self.view())
        self.size = len(temp)
        self.view()[:] = temp
        return self

    uniq = unique = removedups

    @waiting
    def index(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[0]

    @waiting
    def rindex(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[-1]
    
    @waiting
    def search(self, value, key=None, sorted=False):
        if key is None:
            if sorted and self.size > self.minsize:
                i = np.searchsorted(self.view(), value)
                if self.view()[i] != value:
                    raise IndexError(str(value) + " not found.")
                pops = self.__class__()
                pops.append(i)
                for x in range(i + self.offs - 1, -1, -1):
                    if self.data[x] == value:
                        pops.appendleft(x - self.offs)
                    else:
                        break
                for x in range(i + self.offs + 1, self.offs + self.size):
                    if self.data[x] == value:
                        pops.append(x - self.offs)
                    else:
                        break
                return pops
            else:
                return self.__class__(np.arange(self.size, dtype=np.uint32)[self.view() == value])
        if sorted:
            v = value
            d = self.data
            pops = self.__class__()
            x = len(d)
            index = (x >> 1) + self.offs
            gap = x >> 2
            seen = {}
            while index not in seen and index >= self.offs and index < self.offs + self.size:
                check = key(d[index])
                if check < v:
                    seen[index] = True
                    index += gap
                elif check == v:
                    break
                else:
                    seen[index] = False
                    index -= gap
                gap = 1 + gap >> 1
            i = index + seen.get(index, 0)
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i += 1
            i = index + seen.get(index, 0) - 1
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i -= 1
        else:
            pops = self.__class__(i for i, x in enumerate(self.view()) if key(x) == value)
        if not pops:
            raise IndexError(str(value) + " not found.")
        return pops
    
    find = findall = search

    @waiting
    def count(self, value, key=None):
        if key is None:
            return len(self.view() == value)
        return sum(1 for i in self if key(i) == value)

    concat = lambda self, value: self.__class__(np.concatenate([self.view(), value]))

    @blocking
    def appendleft(self, value):
        if self.offs <= 0:
            self.reconstitute(force=True)
        self.offs -= 1
        self.size += 1
        self.data[self.offs] = value
        return self

    @blocking
    def append(self, value):
        if self.offs + self.size >= len(self.data):
            self.reconstitute(force=True)
        self.data[self.offs + self.size] = value
        self.size += 1
        return self

    appendright = append

    @blocking
    def extendleft(self, value):
        value = self.createIterator(reversed(value), force=True)
        if self.offs >= len(value):
            self.data[self.offs - len(value):self.offs] = value
            self.offs -= len(value)
            self.size += len(value)
            return self
        self.__init__(np.concatenate([value, self.view()]))
        return self

    @blocking
    def extend(self, value):
        value = self.createIterator(value, force=True)
        if len(self.data) - self.offs - self.size >= len(value):
            self.data[self.offs + self.size:self.offs + self.size + len(value)] = value
            self.size += len(value)
            return self
        self.__init__(np.concatenate([self.view(), value]))
        return self

    extendright = extend

    @blocking
    def fill(self, value):
        self.view()[:] = value

    keys = lambda self: range(len(self))
    values = lambda self: iter(self)
    items = lambda self: enumerate(self)

    @blocking
    def clip(self, a, b=None):
        if b is None:
            b = -a
        if a > b:
            a, b = b, a
        arr = self.view()
        np.clip(arr, a, b, out=arr)
        return self

    @waiting
    def real(self):
        return self.__class__(np.real(self.view()))

    @waiting
    def imag(self):
        return self.__class__(np.imag(self.view()))
    
    @waiting
    def float(self):
        return self.__class__(float(i.real) for i in self.view())

    @waiting
    def complex(self):
        return self.__class__(complex(i) for i in self.view())

    @waiting
    def mpf(self):
        return self.__class__(mpf(i.real) for i in self.view())

    @blocking
    def reconstitute(self, data=None):
        self.__init__(self.view())
        return self

    @blocking
    def delitems(self, iterable):
        iterable = self.createIterator(iterable, force=True)
        if len(iterable) == 1:
            return self.pop(iterable[0], force=True)
        temp = np.delete(self.view(), iterable)
        self.size = len(temp)
        self.view()[:] = temp
        return self

    pops = delitems

hrange = lambda a, b=None, c=None: hlist(xrange(a, b, c))

hzero = lambda size: hlist((0 for i in range(size)))


class cdict(dict):

    __slots__ = ()

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: self.__class__.__name__ + "(" + super().__repr__() + ")"
    __str__ = lambda self: "<" + super().__repr__()[1:-1] + ">"
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __setattr__ = lambda self, k, v: super().__setitem__(k, v)

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            return self.__class__.__getattribute__(self, k)
        return super().__getitem__(k)

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


class mdict(cdict):

    __slots__ = ()

    count = lambda self: sum(len(v) for v in super().values())
    extend = lambda self, k, v: setDict(super(), k, hlist()).extend(v).removedups()

    def append(self, k, v):
        values = setDict(super(), k, hlist())
        if v not in values:
            values.append(v)

    def popleft(self, k):
        values = super().__getitem__(k)
        if len(values):
            v = values.popleft()
        else:
            v = None
        if not values:
            super().pop(k)
        return v

    def __init__(self, *args, **kwargs):
        super().__init__()
        for it in args:
            for k, v in it.items():
                self.extend(k, v)
        for k, v in kwargs:
            self.extend(k, v)


class demap(collections.abc.Mapping):

    __slots__ = ("a", "b")

    def __init__(self, *args, **kwargs):
        self.a = cdict(*args, **kwargs)
        self.b = cdict(reversed(t) for t in self.a.items())

    def __getitem__(self, k):
        try:
            return self.a.__getitem__(k)
        except KeyError:
            return self.b.__getitem__(k)

    def __delitem__(self, k):
        try:
            temp = self.a.pop(k)
        except KeyError:
            temp = self.b.pop(k)
            if temp in self.a:
                self.__delitem__(temp)
        else:
            if temp in self.b:
                self.__delitem__(temp)
        return self

    def __setitem__(self, k, v):
        if k not in self.a:
            if v not in self.a:
                self.a.__setitem__(k, v)
                self.b.__setitem__(v, k)
            else:
                self.__delitem__(v)
                self.__setitem__(k, v)
        else:
            self.__delitem__(k)
            if v in self.a:
                self.__delitem__(v)
            self.__setitem__(k, v)
        return self

    def get(self, k, v=None):
        try:
            return self.__getitem__(k)
        except KeyError:
            return v

    clear = lambda self: (self.a.clear(), self.b.clear())
    __iter__ = lambda self: iter(self.a.items())
    __reversed__ = lambda self: reversed(self.a.items())
    __len__ = lambda self: self.b.__len__()
    __str__ = lambda self: self.a.__str__()
    __repr__ = lambda self: self.__class__.__name__ + "(" + self.a.___repr__() + ")"
    __contains__ = lambda self, k: k in self.a or k in self.b
    pop = __delitem__


def bytes2Hex(b, space=True):
    if type(b) is str:
        b = b.encode("utf-8")
    if space:
        return b.hex(" ").upper()
    return b.hex().upper()

hex2Bytes = lambda b: bytes.fromhex(b if type(b) is str else b.decode("utf-8", "replace"))

def bytes2B64(b, alt_char_set=False):
    if type(b) is str:
        b = b.encode("utf-8")
    b = base64.b64encode(b)
    if alt_char_set:
        b = b.replace(b"=", b"-").replace(b"/", b".")
    return b

def b642Bytes(b, alt_char_set=False):
    if type(b) is str:
        b = b.encode("utf-8")
    if alt_char_set:
        b = b.replace(b"-", b"=").replace(b".", b"/")
    b = base64.b64decode(b)
    return b

zeroEnc = "\xad\u061c\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\u2065\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\ufeff\x0c"
zeroEncoder = demap({chr(i + 97): c for i, c in enumerate(zeroEnc)})
zeroEncode = "".maketrans(dict(zeroEncoder.a))
zeroDecode = "".maketrans(dict(zeroEncoder.b))
isZeroEnc = lambda s: (s[0] in zeroEnc) if s else None
zwencode = lambda s: (s if type(s) is str else str(s)).lower().translate(zeroEncode)
zwdecode = lambda s: (s if type(s) is str else str(s)).lower().translate(zeroDecode)


shash = lambda s: base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).replace(b"/", b"-").decode("utf-8", "replace")
hhash = lambda s: bytes2Hex(hashlib.sha256(s.encode("utf-8")).digest(), space=False)


class pickled(collections.abc.Callable):

    def __init__(self, obj=None, ignore=()):
        self.data = obj
        self.ignores = {}
        self.__str__ = obj.__str__
        self.__dict__.update(getattr(obj, "__dict__", {}))

    def __call__(self):
        return self

    def ignore(self, item):
        self.ignores[item] = True

    def __repr__(self):
        c = dict(self.data)
        for i in self.ignores:
            c.pop(i)
        d = pickle.dumps(c)
        if len(d) > 1048576:
            return "None"
        return (
            "pickled(pickle.loads(hex2Bytes('''"
            + bytes2Hex(d, space=False)
            + "''')))"
        )