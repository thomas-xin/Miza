"""
Adds many useful math-related functions.
"""

import os, sys, asyncio, threading, subprocess, psutil, traceback, time
import ctypes, collections, ast, copy, pickle
import random, math, cmath, fractions, mpmath, sympy, shlex, numpy, colorsys

import urllib.request

from scipy import interpolate, special, signal
from sympy.parsing.sympy_parser import parse_expr

CalledProcessError = subprocess.CalledProcessError
Process = psutil.Process()

np = numpy
array = numpy.array
deque = collections.deque

random.seed(random.random() + time.time() % 1)
mp = mpmath.mp
mp.dps = 64

math.round = round

mpf = mpmath.mpf
mpc = mpmath.mpc
Mat = mat = matrix = mpmath.matrix

inf = math.inf
nan = math.nan
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


def nop(*args):
    pass


class freeClass:
    
    def __init__(self, **kwargs):
        for i in kwargs:
            self.__setattr__(i, kwargs[i])


def shuffle(it):
    if type(it) is list:
        random.shuffle(it)
        return it
    elif type(it) is tuple:
        it = list(it)
        random.shuffle(it)
        return it
    elif type(it) is dict:
        ir = sorted(it, key=lambda x: random.random())
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
        del temp
        return it
    else:
        try:
            it = list(it)
            random.shuffle(it)
            return it
        except TypeError:
            raise TypeError("Shuffling " + type(it) + " is not supported.")

def sort(it, key=lambda x: x, reverse=False):
    if type(it) is list:
        it.sort(key=key, reverse=reverse)
        return it
    elif type(it) is tuple:
        it = sorted(it, key=key, reverse=reverse)
        return it
    elif type(it) is dict:
        ir = sorted(it, key=key, reverse=reverse)
        new = {}
        for i in ir:
            new[i] = it[i]
        it.clear()
        it.update(new)
        return it
    elif type(it) is deque:
        it = sorted(it, key=key, reverse=reverse)
        return deque(it)
    elif isinstance(it, hlist):
        hlist(sorted(it, key=key, reverse=reverse))
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


def sqr(x):
    return ((sin(x) >= 0) << 1) - 1

def saw(x):
    return (x / pi + 1) % 2 - 1

def tri(x):
    return (abs((0.5 - x / pi) % 2 - 1)) * 2 - 1

def sgn(x):
    return (((x > 0) << 1) - 1) * (x != 0)


def frand(x=1, y=0):
    return (random.random() * max(x, y) / mpf(random.random())) % x + y

def xrand(x, y=None, z=0):
    if y == None:
        y = 0
    if x == y:
        return x
    return random.randint(floor(min(x, y)), ceil(max(x, y)) - 1) + z

def rrand(x=1, y=0):
    return frand(x) ** (1 - y)


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


def getFactors(x):
    f = factorize(x)
    f.append(1)
    s = {}
    l = len(f)
    print(s)


def addDict(a, b, replace=True):
    if replace:
        r = a
    else:
        r = dict(a)
    for k in b:
        r[k] = b[k] + a.get(k, 0)
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
            y = x
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


def mean(*nums):
    return roundMin(numpy.mean(numpy.array(nums)))


def pwr(x, power=2):
    if number.real >= 0:
        return roundMin(number ** power)
    else:
        return roundMin(-((-number) ** power))


def pulse(x, y=0.5):
    p = y * tau
    x = x * (0.5 / length * (x < p) + 0.5 / (1 - length) * (x >= p))
    return x


def hypot(*coordinates):
    return math.hypot(*coordinates)


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
        return "".join([str(c) for c in string])
    else:
        return str(string)


def bytes2Hex(b):
    o = ""
    for a in b:
        c = hex(a).upper()[2:]
        if len(c) < 2:
            c = "0" + c
        o += c + " "
    return o[:-1]

def hex2Bytes(h):
    o = []
    h = h.replace(" ", "").replace("\r", "").replace("\n", "")
    for a in range(0, len(h), 2):
        o.append(int(h[a : a + 2], 16))
    return bytes(o)


def colourCalculation(a, offset=0):
    return adjColour(colorsys.hsv_to_rgb((a / 1536) % 1, 1, 1), offset, 255)

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

def hex2Colour(h):
    return verifyColour(hex2Bytes(h))

def luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

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

def invColour(c):
    return [255 - i for i in c]

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
        resized = numpy.array([v[round(i / new * size) % size] for i in range(new)])
    elif mode <= 5 and mode == int(mode):
        spl = interpolate.splrep(numpy.arange(1 + size), numpy.append(v, v[0]), k=int(min(size, mode)))
        resized = numpy.array([interpolate.splev((i / new * size) % size, spl) for i in range(new)])
    elif mode <= 5:
        if math.floor(mode) == 0:
            resized1 = resizeVector(v, new, 0)
        else:
            spl1 = interpolate.splrep(numpy.arange(1 + size), numpy.append(v, v[0]), k=floor(min(size, mode)))
            resized1 = numpy.array([interpolate.splev((i / new * size) % size, spl1) for i in range(new)])
        spl2 = interpolate.splrep(numpy.arange(1 + size), numpy.append(v, v[0]), k=ceil(min(size, mode)))
        resized2 = numpy.array([interpolate.splev((i / new * size) % size, spl2) for i in range(new)])
        resized = resized1 * (1 - mode % 1) + (mode % 1) * resized2
    else:
        resizing = []
        for i in range(1, floor(mode)):
            resizing.append(resizeVector(v, new, i / floor(mode) * 5))
        resized = numpy.mean(resizing, 0)
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
        return roundMin(interpolate.splev(i, interpolate.splrep(numpy.arange(1 + size), numpy.append(v, v[0]), k=int(min(size, mode)))))
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


def randomPolarCoord(x=1):
    return polarCoords(frand(x), frand(tau))

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
    return array([hypot(d), atan2(*reversed(d))])


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
    for i in range(4):
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
    for x in range(0, 64):
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
    array = function(numpy.arange(0, period, 1 / (size + 1) * period))
    return array

def array2Harmonics(data, precision=1024):
    output = []
    T = len(data)
    t = numpy.arange(T)
    for n in range(precision + 1):
        if n > T / 2 + 1:
            output.append(numpy.array((0, 0)))
        else:
            bn = 2 / T * (data * numpy.cos(2 * pi * n * t / T)).sum()
            an = 2 / T * (data * numpy.sin(2 * pi * n * t / T)).sum()
            R = numpy.sqrt(an ** 2 + bn ** 2)
            p = numpy.arctan2(bn, an)
            if R == 0:
                p = 0
            output.append(numpy.array((R, p)))
    return numpy.array(output[1 : precision + 1])

def harmonics2Array(period, harmonics, func="sin(x)"):
    expression = func
    function = eval("lambda x: " + expression)
    result = 0
    t = numpy.arange(period)
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


def htmlDecode(s):
    while len(s) > 7:
        try:
            i = s.index("&#")
        except ValueError:
            break
        try:
            if s[i + 2] == "x":
                h = "0x"
                p = i + 3
            else:
                h = ""
                p = i + 2
            for a in range(4):
                if s[p + a] == ";":
                    v = int(h + s[p:p + a])
                    break
            c = chr(v)
            s = s[:i] + c + s[p + a + 1:]
        except ValueError:
            continue
        except IndexError:
            continue
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return s.replace("&quot;", '"').replace("&apos;", "'")


__units = {
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
    for i in __units:
        a = None
        t = m = __units[i]
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

def sec2Time(s):
    return " ".join(timeConv(s))

def dhms(s):
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


def noHighlight(s):
    s = str(s).replace("[", "⦍").replace("]", "⦎")
    s = s.replace("@", "＠")
    return s

__fmts = [
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
    "0123456789𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕",
    "𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉",
    "0123456789ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖqʳˢᵗᵘᵛʷˣʸᶻ🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿",
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
]
__map = {__fmts[k][i]: __fmts[-1][i] for k in range(len(__fmts) - 1) for i in range(len(__fmts[k]))}
__trans = "".maketrans(__map)

def uniStr(s, fmt=0):
    if type(s) is not str:
        s = str(s)
    for i in range(len(__fmts[-1])):
        s = s.replace(__fmts[-1][i], __fmts[fmt][i])
    return s

def reconstitute(s):
    return s.translate(__trans)


__hlist_maxoff__ = (1 << 31) - 1

class hlist(collections.abc.MutableSequence):

    """
custom list-like data structure that incorporates the functionality of dicts in \
order to have average O(1) constant time insertion on both sides as well as O(1) \
lookup time for all elements. Includes many array and numeric operations."""

    def waiting(func):
        def call(self, *args, force=False, **kwargs):
            if not force:
                while self.block:
                    time.sleep(0.00001)
            return func(self, *args, **kwargs)
        return call

    def blocking(func):
        def call(self, *args, force=False, **kwargs):
            if not force:
                while self.block:
                    time.sleep(0.00001)
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

    def forceTuple(self, value):
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    def iterator(self, reverse=False):
        if reverse:
            r = xrange(len(self.data) - 1, -1)
        else:
            r = range(len(self.data))
        for i in r:
            if not i + self.offs in self.data:
                break
            yield self.data[self.offs + i]
        return

    def constantIterator(self, other):
        while True:
            yield other

    def createIterator(self, other, force=False):
        d = self.data
        try:
            iterable = iter(other)
            if len(other) != len(d) and not force:
                raise IndexError(
                    "Unable to perform operation on objects with size "
                    + str(len(d)) + " and " + str(len(other)) + "."
                )
            return iterable
        except TypeError:
            return self.constantIterator(other)

    def deleteobj(self, item):
        del item

    @blocking
    def clear(self):
        temp = self.data
        self.data = {}
        self.offs = 0
        doParallel(self.deleteobj, [temp], name="deleter", killable=False)
        return self

    @waiting
    def copy(self):
        return hlist(self)

    @waiting
    def sort(self):
        return hlist(sorted(self))

    @waiting
    def shuffle(self):
        temp = list(self)
        return hlist(shuffle(temp))

    @blocking
    def rotate(self, steps):
        s = len(self.data)
        steps = -steps % s
        if steps > s / 2:
            steps -= s
        if steps < 0:
            for i in xrange(steps):
                self.appendleft(self.popright(force=True), force=True)
        else:
            for i in range(steps):
                self.append(self.popleft(force=True), force=True)
        return self

    @blocking
    def rotateleft(self, steps):
        return self.rotate(-steps, force=True)

    rotateright = rotate

    @blocking
    def isempty(self):
        s = len(self.data)
        if s:
            if abs(self.offs) > self.maxoff:
                self.reconstitute(force=True)
            elif s == 1 and self.offs:
                temp = self.data
                self.data = {0: self.data[self.offs]}
                self.offs = 0
                doParallel(self.deleteobj, [temp], killable=False)
            return False
        self.offs = 0
        return True

    @blocking
    def popleft(self):
        key = self.offs
        temp = self.data.pop(key)
        self.offs += 1
        self.isempty(force=True)
        return temp

    @blocking
    def popright(self):
        key = self.offs + len(self.data) - 1
        temp = self.data[key]
        self.data.pop(key)
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
        index %= len(self.data)
        temp = self.data[index + self.offs]
        if index < len(self.data) / 2:
            for i in range(index + self.offs, self.offs, -1):
                self.data[i] = self.data[i - 1]
            self.popleft(force=True)
        else:
            for i in range(index + self.offs, self.offs + len(self.data) - 1):
                self.data[i] = self.data[i + 1]
            self.popright(force=True)
        self.isempty(force=True)
        return temp

    @blocking
    def insert(self, index, value):
        if index >= len(self.data):
            return self.append(value, force=True)
        elif index == 0:
            return self.appendleft(value, force=True)
        index %= len(self.data)
        if index < len(self.data) / 2:
            for i in range(self.offs, self.offs + index):
                self.data[i - 1] = self.data[i]
            self.offs -= 1
            self.data[index + self.offs] = value
        else:
            for i in range(self.offs + len(self.data) - 1, index + self.offs - 1, - 1):
                self.data[i + 1] = self.data[i]
            self.data[index + self.offs] = value
        return self

    @blocking
    def remove(self, value):
        for i in range(self.offs, self.offs + len(self.data)):
            if self.data[i] == value:
                self.pop(i, force=True)
                self.isempty(force=True)
                return self
        raise IndexError(str(value) + " not found.")

    @waiting
    def index(self, value):
        for i in self:
            if i == value:
                return i
        raise IndexError(str(value) + " not found.")

    @waiting
    def search(self, value):
        output = hlist()
        for i in self:
            if i == value:
                output.append(i, force=True)
        return output

    @waiting
    def count(self, value):
        output = 0
        for i in self:
            if i == value:
                output += 1
        return output

    @waiting
    def concat(self, value):
        temp = self.copy()
        temp.extend(value, force=True)
        return temp

    @blocking
    def appendleft(self, value):
        self.offs -= 1
        self.data[self.offs] = value
        return self

    @blocking
    def append(self, value):
        self.data[self.offs + len(self.data)] = value
        return self

    appendright = append

    @blocking
    def extendleft(self, value):
        value = reversed(self.forceTuple(value))
        for i in value:
            self.appendleft(i, force=True)
        return self

    @blocking
    def extend(self, value):
        value = self.forceTuple(value)
        for i in value:
            self.append(i, force=True)
        return self

    extendright = extend

    @blocking
    def fill(self, value):
        data = (value,) * len(self.data)
        self.__init__(data)

    @blocking
    def clip(self, a, b=None):
        if b is None:
            b = -a
        a, b = sorted(a, b)
        d = self.data
        for i in d:
            if d[i] < a:
                d[i] = a
            elif d[i] > b:
                d[i] = b
        return self

    @waiting
    def real(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = d[i].real
        return temp

    @waiting
    def imag(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = d[i].imag
        return temp

    @waiting
    def float(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = float(d[i])
        return temp

    @waiting
    def complex(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = complex(d[i])
        return temp

    @waiting
    def mpf(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = mpf(d[i])
        return temp

    @blocking
    def reconstitute(self, data=None):
        if data is None:
            data = self.data
        values = deque()
        l = sorted(data)
        for i in l:
            values.append(data.pop(i))
        self.__init__(values)

    @blocking
    def delitems(self, iterable):
        popped = False
        for i in iterable:
            self.data.pop(i + self.offs)
            popped = True
        if popped:
            self.reconstitute(force=True)
        return self

    pops = delitems

    def __init__(self, iterable=(), maxoff=__hlist_maxoff__, **void):
        self.chash = None
        self.block = True
        self.maxoff = maxoff
        if isinstance(iterable, hlist) and len(iterable):
            self.offs = iterable.offs
            self.data = iterable.data.copy()
        else:
            self.offs = 0
            d = self.data = {}
            try:
                iterable = iter(iterable)
                i = 0
                while True:
                    try:
                        d[i] = next(iterable)
                        i += 1
                    except StopIteration:
                        break
            except TypeError:
                d[0] = iterable
        self.block = False

    def __delattr__(self, name, value):
        raise AttributeError("Deleting attributes is not permitted.")

    @waiting
    def __call__(self, arg=1, **void):
        if arg == 1:
            return self.copy()
        return self * arg

    def __hash__(self):
        if self.chash is None:
            self.chash = hash(tuple(self))
        return self.chash

    def __str__(self):
        return "⟨" + ", ".join(str(i) for i in iter(self)) + "⟩"

    def __repr__(self):
        return "hlist(" + str(tuple(self)) + ")"

    def __bool__(self):
        return bool(len(self.data))

    @blocking
    def __iadd__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] += next(iterable)
        return self

    @blocking
    def __isub__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] -= next(iterable)
        return self

    @blocking
    def __imul__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] *= next(iterable)
        return self

    @blocking
    def __imatmul__(self, other):
        temp1 = numpy.array(tuple(self))
        temp2 = numpy.array(self.forceTuple(other))
        result = temp1 @ temp2
        self.__init__(result)
        return self

    @blocking
    def __itruediv__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            try:
                d[i] /= next(iterable)
            except ZeroDivisionError:
                d[i] = inf * sgn(d[i])
        return self

    @blocking
    def __ifloordiv__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            try:
                d[i] //= next(iterable)
            except ZeroDivisionError:
                d[i] = 0
        return self

    @blocking
    def __imod__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            try:
                d[i] %= next(iterable)
            except ZeroDivisionError:
                d[i] = 0
        return self

    @blocking
    def __ipow__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] **= next(iterable)
        return self

    @blocking
    def __ilshift__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] <<= r
            except ValueError:
                d[i] >>= -r
            except TypeError:
                d[i] *= 2 ** r
        return self

    @blocking
    def __irshift__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] >>= r
            except ValueError:
                d[i] <<= -r
            except TypeError:
                d[i] //= 2 ** r
        return self

    @blocking
    def __iand__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] &= r
            except TypeError:
                d[i] = int(d[i]) & int(r)
        return self

    @blocking
    def __ixor__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] ^= r
            except TypeError:
                d[i] = int(d[i]) ^ int(r)
        return self

    @blocking
    def __ior__(self, other):
        d = self.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] |= r
            except TypeError:
                d[i] = int(d[i]) | int(r)
        return self

    @waiting
    def __neg__(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = -d[i]
        return temp

    @waiting
    def __pos__(self):
        return self

    @waiting
    def __abs__(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = abs(d[i])
        return temp

    @waiting
    def __invert__(self):
        temp = self.copy()
        d = temp.data
        for i in d:
            d[i] = ~d[i]
        return temp

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
        temp1 = numpy.array(tuple(self))
        temp2 = numpy.array(self.forceTuple(other))
        result = temp1 @ temp2
        return hlist(result)

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
        temp = numpy.array(tuple(self))
        temp = numpy.round(temp, prec)
        if prec <= 0:
            temp = temp.astype(int)
        return hlist(temp)

    @waiting
    def __trunc__(self):
        temp = numpy.array(tuple(self))
        return hlist(numpy.trunc(temp).astype(int))

    @waiting
    def __floor__(self):
        temp = numpy.array(tuple(self))
        return hlist(numpy.floor(temp).astype(int))

    @waiting
    def __ceil__(self):
        temp = numpy.array(tuple(self))
        return hlist(numpy.ceil(temp).astype(int))

    def __index__(self):
        return round(numpy.sum(tuple(self)))
    
    __radd__ = __add__

    def __rsub__(self, other):
        return -self + other
    
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @waiting
    def __rtruediv__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] = r / d[i]
            except ZeroDivisionError:
                d[i] = inf * sgn(r)
        return temp

    @waiting
    def __rfloordiv__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            try:
                d[i] = next(iterable) // d[i]
            except ZeroDivisionError:
                d[i] = 0
        return temp

    @waiting
    def __rmod__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            try:
                d[i] = next(iterable) % d[i]
            except ZeroDivisionError:
                d[i] = 0
        return temp

    @waiting
    def __rpow__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = next(iterable) ** d[i]
        return temp

    @waiting
    def __rlshift__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] = r << d[i]
            except ValueError:
                d[i] = r >> -d[i]
            except TypeError:
                d[i] = r * 2 ** d[i]
        return temp

    @waiting
    def __rrshift__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            r = next(iterable)
            try:
                d[i] = r >> d[i]
            except ValueError:
                d[i] = r << -d[i]
            except TypeError:
                d[i] = r // 2 ** d[i]
        return temp
    
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    @waiting
    def __lt__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = d[i] < next(iterable)
        return temp

    @waiting
    def __le__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = d[i] <= next(iterable)
        return temp

    @waiting
    def __eq__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = d[i] == next(iterable)
        return temp

    @waiting
    def __ne__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = d[i] != next(iterable)
        return temp

    @waiting
    def __gt__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = d[i] > next(iterable)
        return temp

    @waiting
    def __ge__(self, other):
        temp = self.copy()
        d = temp.data
        iterable = self.createIterator(other)
        for i in d:
            d[i] = d[i] >= next(iterable)
        return temp

    @waiting
    def __getitem__(self, key):
        if type(key) is slice:
            temp = hlist()
            s = key.indices(len(self.data))
            for i in xrange(*s):
                temp.append(self.data[i + self.offs], force=True)
            return temp
        elif type(key) is not int:
            key = complex(key)
            return get(self, key, 1)
        index = self.offs + key % len(self.data)
        return self.data[index]

    @blocking
    def __setitem__(self, key, value):
        if type(key) is slice:
            s = key.indices(len(self.data))
            iterable = self.createIterator(value, True)
            for i in xrange(*s):
                self.data[i + self.offs] = next(iterable)
            return value
        elif type(key) is str:
            key = int(key)
        index = self.offs + key % len(self.data)
        self.data[index] = value
        return value

    @blocking
    def __delitem__(self, key):
        if type(key) is slice:
            temp = hlist()
            s = key.indices(len(self.data))
            for i in xrange(*s):
                temp.append(self.data[i + self.offs], force=True)
                self.data.pop(i + self.offs)
            self.reconstitute(force=True)
            return temp
        self.pop(key, force=True)

    def __len__(self):
        return len(self.data)

    __length_hint__ = __len__

    def __iter__(self):
        return self.iterator()

    def __reversed__(self):
        return self.iterator(True)

    @waiting
    def __bytes__(self):
        temp = bytes()
        for i in self:
            temp += bytes((round(i) & 255,))
        return temp

    def __contains__(self, item):
        for i in self:
            if i == item:
                return True
        return False

    def __copy__(self):
        return self.copy()

def hrange(a, b=None, c=None, maxoff=__hlist_maxoff__):
    return hlist(xrange(a, b, c), maxoff)

def hzero(size, maxoff=__hlist_maxoff__):
    return hlist((0 for i in range(size)), maxoff)


class pickled:

    def __init__(self, obj=None, ignore=()):
        self.data = obj
        self.ignores = {}
        self.__str__ = obj.__str__
        self.__dict__.update(getattr(obj, "__dict__", {}))

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
            + bytes2Hex(d).replace(" ", "")
            + "''')))"
        )


def readline(stream, timeout=10):
    output = bytes()
    t = time.time()
    while b"\n" not in output and time.time() - t < timeout:
        c = stream.read(1)
        if c:
            output += c
        else:
            time.sleep(0.002)
    print(output)
    return output
    

__subs__ = {}

def subCount():
    count = 0
    for i in list(__subs__):
        if __subs__[i].is_running():
            count += 1
        else:
            __subs__.pop(i)
    return count

def subKill():
    for sub in __subs__.values():
        sub.kill()
    __subs__.clear()

def subFunc(key, com, data_in, timeout):    
    if key in __subs__:
        try:
            while __subs__[key].busy:
                time.sleep(0.01)
        except KeyError:
            return subFunc(key, com, data_in, timeout)
    else:
        __subs__[key] = freeClass(
            busy=True,
            is_running=lambda: True
        )
    if isinstance(__subs__[key], psutil.Popen):
        proc = __subs__[key]
        if not proc.is_running():
            __subs__.pop(key)
            del proc
            return subFunc(key, com, data_in, timeout)
    else:
        proc = __subs__[key] = psutil.Popen(
            com,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    thread = freeClass(kill=lambda: None)
    try:
        t = time.time()
        proc.busy = True
        d = repr(bytes(str(data_in), "utf-8")).encode("utf-8") + b"\n"
        print(d)
        proc.stdin.write(d)
        proc.stdin.flush()
        returns = [None]
        thread = doParallel(readline, [proc.stdout, timeout], returns, state=2)
        while returns[0] is None:
            if time.time() - t > timeout:
                raise TimeoutError("Request timed out.")
            time.sleep(0.001)
        resp = eval(returns[0])
        print(resp)
        if issubclass(resp.__class__, Exception):
            raise resp
        resp = eval(resp.decode("utf-8"))
        if issubclass(resp.__class__, Exception):
            raise resp
        resp = str(resp)
        output = [resp]
    except TimeoutError as ex:
        print(traceback.format_exc())
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        __subs__.pop(key)
        output = repr(ex)
    except Exception as ex:
        print(traceback.format_exc())
        output = repr(ex)
    proc.busy = False
    thread.kill()
    return output


class dynamicFunc:
    
    def __init__(self, func):
        self.text = func
        self.func = eval(func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.text

def performAction(action):
    if "delay" in action:
        time.sleep(action["delay"])
    action.get("retn", [0])[0] = action["func"](*action.get("args", ()), **action.get("kwargs", {}))

class _parallel:
    
    def __init__(self):
        self.max = 64
        self.running = {i: self.new(i) for i in range(self.max)}
        for i in self.running:
            self.running[i].start()

    class new(threading.Thread):
        
        def __init__(self, p_id, killable=True):
            threading.Thread.__init__(self)
            self.killable = killable
            self.id = p_id
            self.actions = hlist()
            self.state = 0
            self.action = None
            self.daemon = True

        def __call__(self, action):
            self.actions.append(action)
            self.state = inf

        def run(self):
            while True:
                try:
                    if self.actions:
                        self.state = max(i.get("state", 1) for i in self.actions)
                    time.sleep(0.009 * (random.random() + 1))
                    if self.actions is None:
                        print("EXIT")
                        return
                    while self.actions:
                        self.action = self.actions.popleft()
                        performAction(self.action)
                except TimeoutError:
                    pass
                except:
                    print(traceback.format_exc())
                if type(self.id) is str:
                    print("EXIT")
                    break
                self.state = -1

        def get_id(self):
            if hasattr(self, "_thread_id"):
                return self._thread_id
            for t_id, thread in threading._active.items():
                if thread is self:
                    self._thread_id = t_id
                    return t_id

        def kill(self, destroy=False):
            thread_id = self.get_id()
            if destroy or type(self.id) is str:
                self.actions = None
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    thread_id,
                    ctypes.py_object(KeyboardInterrupt),
                )
            else:
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    thread_id,
                    ctypes.py_object(TimeoutError),
                )
            if res != 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    thread_id,
                    ctypes.py_object(BaseException),
                )
                self.actions = None
                try:
                    del threading._active[thread_id]
                except KeyError:
                    pass
                if not destroy and type(self.id) is not str:
                    processes.running[self.id] = processes.new(self.id)
            elif type(self.id) is str:
                self.actions = None
                try:
                    del threading._active[thread_id]
                except KeyError:
                    pass

def doParallel(func, args=None, data_out=[0], kwargs=None, name=None, **kws):
    """
Performs an action using parallel threads."""
    ps = threads.running
    if name is not None:
        d = str(name)
        ps[d] = threads.new(d, kws.get("killable", True))
        p = ps[d]
        p.start()
    else:
        t = d = 0
        p = ps[0]
        while p.state > 0:
            d = xrand(threads.max)
            p = ps[d]
            if t > threads.max:
                break
            t += 1
        while p.state > 1 or len(p.actions) >= 64:
            time.sleep(0.005)
            d = xrand(threads.max)
            p = ps[d]
    action = kws
    if args is not None:
        action["args"] = args
    if kwargs is not None:
        action["kwargs"] = kwargs
    action["func"] = func
    action["retn"] = data_out
    p(action)
    return p

def killThreads():
    running = tuple(threads.running)
    for i in running:
        if threads.running[i].killable:
            p = threads.running[i]
            p.kill()

def waitParallel(delay):
    t = time.time()
    running = tuple(threads.running)
    for i in running:
        if type(i) is int and i in threads.running:
            p = threads.running[i]
            while p.state > 0 and time.time() - t < delay:
                time.sleep(0.001)

threads = _parallel()


def getLineCount(fn):
    f = open(fn, "rb")
    count = 1
    size = 0
    while True:
        try:
            i = f.read(1024)
            if not i:
                raise EOFError
            size += len(i)
            count += i.count(b"\n")
        except EOFError:
            f.close()
            return hlist((size, count))


def iscode(fn):
    fn = str(fn)
    return fn.endswith(".py") or fn.endswith(".pyw")# or fn.endswith(".c") or fn.endswith(".cpp")

__umap = {
    "<": "",
    ">": "",
    "|": "",
    "*": "",
    " ": "%20",
}
__utrans = "".maketrans(__umap)

def verifyURL(f):
    if "file:" in f:
        raise PermissionError("Unable to open local file " + f + ".")
    return f.strip(" ").translate(__utrans)

class urlBypass(urllib.request.FancyURLopener):
    version = "Mozilla/5." + str(xrand(1, 10))

def urlOpen(url):
    opener = urlBypass()
    resp = opener.open(verifyURL(url))
    if resp.getcode() != 200:
        raise ConnectionError("Error " + str(resp.code))
    return resp
    s = resp.read().decode("utf-8")


def logClear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

class __logPrinter():

    print_temp = ""
    
    def updatePrint(self, file):
        if file is None:
            outfunc = sys.stdout.write
            enc = lambda x: str(x)
        else:
            def filePrint(b):
                f = open(file, "ab+")
                f.write(b)
                f.close()
            outfunc = filePrint
            enc = lambda x: bytes(str(x), "utf-8")
        outfunc(enc("Logging started...\n"))
        while True:
            if self.print_temp:
                if len(self.print_temp) > 4096 or self.print_temp.count("\n") > 48:
                    self.print_temp = limStr(self.print_temp, 2048)
                data = enc(self.print_temp)
                #sys.stdout.write(repr(data))
                outfunc(data)
                self.print_temp = ""
            time.sleep(1)
            #sys.stdout.write(str(f))

    def logPrint(self, *args, sep=" ", end="\n", prefix="", **void):
        self.print_temp += str(sep).join((str(i) for i in args)) + str(end) + str(prefix)

    def __init__(self, file=None):
        doParallel(self.updatePrint, [file], name="printer", killable=False)

__printer = __logPrinter("log.txt")
print = __printer.logPrint
