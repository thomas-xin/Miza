"""
Adds many useful math-related functions.
"""
import math, cmath, fractions, mpmath, sympy, ctypes
import numpy, tinyarray

array = tinyarray.array
import colorsys, random, threading, time
from scipy import interpolate, special

random.seed(random.random() + time.time() % 1)
sympy.init_printing(use_unicode=True)

mp = mpmath.mp
mp.dps = 128

math.round = round
c_ = 299792458
inf = math.inf
nan = math.nan

mpf = mpmath.mpf
mpc = mpmath.mpc
matrix = mpmath.matrix

pi = mp.pi
e_ = mp.e
tau = pi * 2
d2r = mp.degree
phi = mp.phi
euler = mp.euler
twinprime = mp.twinprime

Function = sympy.Function
Symbol = sympy.Symbol
factorize = factorint = primeFactors = sympy.ntheory.factorint
mobius = sympy.ntheory.mobius

from sympy.parsing.sympy_parser import parse_expr

def diff(string):
    func = parse_expr(string)
    return str(sympy.diff(func))

def intg(string):
    func = parse_expr(string)
    return str(sympy.integrate(func))

def simplify(string):
    func = parse_expr(string)
    return str(func)

def solve(string):
    func = parse_expr(string)
    return sympy.solve(func)

derivative = differentiate = diff
integral = integrate = intg

def shuffle(it):
    if type(it) is list:
        random.shuffle(it)
        return it
    elif type(it) is tuple:
        it = list(it)
        random.shuffle(it)
        return it
    elif type(it) is dict:
        ir = list(it)
        random.shuffle(ir)
        return {k: it[k] for k in ir}
    else:
        raise TypeError("Shuffling " + type(it) + " is not supported.")

def nop(*args):
    pass

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
    return x


def ceil(x):
    try:
        return math.ceil(x)
    except:
        if type(x) is complex:
            return ceil(x.real) + ceil(x.imag) * 1j
    return x


def floor(x):
    try:
        return math.floor(x)
    except:
        if type(x) is complex:
            return floor(x.real) + floor(x.imag) * 1j
    return x


def trunc(x):
    try:
        return math.trunc(x)
    except:
        if type(x) is complex:
            return trunc(x.real) + trunc(x.imag) * 1j
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
    primes = []
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
        carry //= 1000
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
    h = h.replace(" ", "")
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


__units = {
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
        return [str(s) + " millenia"]
    taken = []
    for i in __units:
        a = None
        t = m = __units[i]
        if type(t) is list:
            t = t[0]
        if type(t) is int:
            a = round(s, 3)
        elif s >= t:
            a = int(s / t)
            s = s % t
        if a is not None:
            if a != 1:
                if type(m) is list:
                    i = m[1]
                else:
                    i += "s"
            taken.append(str(roundMin(a)) + " " + str(i))
    if not len(taken):
        return [str(roundMin(s)) + " seconds"]
    return taken


def dhms(s):
    if not isValid(s):
        return str(s)
    output = str(round(s % 60))
    if len(output) < 2:
        output = "0" + output
    if s >= 60:
        temp = str(int((s // 60) % 60))
        if len(temp) < 2 and s > 3600:
            temp = "0" + temp
        output = temp + ":" + output
        if s >= 3600:
            temp = str(int(((s // 60) // 60) % 24))
            if len(temp) < 2 and s >= 86400:
                temp = "0" + temp
            output = temp + ":" + output
            if s >= 86400:
                output = str(int(((s // 60) // 60) // 24)) + ":" + output
    else:
        output = "0:" + output
    return output


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


class dynamicFunc:
    def __init__(self, func):
        self.text = func
        self.func = eval(func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.text


def performAction(action):
    try:
        time.sleep(action[-1])
    except:
        pass
    if len(action) > 1:
        x = action[1]
    else:
        x = None
    if x is not None:
        if type(x) is list:
            y = action[0](*x)
        else:
            y = action[0](*x[0], **x[1])
    else:
        y = action[0]()
    if len(action) > 4:
        action[2][action[3]] = y


class _parallel:
    def __init__(self):
        self.max = 32
        self.running = {i: self.new() for i in range(self.max)}
        for i in self.running:
            self.running[i].start()

    class new(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.actions = []
            self.state = 0
            self.daemon = True
            self._stop = threading.Event()

        def __call__(self, *action):
            self.actions.append(action)
            self.state = 1

        def run(self):
            while True:
                try:
                    while self.actions:
                        action = self.actions[0]
                        self.actions = self.actions[1:]
                        performAction(action)
                    self.state = -1
                    time.sleep(0.007)
                except TimeoutError:
                    pass

        def stop(self):
            self._stop.set()

        def get_id(self):
            if hasattr(self, "_thread_id"):
                return self._thread_id
            for t_id, thread in threading._active.items():
                if thread is self:
                    return t_id

        def kill(self):
            thread_id = self.get_id()
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                thread_id, ctypes.py_object(TimeoutError)
                )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    thread_id, ctypes.py_object(BaseException)
                    )
                self.stop()


def doParallel(func, data_in=None, data_out=[0], start=0, end=None,
               per=1, delay=0, maxq=64, name=False):
    global processes
    if end == None:
        end = len(data_out)
    ps = processes.running
    for i in range(start, end):
        if name:
            d = name
            ps[d] = processes.new()
            p = ps[d]
            p.start()
        else:
            t = d = 0
            p = ps[0]
            while p.state > 0:
                d = xrand(processes.max)
                p = ps[d]
                if t > processes.max:
                    break
                t += 1
            while p.state > 1 or len(p.actions) >= maxq:
                d = xrand(processes.max)
                p = ps[d]
        p(func, data_in, data_out, i, delay)


def killThreads():
    global processes
    running = tuple(processes.running)
    for i in running:
        if type(i) is int and i in processes.running:
            p = processes.running[i]
            p.kill()
            del p
    processes = _parallel()


def waitParallel(delay):
    global processes
    t = time.time()
    running = tuple(processes.running)
    for i in running:
        if type(i) is int and i in processes.running:
            p = processes.running[i]
            while p.state > 0 and time.time() - t < delay:
                time.sleep(0.001)


def updatePrint():
    global printGlobals, printLocals, printVars, origPrint
    while True:
        if printLocals:
            printLocals += printVars
            origPrint(printLocals,end="")
            printGlobals += printLocals
            printLocals = ""
        time.sleep(0.1)


def logPrint(*args, sep=" ", end="\n"):
    global printLocals
    printLocals += str(sep).join((str(i) for i in args)) + str(end)


def setPrint(string):
    global printVars
    printVars = string


def dumpLogData():
    global printGlobals
    f = open("cache/log.txt", "w")
    f.write(printGlobals)
    f.close()


processes = _parallel()
printVars = ""
printLocals = ""
printGlobals = ""
origPrint = print
print = logPrint
doParallel(updatePrint, name="printer")
