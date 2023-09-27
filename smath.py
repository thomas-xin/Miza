"""
Adds many useful math-related functions.
"""

import os, contextlib, concurrent.futures
from concurrent.futures import thread

def _adjust_thread_count(self):
	# if idle threads are available, don't spin new threads
	try:
		if self._idle_semaphore.acquire(timeout=0):
			return
	except AttributeError:
		pass

	# When the executor gets lost, the weakref callback will wake up
	# the worker threads.
	def weakref_cb(_, q=self._work_queue):
		q.put(None)

	num_threads = len(self._threads)
	if num_threads < self._max_workers:
		thread_name = '%s_%d' % (self._thread_name_prefix or self, num_threads)
		args = [
			thread.weakref.ref(self, weakref_cb),
			self._work_queue,
		]
		try:
			args.extend((
				self._initializer,
				self._initargs,
			))
		except AttributeError:
			pass
		t = thread.threading.Thread(
			name=thread_name,
			target=thread._worker,
			args=args,
			daemon=True
		)
		t.start()
		self._threads.add(t)
		thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = lambda self: _adjust_thread_count(self)

import_exc = concurrent.futures.ThreadPoolExecutor(max_workers=48)
submit = import_exc.submit

class MultiAutoImporter:

	class ImportedModule:

		def __init__(self, module, pool, _globals, start=True):
			object.__setattr__(self, "__module", module)
			if start:
				fut = pool.submit(__import__, module)
				object.__setattr__(self, "__fut", fut)
			object.__setattr__(self, "__globals", _globals)

		def __getattr__(self, k):
			m = self.force()
			return getattr(m, k)

		def __setattr__(self, k, v):
			m = self.force()
			return setattr(m, k, v)

		def force(self):
			module = object.__getattribute__(self, "__module")
			_globals = object.__getattribute__(self, "__globals")
			try:
				_globals[module] = m = object.__getattribute__(self, "__fut").result()
			except AttributeError:
				_globals[module] = m = __import__(module)
			return m

	def __init__(self, *args, pool=None, _globals=None):
		self.pool = pool
		if not _globals:
			_globals = globals()
		args = " ".join(args).replace(",", " ").split()
		if not pool:
			_globals.update((k, __import__(k)) for k in args)
		else:
			futs = []
			for arg in args:
				futs.append(self.ImportedModule(arg, pool, _globals, start=len(futs) < 3))
			self.futs = futs
			_globals.update(zip(args, futs))
			submit(self.scan)

	def scan(self):
		for i, sub in enumerate(self.futs):
			object.__getattribute__(sub, "__fut").result()
			for j in range(i + 1, len(self.futs)):
				try:
					object.__getattribute__(self.futs[j], "__fut")
				except AttributeError:
					module = object.__getattribute__(self.futs[j], "__module")
					fut = self.pool.submit(__import__, module)
					# sys.stderr.write(str(fut) + "\n")
					object.__setattr__(self.futs[j], "__fut", fut)
					break

MultiAutoImporter(
	"sys",
	"collections",
	"time",
	"requests",
	"traceback",
	"numpy",
	"sympy",
	"dateutil",
	"datetime",
	"colormath",
	"pytz",
	"ast",
	"copy",
	"pickle",
	"io",
	"random",
	"cmath",
	"fractions",
	"mpmath",
	"shlex",
	"colorsys",
	"re",
	"hashlib",
	"base64",
	"itertools",
	pool=import_exc,
	_globals=globals(),
)
collections2f = "misc/collections2.py"

def update_collections2():
	with requests.get("https://raw.githubusercontent.com/thomas-xin/Python-Extra-Classes/main/full.py") as resp:
		b = resp.content
	with open(collections2f, "wb") as f:
		f.write(b)
	print("collections2.py updated.")
	if "alist" in globals():
		return
	exec(compile(b, "collections2.py", "exec"), globals())

if not os.path.exists(collections2f):
	update_collections2()
with open(collections2f, "rb") as f:
	b = f.read()
if time.time() - os.path.getmtime(collections2f) > 3600:
	import_exc.submit(update_collections2)

import math
from math import *
dateutil.force()
sympy.force()
colormath.force()
from dateutil import parser as tparser
from sympy.parsing.sympy_parser import parse_expr
from colormath import color_objects, color_conversions

if not hasattr(time, "time_ns"):
	time.time_ns = lambda: int(time.time() * 1e9)


suppress = lambda *args, **kwargs: contextlib.suppress(BaseException) if not args and not kwargs else contextlib.suppress(*args + tuple(kwargs.values()))
closing = contextlib.closing
repeat = itertools.repeat
chain = itertools.chain


print_exc = lambda *args: sys.stdout.write(("\n".join(as_str(i) for i in args) + "\n" if args else "") + traceback.format_exc())

loop = lambda x: repeat(None, x)

def try_int(i):
	if type(i) is str and not i.isnumeric():
		return i
	try:
		return int(i)
	except:
		return i

array = numpy.array
np = numpy
exec(compile(b, "collections2.py", "exec"), globals())
try:
	np.float80 = np.longdouble
except AttributeError:
	np.float80 = np.float64
deque = collections.deque

ts_us = lambda: time.time_ns() // 1000
utc = lambda: time.time_ns() / 1e9

random.seed(random.randint(0, (1 << 32) - 1) - time.time_ns())
mp = mpmath.mp
mp.dps = 128
mpf = mpmath.mpf
mpf.__floordiv__ = lambda x, y: int(x / y)
mpf.__rfloordiv__ = lambda y, x: int(x / y)
mpf.__lshift__ = lambda x, y: x * (1 << y if type(y) is int else 2 ** y)
mpf.__rshift__ = lambda x, y: x // (1 << y if type(y) is int else 2 ** y)
mpf.__rlshift__ = lambda y, x: x * (1 << y if type(y) is int else 2 ** y)
mpf.__rrshift__ = lambda y, x: x * (1 << y if type(y) is int else 2 ** y)
mpc = mpmath.mpc
Mat = mat = matrix = mpmath.matrix

math.round = round
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

# Not completely safe, but much safer than regular eval
safe_eval = lambda s: eval(as_str(s).replace("__", ""), {}, eval_const) if not s.isnumeric() else int(s)

null = None
i = I = j = J = 1j
π = pi = mp.pi
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
factorize = factorint = prime_factors = sympy.ntheory.factorint
mobius = sympy.ntheory.mobius

TRUE, FALSE = True, False
true, false = True, False


nop = lambda *void1, **void2: None
nofunc = lambda arg, *void1, **void2: arg

capwords = lambda s, spl=None: (" " if spl is None else spl).join(w.capitalize() for w in s.split(spl))


def choice(*args):
	if not args:
		return
	it = args if len(args) > 1 or not issubclass(type(args[0]), collections.abc.Sized) else args[0]
	if not issubclass(type(it), collections.abc.Sequence):
		if not issubclass(type(it), collections.abc.Sized):
			it = tuple(it)
		else:
			size = len(it)
			it = iter(it)
			i = xrand(size)
			for _ in loop(i):
				next(it)
			return next(it)
	return random.choice(it)


# Shuffles an iterable, in-place if possible, returning it.
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
	elif isinstance(it, alist):
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
			raise TypeError(f"Shuffling {type(it)} is not supported.")

# Reverses an iterable, in-place if possible, returning it.
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
	elif isinstance(it, alist):
		temp = it.reverse()
		it.data = temp.data
		it.offs = temp.offs
		it.hash = None
		del temp
		return it
	else:
		try:
			return list(reversed(it))
		except TypeError:
			raise TypeError(f"Reversing {type(it)} is not supported.")

# Sorts an iterable with an optional key, in-place if possible, returning it.
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
	elif isinstance(it, alist):
		it.__init__(sorted(it, key=key, reverse=reverse))
		it.hash = None
		return it
	else:
		try:
			it = list(it)
			it.sort(key=key, reverse=reverse)
			return it
		except TypeError:
			raise TypeError(f"Sorting {type(it)} is not supported.")


literal_eval = lambda s: ast.literal_eval(as_str(s).lstrip())

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

# Returns a generator iterating through bits and their respective positions in an int.
bits = lambda x: (i for i in range((x if type(x) is int else int(x)).bit_length()) if x & (1 << i))

# Returns the integer floor square root of a number.
def isqrt(x):
	x = x if type(x) is int else int(x)
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

_divmod = divmod
def divmod(x, y):
	with suppress(TypeError):
		return _divmod(x, y)
	return x // y, x % y

# Rounds a number to a certain amount of decimal places.
def round(x, y=None):
	if isinstance(x, int):
		return x
	try:
		if is_finite(x):
			try:
				if x == int(x):
					return int(x)
				if y is None:
					return int(math.round(x))
			except:
				pass
			return round_min(math.round(x, y))
		else:
			return x
	except:
		pass
	if isinstance(x, complex):
		return round(x.real, y) + round(x.imag, y) * 1j
	try:
		return math.round(x, y)
	except:
		pass
	return x

# Rounds a number to the nearest integer, with a probability determined by the fractional part.
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

# Rounds x to the nearest multiple of y.
round_multiple = lambda x, y=1: round_min(math.round(x / y) * y) if y else x
# Randomly rounds x to the nearest multiple of y.
round_random_multiple = lambda x, y=1: round_min(round_random(x / y) * y) if y else x

# Returns integer ceiling value of x, for all complex x.
def ceil(x):
	with suppress(Exception):
		return math.ceil(x)
	if isinstance(x, str):
		x = float(x)
	elif isinstance(x, complex):
		return ceil(x.real) + ceil(x.imag) * 1j
	with suppress(Exception):
		return math.ceil(x)
	return x

# Returns integer floor value of x, for all complex x.
def floor(x):
	with suppress(Exception):
		return math.floor(x)
	if isinstance(x, str):
		x = float(x)
	elif isinstance(x, complex):
		return floor(x.real) + floor(x.imag) * 1j
	with suppress(Exception):
		return math.floor(x)
	return x

# Returns integer truncated value of x, for all complex x.
def trunc(x):
	with suppress(Exception):
		return math.trunc(x)
	if isinstance(x, str):
		x = float(x)
	elif isinstance(x, complex):
		return trunc(x.real) + trunc(x.imag) * 1j
	with suppress(Exception):
		return math.trunc(x)
	return x


# Square wave function with period 2π.
sqr = lambda x: ((sin(x) >= 0) << 1) - 1

# Saw wave function with period 2π.
saw = lambda x: (x / pi + 1) % 2 - 1

# Triangle wave function with period 2π.
tri = lambda x: (abs((0.5 - x / pi) % 2 - 1)) * 2 - 1


# Sign function of a number.
sgn = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)

# Floating point random function.
frand = lambda x=1, y=0: (random.random() * max(x, y) / mpf(random.random())) % x + y

# Integer random function.
def xrand(x, y=None, z=0):
	if y is None:
		y = 0
	if x == y:
		return x
	return random.randint(round_random(min(x, y)), round_random(max(x, y)) - 1) + z

# Returns a floating point number reduced to a power.
rrand = lambda x=1, y=0: frand(x) ** (1 - y)


# Computes modular inverse of two integers.
def modular_inv(a, b):
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


# Computes Pisano period of an integer.
def pisano_period(x):
	a, b = 0, 1
	for i in range(0, x * x):
		a, b = b, (a + b) % x
		if a == 0 and b == 1:
			return i + 1


# Computes Jacobi value of two numbers.
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


# Generator that iterates through numbers 6n±1
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


# Checks if a number is prime using multiple probability tests limited to O(log^2(n)) iterations.
def is_prime(n):

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

	def solovoy_strassen(n):
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
		if not solovoy_strassen(n):
			return False
		return True
	return None

# Generates a number of prime numbers between a and b.
def generate_primes(a=2, b=inf, c=1):
	primes = alist()
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


# Returns the sum of an iterable, using the values rather than keys for dictionaries.
def iter_sum(it):
	if issubclass(type(it), collections.abc.Mapping):
		return sum(tuple(it.values()))
	with suppress(TypeError):
		return sum(iter(it))
	return it

# Returns the maximum value of an iterable, using the values rather than keys for dictionaries.
def iter_max(it):
	if issubclass(type(it), collections.abc.Mapping):
		keys, values = tuple(it.keys()), tuple(it.values())
		m = max(values)
		for k in keys:
			if it[k] >= m:
				return k
	with suppress(TypeError):
		return max(iter(it))
	return it

# This is faster than dict.setdefault apparently
def set_dict(d, k, v, ignore=False):
	try:
		v = d.__getitem__(k)
		if v is None and ignore:
			raise LookupError
	except LookupError:
		d.__setitem__(k, v)
	return v

# Adds two dictionaries similar to dict.update, but adds conflicting values rather than replacing.
def add_dict(a, b, replace=True, insert=None):
	if not issubclass(type(a), collections.abc.MutableMapping):
		if replace:
			r = b
		else:
			r = copy.copy(b)
		try:
			r[insert] += a
		except KeyError:
			r[insert] = a
		return r
	elif not issubclass(type(b), collections.abc.MutableMapping):
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
			if issubclass(type(temp), collections.abc.MutableMapping) or issubclass(type(b[k]), collections.abc.MutableMapping):
				r[k] = add_dict(b[k], temp, replace)
				continue
			r[k] = b[k] + temp
	return r

# Increments a key value pair in a dictionary, replacing if nonexistent.
def inc_dict(_, **kwargs):
	for k, v in kwargs.items():
		try:
			_[k] += v
		except KeyError:
			_[k] = v
	return _

# Subtracts a list of keys from a dictionary.
def sub_dict(d, key):
	output = dict(d)
	try:
		key[0]
	except TypeError:
		key = [key]
	for k in key:
		output.pop(k, None)
	return output


# Casts a number to integers if the conversion would not alter the value.
def round_min(x):
	if isinstance(x, int):
		return x
	if isinstance(x, str):
		if not x:
			return
		if x[0] == "-" and x[1:].isnumeric():
			return -int(x[1:])
		if x.isnumeric():
			return int(x)
		if "." in x:
			x = x.strip("0")
			if x == ".":
				return 0
			x = float(x)
		else:
			try:
				return int(x)
			except ValueError:
				return float(x)
	if isinstance(x, complex):
		if x.imag == 0:
			return round_min(x.real)
		else:
			return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)
	if math.isfinite(x):
		y = round(x)
		if x == y:
			return int(y)
	return x

# Rounds a number to various fractional positions if possible.
def close_round(n):
	rounds = [0.125, 0.375, 0.625, 0.875, 0.25, 0.5, 0.75, 1 / 3, 2 / 3]
	a = math.floor(n)
	b = n % 1
	c = round(b, 1)
	for i in range(0, len(rounds)):
		if abs(b - rounds[i]) < 0.02:
			c = rounds[i]
	return mpf(a + c)

# Converts a float to a fraction represented by a numerator/denominator pair.
def to_frac(num, limit=2147483647):
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


def astype(obj, types, *args, **kwargs):
	if isinstance(types, tuple):
		tl = tuple(t for t in types if isinstance(t, type))
	else:
		tl = None
	tl = tl or types
	try:
		if not isinstance(obj, tl):
			raise TypeError
	except TypeError:
		t = types[0] if isinstance(types, tuple) else types
		if callable(t):
			return t(obj, *args, **kwargs)
		return t
	return obj


gcd = math.gcd

if sys.version_info[1] >= 9:
	lcm = lcm2 = math.lcm
else:
	# Computes the lowest common multiple of two numbers.
	def lcm2(x, y=1):
		if x != y:
			x = abs(x)
			y = abs(y)
			i = True
			if x != int(x):
				i = False
				x = to_frac(x)[0]
			if y != int(y):
				i = False
				y = to_frac(y)[0]
			if i:
				return x * y // gcd(x, y)
			else:
				return to_frac(x / y)[0]
		return x

	# Computes the lowest common multiple of numbers in an arbitrary amount of inputs.
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
	primes = generate_primes(1, x, -1)
	y = 1
	for p in primes:
		y *= p ** floor(log(x, p))
	return y

def fold(f, it):
	out = None
	for i in it:
		if out is None:
			out = i
		else:
			out = f(out, i)
	return out


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

def predict_next(seq, limit=12):
	seq = np.array(deque(astype(x, mpf) for x in seq), dtype=object)
	for i in range(min(8, limit), 1 + max(8, min(len(seq), limit))):
		temp = _predict_next(seq[-i:])
		if temp is not None:
			return temp


# Performs super-sampling linear interpolation.
def supersample(a, size):
	n = len(a)
	if n == size:
		return a
	if n < size:
		interp = np.linspace(0, n - 1, size)
		return np.interp(interp, range(n), a)
	try:
		dtype = a.dtype
	except AttributeError:
		dtype = object
	ftype = np.float64 if dtype is object or issubclass(dtype.type, np.integer) else dtype
	x = ceil(n / size)
	interp = np.linspace(0, n - 1, x * size, dtype=ftype)
	a = np.interp(interp, range(n), a)
	return np.mean(a.reshape(-1, x), 1, dtype=dtype)


# Computes the mean of all numbers in an iterable.
mean = lambda *nums: round_min(np.mean(nums))


# Raises a number to a power, keeping sign.
def pwr(x, power=2):
	if x.real >= 0:
		return round_min(x ** power)
	else:
		return round_min(-((-x) ** power))


# Alters the pulse width of an array representing domain values for a function with period 2π.
def pulse(x, y=0.5):
	p = y * tau
	x *= 0.5 / len(x) * (x < p) + 0.5 / (1 - len(x)) * (x >= p)
	return x


isnan = cmath.isnan


# Checks if a number is finite in value.
def is_finite(x):
	if type(x) is int:
		return True
	if type(x) is complex:
		return not (cmath.isinf(x) or cmath.isnan(x))
	with suppress():
		return x.is_finite()
	return math.isfinite(x)


# Inverse exponential function to approach a destination smoothly.
def approach(x, y, z, threshold=0.125):
	if z <= 1:
		x = y
	else:
		x = (x * (z - 1) + y) / z
		if abs(x - y) <= threshold / z:
			x = y
	return x


# I forgot what this was for oops
def scale_ratio(x, y):
	with suppress(ZeroDivisionError):
		return x * (x - y) / (x + y)
	return 0


# Returns a python range object but automatically reversing if the direction is not specified.
def xrange(a, b=None, c=None):
	if b == None:
		b = round(a)
		a = 0
	if c == None:
		if a > b:
			c = -1
		else:
			c = 1
	return range(floor(a), ceil(b), c)


# Returns the Roman Numeral representation of an integer.
def roman_numerals(num, order=0):
	num = num if type(num) is int else int(num)
	carry = 0
	over = ""
	sym = ""
	output = ""
	if num >= 4000:
		carry = num // 1000
		num %= 1000
		over = roman_numerals(carry, order + 1)
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


NumWords = {
	"zero": 0,
	"a": 1,
	"an": 1,
	"one": 1,
	"two": 2,
	"three": 3,
	"four": 4,
	"five": 5,
	"six": 6,
	"seven": 7,
	"eight": 8,
	"nine": 9,
	"ten": 10,
	"eleven": 11,
	"twelve": 12,
	"thirteen": 13,
	"fourteen": 14,
	"fifteen": 15,
	"sixteen": 16,
	"seventeen": 17,
	"eighteen": 18,
	"nineteen": 19,
	"twenty": 20,
	"thirty": 30,
	"forty": 40,
	"fifty": 50,
	"sixty": 60,
	"seventy": 70,
	"eighty": 80,
	"ninety": 90,
	"hundred": 100,
	"thousand": 1000,
	"million": 10 ** 6,
	"billion": 10 ** 9,
	"trillion": 10 ** 12,
	"quadrillion": 10 ** 15,
	"quintillion": 10 ** 18,
	"sextillion": 10 ** 21,
	"septillion": 10 ** 24,
	"octillion": 10 ** 27,
	"nonillion": 10 ** 30,
	"decillion": 10 ** 33,
	"undecillion": 10 ** 36,
	"duodecillion": 10 ** 39,
	"tredecillion": 10 ** 42,
	"quattuordecillion": 10 ** 45,
	"quindecillion": 10 ** 48,
	"sexdecillion": 10 ** 51,
	"septendecillion": 10 ** 54,
	"octodecillion": 10 ** 57,
	"novemdecillion": 10 ** 60,
	"vigintillion": 10 ** 63,
	"unvigintillion": 10 ** 66,
	"duovigintillion": 10 ** 69,
	"tresvigintillion": 10 ** 72,
	"quattuorvigintillion": 10 ** 75,
	"quinvigintillion": 10 ** 78,
	"sesvigintillion": 10 ** 81,
	"septemvigintillion": 10 ** 84,
	"octovigintillion": 10 ** 87,
	"novemvigintillion": 10 ** 90,
	"trigintillion": 10 ** 93,
	"googol": 10 ** 100,
	"centillion": 1e303,
	"googolplex": inf,
	"infinity": inf,
	"inf": inf
}

# Parses English words as numbers.
def num_parse(s):
	out = 0
	words = single_space(s).casefold().split()
	i = 0
	while i < len(words):
		w = words[i]
		x = NumWords.get(w, w)
		if type(x) is str:
			x = int(x)
		while i < len(words) - 1:
			i += 1
			w = words[i]
			y = NumWords.get(w, w)
			if type(y) is str:
				y = int(y)
			if x < y:
				x *= y
			elif x <= 1000:
				x += y
			else:
				i -= 1
				break
		out += x
		i += 1
	return out


__scales = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")

def byte_scale(n, ratio=1024):
	e = 0
	while n >= ratio:
		n /= ratio
		e += 1
		if e >= len(__scales) - 1:
			break
	return f"{round(n, 4)} {__scales[e]}"



# Returns a string representation of a number with a limited amount of characters, using scientific notation when required.
def exp_num(num, maxlen=10, decimals=0):
	if not is_finite(num):
		if num.real > 0:
			return "inf"
		elif num.real < 0:
			return "-inf"
		else:
			return "NaN"
	if type(num) is complex:
		i = exp_num(num.imag, maxlen // 2 - 1, decimals)
		p = "+" if num.imag > 0 else ""
		return exp_num(num.real, ceil(maxlen / 2) - 1, decimals) + p + i + "i"
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
		return n + round_at(num, min(maxlen - numlen - 2 - len(n), decimals))
	else:
		if numlen > 0:
			try:
				loglen = floor(numlen.log10())
			except:
				loglen = floor(math.log10(numlen)) + len(n)
		else:
			loglen = 0
		s = round_at(num / 10 ** numlen, maxlen - loglen - 5)[: max(1, maxlen - loglen - 2)]
		if s[:3] == "10.":
			s = "9." + "9" * (maxlen - loglen - 4)
		return n + s + "e+" + str(numlen)


# Rounds a number to a certain amount of decimal places, appending 0s if the number is too short.
def round_at(num, prec):
	if prec > 0:
		s = str(round(num.real, round(prec)))
		if "." in s:
			while len(s) - s.index(".") <= prec:
				s += "0"
		else:
			s += "." + "0" * prec
		return s
	return str(round(num.real))


# Limits a string to a maximum length, cutting from the middle and replacing with ".." when possible.
def lim_str(s, maxlen=10, mode="centre"):
	if maxlen is None:
		return s
	if type(s) is not str:
		s = str(s)
	over = (len(s) - maxlen) / 2
	if over > 0:
		if mode == "centre":
			half = len(s) / 2
			s = s[:ceil(half - over - 1)] + ".." + s[ceil(half + over + 1):]
		else:
			s = s[:maxlen - 3] + "..."
	return s


# Attempts to convert an iterable to a string if it isn't already
def verify_string(s):
	if type(s) is str:
		return s
	with suppress(TypeError):
		return "".join(s)
	return str(s)


# A hue to colour conversion function with maximum saturation and lightness.
hue2colour = lambda a, offset=0: adj_colour(colorsys.hsv_to_rgb((a / 1536) % 1, 1, 1), offset, 255)

# Converts a colour tuple to a single integer.
def colour2raw(*c):
	while len(c) == 1:
		c = c[0]
	if len(c) == 3:
		return (c[0] << 16) + (c[1] << 8) + c[2]
	return (c[0] << 16) + (c[1] << 8) + c[2] + (c[3] << 24)

# Converts an integer to a colour tuple.
def raw2colour(x):
	if x > 1 << 24:
		return verify_colour(((x >> 16) & 255, (x >> 8) & 255, x & 255, (x >> 24) & 255))
	return verify_colour(((x >> 16) & 255, (x >> 8) & 255, x & 255))

r2c = raw2colour

# Colour space conversion functions
rgb_to_hsv = lambda c: list(colorsys.rgb_to_hsv(*c[:3])) + c[3:]

def rgb_to_hsl(c):
	col = colorsys.rgb_to_hls(*c[:3])
	return [col[0], col[2], col[1]] + c[3:]

hsv_to_rgb = lambda c: list(colorsys.hsv_to_rgb(*c[:3])) + c[3:]
hsl_to_rgb = lambda c: list(colorsys.hls_to_rgb(c[0], c[2], c[1])) + c[3:]
rgb_to_cmy = lambda c: [1 - x for x in c[:3]] + c[3:]
cmy_to_rgb = rgb_to_cmy
rgb_to_lab = lambda c: list(color_conversions.convert_color(color_objects.sRGBColor(*c[:3]), color_objects.LabColor).get_value_tuple()) + c[3:]
lab_to_rgb = lambda c: list(color_conversions.convert_color(color_objects.LabColor(*c[:3]), color_objects.sRGBColor).get_value_tuple()) + c[3:]
rgb_to_luv = lambda c: list(color_conversions.convert_color(color_objects.sRGBColor(*c[:3]), color_objects.LuvColor).get_value_tuple()) + c[3:]
luv_to_rgb = lambda c: list(color_conversions.convert_color(color_objects.LuvColor(*c[:3]), color_objects.sRGBColor).get_value_tuple()) + c[3:]
rgb_to_xyz = lambda c: list(color_conversions.convert_color(color_objects.sRGBColor(*c[:3]), color_objects.XYZColor).get_value_tuple()) + c[3:]
xyz_to_rgb = lambda c: list(color_conversions.convert_color(color_objects.XYZColor(*c[:3]), color_objects.sRGBColor).get_value_tuple()) + c[3:]

# Converts hex to an colour.
hex2colour = lambda h: verify_colour(hex2bytes(h))

# Computes luma (observed brightness) of a colour.
luma = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

# Verifies a colour is valid.
def verify_colour(c):
	if type(c) is not list:
		c = list(c)
	for i in range(len(c)):
		if c[i] > 255:
			c[i] = 255
		elif c[i] < 0:
			c[i] = 0
		else:
			c[i] = int(abs(c[i]))
	return c

# Generates a 3 colour tuple filled with a single value.
def fill_colour(a):
	if type(a) is complex:
		a = a.real
	a = round(a)
	if a > 255:
		a = 255
	elif a < 0:
		a = 0
	return [a, a, a]

# Returns a black or white background for a certain colour, using the luma value to determine which one to use.
def neg_colour(c, t=127):
	i = luma(c)
	if i > t:
		return fill_colour(0)
	return fill_colour(255)

# Inverts a colour.
inv_colour = lambda c: [255 - i for i in c]

# Adjusts a colour with optional settings.
def adj_colour(colour, brightness=0, intensity=1, hue=0, bits=0, scale=False):
	if hue != 0:
		h = list(colorsys.rgb_to_hsv(*(array(colour) / 255)))
		c = adj_colour(colorsys.hsv_to_rgb((h[0] + hue) % 1, h[1], h[2]), intensity=255)
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
	c = bit_crush(c, bits)
	return verify_colour(c)


# Reduces a number's bit precision.
def bit_crush(dest, b=0, f=round):
	try:
		a = 1 << b
	except (TypeError, ValueError):
		a = 2 ** b
	try:
		len(dest)
		dest = list(dest)
		for i in range(len(dest)):
			dest[i] = f(dest[i] / a) * a
	except TypeError:
		dest = f(dest / a) * a
	return dest


# Returns the permutations of values in a list.
def list_permutation(dest):
	order = np.zeros(len(dest))
	for i in range(len(dest)):
		for j in range(i, len(dest)):
			if dest[i] > dest[j]:
				order[i] += 1
			elif dest[i] < dest[j]:
				order[j] += 1
	return order


# Computes product of values in an iterable.
def product(*nums):
	try:
		return np.prod(nums)
	except:
		p = 1
		for i in nums:
			p *= i
		return p

# Compues dot product of one or multiple 1 dimensional values.
def dot_product(*vects):
	if len(vects) > 1:
		return sum(product(*(array(v) for v in vects)))
	return sum((i ** 2 for i in vects[-1]))

dot = dot_product


# Clips the values in the source iterable to the values in the destination value.
def clip_list(source, dest, direction=False):
	for i in range(len(source)):
		if direction:
			if source[i] < dest[i]:
				source[i] = dest[i]
		else:
			if source[i] > dest[i]:
				source[i] = dest[i]
	return source


# Generates a random polar coordinate on a circle of radius x.
rand_circum_pos = lambda x=1: pol2cart(frand(x), frand(tau))

# Converts polar coordinates to cartesian coordinates.
def pol2cart(dist, angle, pos=None):
	p = array(x * dist for x in (math.cos(angle), math.sin(angle)))
	if pos is None:
		return p
	return p + pos

# Converts cartesian coordinates to polar coordinates.
def cart2pol(x, y, pos=None):
	if pos is None:
		d = (x, y)
	else:
		d = (x - pos[0], y - pos[1])
	return array([hypot(*d), atan2(*reversed(d))])


# Computes a rect object using another, with an offset from each side.
def convert_rect(rect, edge=0):
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

# Checks whether a point is within a rect.
def in_rect(pos, rect, edge=0):
	dest_rect = convert_rect(rect, edge)
	if pos[0] - dest_rect[0] <= 0:
		return False
	if pos[1] - dest_rect[1] <= 0:
		return False
	if pos[0] - dest_rect[2] > 0:
		return False
	if pos[1] - dest_rect[3] > 0:
		return False
	return True

# Moves a position into a rect based on the position it should be in.
def move_to_rect(pos, rect, edge=0):
	p = list(pos)
	if not all(is_finite(i) for i in pos):
		return p, True, True
	dest_rect = convert_rect(rect, 0)
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

# Moves a position into a circle around the centre of a rect if it is outside.
def round_to_rect(pos, rect, edge=0):
	dest_rect = convert_rect(rect, edge)
	if not in_rect(pos, rect, edge):
		s = array(dest_rect[:2])
		t = array(pos)
		p = array(dest_rect[2:]) - s
		m = p / 2
		diff = t - s - m
		angle = atan2(*reversed(diff))
		vel = pol2cart(hypot(*m), angle)
		pos = vel + s + m
	return pos


# Returns the predicted position of an object with given velocity and decay at a certain time.
def time2disp(r, s, t):
	if r == 1:
		return s * t
	return log(s * (r ** t - 1), r)

# Returns the predicted time taken for an object with given velocity and decay to reach a certain position.
def disp2time(r, s, d):
	coeff = d * log(r) / s + 1
	if coeff < 0:
		return inf
	return log(coeff, r)

# Computes approximate intercept angle for a particle, with speed and optional decay values.
def predict_trajectory(src, dest, vel, spd, dec=1, boundary=None, edge=0):
	pos = array(dest)
	dist = hypot(*(src - dest))
	for _ in loop(64):
		time = disp2time(dec, spd, dist)
		new_pos = dest + vel * min(time, 1 << 32)
		if boundary:
			new_pos = array(move_to_rect(new_pos, boundary, edge)[0])
		new_dist = hypot(*(new_pos - pos))
		pos = new_pos
		dist = hypot(*(src - pos))
		if new_dist < 0.0625:
			break
	return pos


# A elastic circle collision function that takes into account masses and radii.
def process_collision(pos1, pos2, vel1, vel2, mass1, mass2, radius1, radius2):
	diff = pos1 - pos2
	dist = frame_dist(pos1, pos2, -vel1, -vel2)
	mindist = radius1 + radius2
	if dist < mindist:
		pos1, pos2 = array(pos1), array(pos2)
		vel1, vel2 = array(vel1), array(vel2)
		dist -= 1
		angle = atan2(*reversed(diff))
		mov = pol2cart(mindist - dist + 1, angle)
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
		arg1 = dot(veld1, vect1) / dot(vect1)
		arg2 = dot(veld2, vect2) / dot(vect2)
		vect1 *= coeff1 * arg1
		vect2 *= coeff2 * arg2
		vel1 -= vect1
		vel2 -= vect2
		hit = True
	else:
		hit = False
	return hit, pos1, pos2, vel1, vel2


# Returns the difference between two angles.
def angle_diff(angle1, angle2, unit=tau):
	angle1 %= unit
	angle2 %= unit
	if angle1 > angle2:
		angle1, angle2 = angle2, angle1
	a = abs(angle2 - angle1)
	b = abs(angle2 - unit - angle1)
	return min(a, b)

# Returns the distance between two angles.
def angle_dist(angle1, angle2, unit=tau):
	angle1 %= unit
	angle2 %= unit
	a = angle2 - angle1
	b = angle2 - unit - angle1
	c = angle2 + unit - angle1
	return sorted((a, b, c), key=lambda x: abs(x))[0]


# Returns the closest approach distance between two objects with constant velocity, over a certain time frame.
def frame_dist(pos1, pos2, vel1, vel2):
	line1 = [pos1 - vel1, pos1]
	line2 = [pos2 - vel2, pos2]
	return interval_interval_dist(line1, line2)

# Returns the distance between two intervals.
def interval_interval_dist(line1, line2):
	if intervals_intersect(line1, line2):
		return 0
	distances = (
		point_interval_dist(line1[0], line2),
		point_interval_dist(line1[1], line2),
		point_interval_dist(line2[0], line1),
		point_interval_dist(line2[1], line1),
	)
	return min(distances)

# Returns the distance between a point and an interval.
def point_interval_dist(point, line):
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

# Checks if two intervals intersect at a point.
def intervals_intersect(line1, line2):
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
	t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / -delta
	return (0 <= s <= 1) and (0 <= t <= 1)


# Evaluates an expression along a domain of input values.
def func2array(func, size=4096):
	function = eval("lambda x: " + str(func))
	period = 2 * pi
	array = function(np.arange(0, period, 1 / (size + 1) * period))
	return array

# Computes harmonics (Fourier Transform) of an array.
def array2harmonics(data, precision=1024):
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

# Computes an array (Inverse Fourier Transform) of an array.
def harmonics2array(period, harmonics, func="sin(x)"):
	expression = func
	function = eval("lambda x: " + expression)
	result = 0
	t = np.arange(period)
	for n, (a, b) in enumerate(harmonics):
		result += a * function((n + 1) * t * 2 * pi / period + b)
	return result


# Limits a string to an amount of lines.
def lim_line(s, lim):
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


# Removes an argument from a string, separated by spaces.
def remove_str(s, arg):
	if arg + " " in s:
		s = s.replace(arg + " ", "")
		return s, True
	elif " " + arg in s:
		s = s.replace(" " + arg, "")
		return s, True
	return s, False


# Returns a string representation of an iterable, with options.
def iter2str(it, key=None, limit=3840, offset=0, left="[", right="]", sep=" "):
	try:
		try:
			len(it)
		except TypeError:
			it = alist(i for i in it)
	except:
		it = alist(it)
	if issubclass(type(it), collections.abc.Mapping):
		keys = it.keys()
		values = iter(it.values())
	else:
		keys = range(offset, offset + len(it))
		values = iter(it)
	spacing = int(math.log10(max(1, len(it) + offset - 1)))
	s = ""
	with suppress(StopIteration):
		for k in keys:
			index = k if type(k) is str else sep * (spacing - int(math.log10(max(1, k)))) + str(k)
			s += f"\n{left}{index}{right} "
			if key is None:
				s += str(next(values))
			else:
				s += str(key(next(values)))
	return lim_str(s, limit)


# Recognises "st", "nd", "rd" and "th" in numbers.
_ith = "st nd rd th".split()
rank_format = lambda n: str(n) + (_ith[min((n - 1) % 10, 3)] if n not in range(11, 14) else "th")


# Returns a copy of a mapping object, with keys cast to integers where possible.
def int_key(d):
	c = d.__class__(d)
	for k in tuple(d):
		try:
			t = d[k]
		except KeyError:
			continue
		with suppress(TypeError, ValueError):
			k = int(k)
		if isinstance(t, dict):
			t = int_key(t)
		c[k] = t
	return c


# Time functions
class DynamicDT:

	__slots__ = ("_dt", "_offset", "_ts", "tzinfo")

	def __getstate__(self):
		return self.timestamp(), getattr(self, "tzinfo", None)

	def __setstate__(self, s):
		if len(s) == 2:
			ts, tzinfo = s
			offs, ots = divmod(ts, 12622780800)
			self._dt = datetime.datetime.fromtimestamp(ots, tz=tzinfo)
			self.tzinfo = tzinfo
			self._offset = round(offs * 400)
			return
		raise TypeError("Unpickling failed:", s)

	def __init__(self, *args, **kwargs):
		tzinfo = kwargs.pop("tzinfo", None)
		if type(args[0]) is bytes:
			self._dt = datetime.datetime(args[0], tzinfo=tzinfo)
			return
		offs, y = divmod(args[0], 400)
		y += 2000
		offs *= 400
		offs -= 2000
		self._dt = datetime.datetime(y, *args[1:], tzinfo=tzinfo, **kwargs)
		self.set_offset(offs)

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		return getattr(self._dt, k)

	def __str__(self):
		y = self.year_repr()
		return y + str(self._dt)[4:].rsplit("+", 1)[0]

	def __repr__(self):
		return self.__class__.__name__ + "(" + ", ".join(str(i) for i in self._dt.timetuple()[:6]) + (f", microsecond={self._dt.microsecond}" if getattr(self._dt, "microsecond", 0) else "") + ").set_offset(" + str(self.offset()) + ")"

	def year_repr(self):
		y = self.year
		if y >= 0:
			return "0" * max(0, 3 - int(math.log10(max(1, y)))) + str(y)
		y = -y
		return "0" * max(0, 3 - int(math.log10(max(1, y)))) + str(y) + " BC"

	def timestamp(self):
		with suppress(AttributeError):
			return self._ts
		return self.update_timestamp()

	def update_timestamp(self):
		offs = self.offset() * 31556952
		try:
			self._ts = offs + round_min(self._dt.timestamp())
		except OSError:
			self._ts = offs + round_min((self - ep).total_seconds())
		return self._ts

	def offset(self):
		with suppress(AttributeError):
			return self._offset
		self._offset = 0
		return 0

	def set_offset(self, offs, update_ts=True):
		self._offset = round(offs)
		if update_ts:
			self.update_timestamp()
		return self

	def __add__(self, other):
		if type(other) is not datetime.timedelta:
			return self.__class__.fromtimestamp(self.timestamp() + other, tzinfo=self.tzinfo)
		ts = self._dt.timestamp() + other.timestamp()
		if abs(self.offset()) >= 25600:
			ts = round(ts)
		return self.__class__.fromtimestamp(ts + self.offset() * 31556952, tzinfo=self.tzinfo)
	__radd__ = __add__

	def __sub__(self, other):
		if isinstance(other, self.__class__):
			return datetime.timedelta(seconds=self.offset() - other.offset() + (self._dt - other._dt).total_seconds())
		if type(other) not in (datetime.timedelta, datetime.datetime, datetime.date):
			return self.__class__.fromtimestamp(self.timestamp() - other, tzinfo=self.tzinfo)
		ts = other.total_seconds() if isinstance(other, datetime.timedelta) else other.timestamp()
		ts = self._dt.timestamp() - ts
		if abs(self.offset()) >= 25600:
			ts = round(ts)
		ts = ts + self.offset() * 31556952
		if isinstance(other, (DynamicDT, datetime.datetime, datetime.date)):
			return datetime.timedelta(seconds=ts)
		tzinfo = self.tzinfo if isinstance(self, DynamicDT) else other.tzinfo
		return self.__class__.fromtimestamp(ts, tzinfo=tzinfo)

	__rsub__ = lambda self, other: self.__class__.fromtimestamp(other.timestamp()) - self

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			if self.offset() == other.offset():
				return self._dt == other._dt
		elif isinstance(other, datetime.datetime):
			if self.year == other.year:
				return self._dt == other
		return False

	def __lt__(self, other):
		if isinstance(other, self.__class__):
			if self.offset() < other.offset():
				return True
			elif self.offset() == other.offset():
				return self._dt < other._dt
		elif isinstance(other, datetime.datetime):
			if self.year < other.year:
				return True
			if self.year == other.year:
				return self._dt < other
		return False
	
	def __le__(self, other):
		if isinstance(other, self.__class__):
			if self.offset() < other.offset():
				return True
			elif self.offset() == other.offset():
				return self._dt <= other._dt
		elif isinstance(other, datetime.datetime):
			if self.year < other.year:
				return True
			if self.year == other.year:
				return self._dt <= other
		return False

	def __gt__(self, other):
		if isinstance(other, self.__class__):
			if self.offset() > other.offset():
				return True
			elif self.offset() == other.offset():
				return self._dt > other._dt
		elif isinstance(other, datetime.datetime):
			if self.year > other.year:
				return True
			if self.year == other.year:
				return self._dt > other
		return False
	
	def __ge__(self, other):
		if isinstance(other, self.__class__):
			if self.offset() > other.offset():
				return True
			elif self.offset() == other.offset():
				return self._dt >= other._dt
		elif isinstance(other, datetime.datetime):
			if self.year > other.year:
				return True
			if self.year == other.year:
				return self._dt >= other
		return False

	def add_years(self, years=1):
		if not years:
			return self
		added = years >= 0
		offs = self.offset()
		if abs(years) >= 400:
			x, years = divmod(years, 400)
		else:
			x = 0
		try:
			new_dt = self.__class__(self.year + years, self.month, self.day, self.hour, self.minute, self.second, self.microsecond, tzinfo=self.tzinfo)
		except ValueError:
			if added:
				month = self.month + 1
				if month > 12:
					month = 1
					years += 1
				new_dt = self.__class__(self.year + years, month, 1, self.hour, self.minute, self.second, self.microsecond, tzinfo=self.tzinfo)
			else:
				month = self.month
				day = month_days(self._dt.year + years, month)
				new_dt = self.__class__(self.year + years, month, day, self.hour, self.minute, self.second, self.microsecond, tzinfo=self.tzinfo)
		return new_dt.set_offset(offs + x * 400)

	def add_months(self, months=1):
		if not months:
			return self
		offs = self.offset()
		years = 0
		month = self.month + months
		if month < 0 or month > 12:
			years, month = divmod(month, 12)
		if abs(years) >= 400:
			x, years = divmod(years, 400)
		else:
			x = 0
		try:
			new_dt = self.__class__(self.year + years, month, self.day, self.hour, self.minute, self.second, self.microsecond, tzinfo=self.tzinfo)
		except ValueError:
			month += 1
			if month > 12:
				month = 1
				years += 1
			new_dt = self.__class__(self.year + years, month, 1, self.hour, self.minute, self.second, self.microsecond, tzinfo=self.tzinfo)
		return new_dt.set_offset(offs + x * 400)

	@property
	def year(self):
		return self._dt.year + self.offset()

	def as_date(self):
		y = self.year_repr()
		if y.endswith(" BC"):
			bc = " BC"
			y = y[:-3]
		else:
			bc = ""
		m = self.month
		d = self.day
		return y + "-" + ("0" if m < 10 else "") + str(m) + "-" + ("0" if d < 10 else "") + str(d) + bc

	@classmethod
	def utcfromtimestamp(cls, ts):
		return cls.fromtimestamp(ts, tzinfo=datetime.timezone.utc)

	@classmethod
	def fromtimestamp(cls, ts, tzinfo=None):
		offs, ots = divmod(ts, 12622780800)
		if abs(offs) >= 400:
			ots = round(ots)
			ts = round(ts)
		d = datetime.datetime.fromtimestamp(ots, tz=tzinfo)
		dt = cls(*d.timetuple()[:6], d.microsecond, tzinfo=tzinfo)
		dt._offset += round(offs * 400)
		dt._ts = ts
		return dt

	@classmethod
	def fromdatetime(cls, dt):
		if type(dt) is cls:
			return dt
		return cls(*dt.timetuple()[:6], getattr(dt, "microsecond", 0), tzinfo=getattr(dt, "tzinfo", None))

	@classmethod
	def utcnow(cls):
		return cls.utcfromtimestamp(utc())

	@classmethod
	def now(cls):
		return cls.fromtimestamp(utc())


dtn = datetime.datetime.now
utc_dt = datetime.datetime.utcnow
utc_ft = datetime.datetime.utcfromtimestamp
utc_ddt = DynamicDT.utcnow
utc_dft = DynamicDT.utcfromtimestamp
dt2dt = DynamicDT.fromdatetime
ep = datetime.datetime(1970, 1, 1)

def zerot():
	today = utc_dt()
	return datetime.datetime(today.year, today.month, today.day, tzinfo=datetime.timezone.utc).timestamp()

to_utc = lambda dt: dt.replace(tzinfo=datetime.timezone.utc)
to_naive = lambda dt: dt.replace(tzinfo=None)

def utc_ts(dt):
	if type(dt) is DynamicDT:
		return dt.timestamp()
	with suppress(TypeError):
		return (dt - ep).total_seconds()
	return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

ZONES = {zone.split("/", 1)[-1].replace("-", "").replace("_", "").casefold(): zone for zone in pytz.all_timezones if not zone.startswith("Etc/")}
COUNTRIES = mdict()
for tz in pytz.all_timezones:
	if "/" in tz and not tz.startswith("Etc/"):
		COUNTRIES.append(tz.split("/", 1)[0], tz.split("/", 1)[-1])
CITIES = {city.split("/", 1)[-1].replace("-", "").replace("_", "").casefold(): city for country in COUNTRIES.values() for city in country}

def city_timezone(city):
	return pytz.timezone(ZONES[full_prune(city)])

def city_time(city):
	return to_utc(city_timezone(city).fromutc(utc_dt()))

# Values in seconds of various time intervals.
TIMEUNITS = {
	"galactic year": 7157540528801820.28133333333333,
	"millennium": [31556925216., "millennia"],
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

# Converts a time input in seconds to a list of time intervals.
def time_convert(s):
	if not is_finite(s):
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
			taken.append("-" * r + str(round_min(a)) + " " + str(i))
	if not len(taken):
		return [str(round_min(s)) + " seconds"]
	return taken

# Returns the string representation of a time value in seconds, in word form.
sec2time = lambda s: " ".join(time_convert(s))

def month_days(year, month):
	if month in (4, 6, 9, 11):
		return 30
	elif month == 2:
		if not year % 400:
			return 29
		elif not year % 100:
			return 28
		elif not year % 4:
			return 29
		return 28
	return 31

strnum = lambda num: str(round(num, 6))

# Returns a representation of a time interval using days:hours:minutes:seconds.
def time_disp(s, rounded=True):
	if not is_finite(s):
		return str(s)
	if rounded:
		s = round(s)
	output = strnum(s % 60)
	if len(output) < 2:
		output = "0" + output
	if s >= 60:
		temp = strnum((s // 60) % 60)
		if len(temp) < 2 and s >= 3600:
			temp = "0" + temp
		output = temp + ":" + output
		if s >= 3600:
			temp = strnum((s // 3600) % 24)
			if len(temp) < 2 and s >= 86400:
				temp = "0" + temp
			output = temp + ":" + output
			if s >= 86400:
				output = strnum(s // 86400) + ":" + output
	else:
		output = "0:" + output
	return output

# Converts a time interval represented using days:hours:minutes:seconds, to a value in seconds.
def time_parse(ts):
	if ts == "N/A":
		return inf
	data = ts.split(":")
	if len(data) >= 5: 
		raise TypeError("Too many time arguments.")
	mults = (1, 60, 3600, 86400)
	return round_min(sum(float(count) * mult for count, mult in zip(data, reversed(mults[:len(data)]))))

def time_sum(t2, t1):
	out = ""
	galactic_years = 0
	millennia = 0
	years = t2.year + t1.year
	months = t2.month + t1.month
	days = t2.day + t1.day
	hours = getattr(t2, "hour", 0) + getattr(t1, "hour", 0)
	minutes = getattr(t2, "minute", 0) + getattr(t1, "minute", 0)
	seconds = round(getattr(t2, "second", 0) + getattr(t1, "second", 0) + (getattr(t2, "microsecond", 0) + getattr(t1, "microsecond", 0)) / 1000000, 6)
	while seconds >= 60:
		minutes += 1
		seconds -= 60
	while minutes >= 60:
		hours += 1
		minutes -= 60
	while hours >= 24:
		days += 1
		hours -= 24
	md = month_days(t2.year, t2.month)
	while days > md:
		months += 1
		days -= md
	while months > 12:
		years += 1
		months -= 12
	if abs(years) >= 1000:
		millennia, years = divmod(years, 1000)
	if abs(millennia) >= 226814:
		galactic_years, millennia = divmod(millennia, 226814)
	if galactic_years:
		out += f"{galactic_years} galactic year"
		if galactic_years != 1:
			out += "s"
		out += " "
	if millennia:
		out += f"{millennia} millenni"
		if millennia != 1:
			out += "a"
		else:
			out += "um"
		out += " "
	if years:
		out += f"{years} year"
		if years != 1:
			out += "s"
		out += " "
	if months:
		out += f"{months} month"
		if months != 1:
			out += "s"
		out += " "
	if days:
		out += f"{days} day"
		if days != 1:
			out += "s"
		out += " "
	if hours:
		out += f"{hours} hour"
		if hours != 1:
			out += "s"
		out += " "
	if minutes:
		out += f"{minutes} minute"
		if minutes != 1:
			out += "s"
		out += " "
	if seconds or not out:
		s = str(seconds)
		if "." in s:
			spl = s.split(".", 1)
			s = spl[0] + "." + spl[1][:6].rstrip("0")
		out += f"{s} second"
		if seconds != 1:
			out += "s"
	return out.strip()

def time_diff(t2, t1):
	out = ""
	galactic_years = 0
	millennia = 0
	years = t2.year - t1.year
	months = t2.month - t1.month
	days = t2.day - t1.day
	hours = getattr(t2, "hour", 0) - getattr(t1, "hour", 0)
	minutes = getattr(t2, "minute", 0) - getattr(t1, "minute", 0)
	microsecond = (getattr(t2, "microsecond", 0) - getattr(t1, "microsecond", 0)) / 1000000
	if abs(microsecond) < 0.002:
		microsecond = 0
	seconds = round(getattr(t2, "second", 0) - getattr(t1, "second", 0) + microsecond, 6)
	while seconds < 0:
		minutes -= 1
		seconds += 60
	while minutes < 0:
		hours -= 1
		minutes += 60
	while hours < 0:
		days -= 1
		hours += 24
	while days < 0:
		months -= 1
		days += month_days(t2.year, t2.month - 1)
	while months < 0:
		years -= 1
		months += 12
	if abs(years) >= 1000:
		millennia, years = divmod(years, 1000)
	if abs(millennia) >= 226814:
		galactic_years, millennia = divmod(millennia, 226814)
	if galactic_years:
		out += f"{galactic_years} galactic year"
		if galactic_years != 1:
			out += "s"
		out += " "
	if millennia:
		out += f"{millennia} millenni"
		if millennia != 1:
			out += "a"
		else:
			out += "um"
		out += " "
	if years:
		out += f"{years} year"
		if years != 1:
			out += "s"
		out += " "
	if months:
		out += f"{months} month"
		if months != 1:
			out += "s"
		out += " "
	if days:
		out += f"{days} day"
		if days != 1:
			out += "s"
		out += " "
	if hours:
		out += f"{hours} hour"
		if hours != 1:
			out += "s"
		out += " "
	if minutes:
		out += f"{minutes} minute"
		if minutes != 1:
			out += "s"
		out += " "
	if seconds or not out:
		s = str(seconds)
		if "." in s:
			spl = s.split(".", 1)
			s = spl[0] + "." + spl[1][:6].rstrip("0")
		out += f"{s} second"
		if seconds != 1:
			out += "s"
	return out.strip()

def dyn_time_diff(t2, t1):
	if isnan(t2) or isnan(t1):
		return "NaN"
	if t2 >= inf:
		return "inf galactic years"
	if t2 <= -inf:
		return "-inf galactic years"
	if t2 <= t1:
		return "0 seconds"
	return time_diff(DynamicDT.fromtimestamp(t2), DynamicDT.fromtimestamp(t1))

def time_until(ts):
	return dyn_time_diff(ts, utc())

def next_date(dt):
	t = utc_dt()
	new = datetime.datetime(t.year, dt.month, dt.day)
	while new < t:
		new = new.replace(year=new.year + 1)
	if dt.tzinfo:
		return new.replace(tzinfo=dt.tzinfo)
	return new


def parse_fs(fs):
	if type(fs) is not bytes:
		fs = str(fs).encode("utf-8")
	if fs.endswith(b"TB"):
		scale = 1099511627776
	if fs.endswith(b"GB"):
		scale = 1073741824
	elif fs.endswith(b"MB"):
		scale = 1048576
	elif fs.endswith(b"KB"):
		scale = 1024
	else:
		scale = 1
	return float(fs.split(None, 1)[0]) * scale


RE = cdict()

def regexp(s, flags=0):
	global RE
	if issubclass(type(s), re.Pattern):
		return s
	s = as_str(s)
	t = (s, flags)
	try:
		return RE[t]
	except KeyError:
		RE[t] = re.compile(s, flags)
	return RE[t]


word_count = lambda s: 1 + sum(1 for _ in regexp("\\W+").finditer(s))
single_space = lambda s: regexp("\\s\\s+").sub(" ", s)

# A fuzzy substring search that returns the ratio of characters matched between two strings.
def fuzzy_substring(sub, s, match_start=False, match_length=True):
	if not match_length and s in sub:
		return 1
	if s.startswith(sub):
		return len(sub) / len(s) * 2
	match = 0
	if not match_start or sub and s.startswith(sub[0]):
		found = [0] * len(s)
		x = 0
		for i, c in enumerate(sub):
			temp = s[x:]
			if temp.startswith(c):
				if found[x] < 1:
					match += 1
					found[x] = 1
				x += 1
			elif c in temp:
				y = temp.index(c)
				x += y
				if found[x] < 1:
					found[x] = 1
					match += 1 - y / len(s)
				x += 1
			else:
				temp = s[:x]
				if c in temp:
					y = temp.rindex(c)
					if found[y] < 1:
						match += 1 - (x - y) / len(s)
						found[y] = 1
					x = y + 1
		if len(sub) > len(s) and match_length:
			match *= len(s) / len(sub)
	# ratio = match / len(s)
	ratio = max(0, match / len(s))
	return ratio

# Replaces words in a string from a mapping similar to str.replace, but performs operation both ways.
def replace_map(s, mapping):
	temps = {k: chr(65535 - i) for i, k in enumerate(mapping.keys())}
	trans = "".maketrans({chr(65535 - i): mapping[k] for i, k in enumerate(mapping.keys())})
	for key, value in temps.items():
		s = s.replace(key, value)
	for key, value in mapping.items():
		s = s.replace(value, key)
	return s.translate(trans)

def belongs(s):
	s = s.strip()
	if not s:
		return ""
	if full_prune(s[-1]) == "s":
		return s + "'"
	return s + "'s"


# Converts a bytes object to a hex string.
def bytes2hex(b, space=True):
	if type(b) is str:
		b = b.encode("utf-8")
	if space:
		return b.hex(" ").upper()
	return b.hex().upper()

# Converts a hex string to a bytes object.
def hex2bytes(b):
	s = as_str(b).replace(" ", "")
	if len(s) & 1:
		s = s[:-1] + "0" + s[-1]
	return bytes.fromhex(s)

# Converts a bytes object to a base64 string.
def bytes2b64(b, alt_char_set=False):
	if type(b) is str:
		b = b.encode("utf-8")
	if alt_char_set:
		b = base64.urlsafe_b64encode(b)
		return b.rstrip(b"=")
	return base64.b64encode(b)

# Converts a base 64 string to a bytes object.
def b642bytes(b, alt_char_set=False):
	if type(b) is str:
		b = b.encode("utf-8")
	if alt_char_set:
		if not b.endswith(b"="):
			b += b"=="
		return base64.urlsafe_b64decode(b)
	return base64.b64decode(b)

if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
	randbytes = random.randbytes
else:
	randbytes = lambda size: np.random.randint(0, 256, size=size, dtype=np.uint8).data

# SHA256 operations: base64 and base16.
shash = lambda s: as_str(base64.urlsafe_b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).rstrip(b"="))
uhash = lambda s: shash(s) if len(s) > 43 else s
hhash = lambda s: bytes2hex(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), space=False)
ihash = lambda s: int.from_bytes(hashlib.md5(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), "little") % 4294967296 - 2147483648
nhash = lambda s: int.from_bytes(hashlib.md5(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), "little")

def bxor(b1, b2):
	x = np.frombuffer(b1, dtype=np.uint8)
	y = np.frombuffer(b2, dtype=np.uint8)
	return (x ^ y).tobytes()


# Manages a dict object and uses pickle to save and load it.
class pickled(collections.abc.Callable):

	def __init__(self, obj=None, ignore=()):
		self.data = obj
		self.ignores = set(ignore)
		self.__str__ = obj.__str__

	def __getattr__(self, key):
		try:
			return self.__getattribute__(key)
		except AttributeError:
			pass
		return getattr(self.__getattribute__("data"), key)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(dir(self.data))
		return data

	def __call__(self):
		return self

	def ignore(self, item):
		self.ignores.add(item)

	def __repr__(self):
		c = dict(self.data)
		for i in self.ignores:
			c.pop(i)
		d = pickle.dumps(c)
		if len(d) > 1048576:
			return "None"
		return (
			"pickled(pickle.loads(hex2Bytes('''"
			+ bytes2hex(d, space=False)
			+ "''')))"
		)
