#!/usr/bin/python3

import sympy, mpmath, math, time, os, sys, subprocess, psutil, traceback, random
import collections, itertools, pickle, base64, ast, re
import sympy.stats, scipy.stats
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
print = lambda *args, sep=" ", end="\n": sys.stdout.buffer.write(f"~print({repr(sep.join(map(str, args)))},end={repr(end)})\n".encode("utf-8"))


def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)

literal_eval = lambda s: ast.literal_eval(as_str(s).lstrip())


BF_PREC = 256
sys.set_int_max_str_digits(max(1024, BF_PREC * 256))
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
	if not math.isfinite(s):
		high = "galactic years"
		return [str(s) + " " + high]
	r = 1 if s < 0 else 0
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

# Returns the Roman Numeral representation of an integer.
def roman_numerals(num, order=0):
	num = int(num)
	if num <= 0:
		raise ValueError("Number is not legally representable in Roman Numerals.")
	carry = 0
	over = ""
	sym = ""
	output = ""
	if num >= 4000:
		carry = num // 1000
		num %= 1000
		over = roman_numerals(carry, order + 1)
	elif order:
		over = "(" * order
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
	if order:
		output += ")"
	return over + output


def Random(a=None, b=None):
	if a is None:
		return (
			sympy.Float(random.random())
			+ sympy.Float(random.random()) / 2 ** 52
			+ sympy.Float(random.random()) / 2 ** 104
		)
	elif b is None:
		return random.randint(0, round_random(a) - 1)
	else:
		return random.randint(round_random(a), round_random(b))


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

sympy.Basic.__and__ = lambda self, other: iand(self, other)
sympy.Basic.__or__ = lambda self, other: ior(self, other)
sympy.Basic.__xor__ = lambda self, other: ixor(self, other)
sympy.core.numbers.Infinity.__str__ = lambda self: "inf"
sympy.core.numbers.NegativeInfinity.__str__ = lambda self: "-inf"
sympy.core.numbers.ComplexInfinity.__str__ = lambda self: "ℂ∞"
sympy.erfcinv = lambda z: sympy.erfinv(1 - z)
sympy.erfcinv.inverse = sympy.erfc
sympy.erfc.inverse = sympy.erfcinv

r_evalf = sympy.Rational.evalf
from sympy.solvers.diophantine.diophantine import divisible
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm

_pow = sympy.Float.__pow__

def pow(a, b):
	print("_POW:", a, b)
	if isinstance(a, sympy.Integer) and isinstance(a, sympy.Integer):
		return _pow(a, b)
	temp = _pow(r_evalf(a, BF_PREC), b)
	if temp != 0 and math.log10(abs(temp)) > BF_PREC:
		return temp
	return _pow(a, b)

sympy.Basic.__pow__ = lambda self, other: pow(self, other)
sympy.Basic.__rpow__ = lambda self, other: pow(other, self)

def carmichael(n):
	temp = _factorint(n)
	res = 1
	if temp.get(2, 0) >= 3:
		res = sympy.totient(2 ** (temp.pop(2) - 1))
	for k, v in temp.items():
		res = sympy.lcm(res, sympy.totient(k ** v))
	return res

def simplify_recurring(r, prec=100):
	p, q = r.p, r.q
	try:
		temp = _factorint(q, timeout=1)
	except subprocess.TimeoutExpired as ex:
		print(repr(ex))
		return
	if all(i in (2, 5) for i in temp):
		return
	tq = np.prod([k ** v for k, v in temp.items() if k not in (2, 5)], dtype=object)
	try:
		pr = sympy.ntheory.residue_ntheory.is_primitive_root(10, tq)
	except ValueError:
		pr = False
	cq = carmichael(tq)
	if pr:
		digits = cq
	else:
		try:
			facts = factors(cq, lim=prec ** 2)
		except OverflowError:
			return
		digits = cq
		for f in facts:
			if f < prec * 16 and divisible(10 ** f - 1, tq):
				digits = f
				break
	if digits > prec * 2:
		return
	transient = max(temp.get(2, 0), temp.get(5, 0))
	s = str(r_evalf(r, transient + digits * 3 + 2))
	dec = s.split(".", 1)[-1][transient:]
	if len(dec) < digits * 2:
		return
	assert dec[:digits] == dec[digits:digits * 2]
	if digits > 16:
		return s[:s.index(".") + transient + 1] + "[" + dec[:digits] + "]"
	if digits > 4:
		return s[:s.index(".") + transient + digits + 1] + "[" + dec[:digits] + "]"
	return s[:s.index(".") + transient + digits * 2 + 1] + "[" + dec[:digits] + "]"

class FakeFloat(sympy.Rational):

	__slots__ = ("recur",)

	def __new__(cls, p, q=1, r=None):
		obj = sympy.Expr.__new__(cls)
		if math.log10(max(abs(p), abs(q))) > BF_PREC:
			return r_evalf(obj, BF_PREC, chop=True)
		obj.p, obj.q = p, q
		r = r or simplify_recurring(obj, prec=BF_PREC)
		if r:
			obj.recur = r
		return obj

	def __str__(self):
		return getattr(self, "recur", None) or sympy.Float.__str__(self)

	@property
	def is_Rational(self):
		return hasattr(self, "recur")

def evalf_true(n, prec=100, **void):
	if isinstance(n, sympy.Float):
		return n
	if not isinstance(n, sympy.Rational):
		return r_evalf(n, prec=prec)
	r = simplify_recurring(n, prec=prec)
	if r:
		return FakeFloat(n.p, n.q, r)
	return r_evalf(n, prec)

def _print_Fraction(self, expr):
	if expr.q == 1:
		return expr.p
	if math.log10(max(expr.p, expr.q)) > BF_PREC:
		return r_evalf(expr, BF_PREC, chop=True)
	if self._settings.get("sympy_integers", False):
		return "S(%s)/%s" % (expr.p, expr.q)
	return "%s/%s" % (expr.p, expr.q)

def _pprint_Fraction(self, expr):
	if expr.q == 1:
		return self._print_Atom(expr.p)
	if math.log10(max(expr.p, expr.q)) > BF_PREC:
		return self._print_Float(r_evalf(expr, BF_PREC, chop=True))
	p, q = expr.p, expr.q
	if abs(p) >= 10 and abs(q) >= 10:
		if p < 0:
			return prettyForm(str(p), binding=prettyForm.NEG)/prettyForm(str(q))
		else:
			return prettyForm(str(p))/prettyForm(str(q))
	return self.emptyPrinter(expr)

sympy.Rational.evalf = FakeFloat.evalf = evalf_true
sympy.StrPrinter._print_FakeFloat = lambda self, expr: FakeFloat.__str__(expr)
sympy.StrPrinter._print_Rational = lambda self, expr: _print_Fraction(self, expr)
PrettyPrinter._print_Rational = lambda self, expr: _pprint_Fraction(self, expr)


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

def array(*args, **kwargs):
	if not kwargs.get("dtype"):
		kwargs["dtype"] = object
	if len(args) == 1:
		arr = args[0]
		if type(arr) is str:
			arr = re.split("[^0-9\\-+e./]+", arr)
			arr = list(map(sympy.Rational, arr))
		return np.asanyarray(arr, **kwargs)
	return np.array(args, **kwargs)

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
	for i in range(min(8, limit) - 1, 3, -1):
		temp = _predict_next(seq[-i:])
		if temp is not None:
			return temp

# Multiple variable limit
def lim(f, kwargs=None, **_vars):
	if kwargs:
		_vars.update({str(k): v for k, v in kwargs.items()})
	if hasattr(f, "subs"):
		g = f.subs(_vars)
		try:
			if not math.isnan(g):
				return g
		except TypeError:
			return g
	for i in _vars:
		g = sympy.limit(f, i, _vars[i], "+")
		h = sympy.limit(f, i, _vars[i], "-")
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

has_maxima = os.name == "nt"

def get_maxima():
	try:
		proc = globals()["MAXIMA"]
	except KeyError:
		import psutil
		proc = globals()["MAXIMA"] = psutil.Popen("maxima", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
		o = None
		while o != b"%" and proc.is_running():
			o = proc.stdout.read(1)
		if not proc.is_running():
			raise FileNotFoundError("Process closed.")
	except FileNotFoundError:
		globals()["has_maxima"] = False
		raise
	return proc

def _integrate(*args, **kwargs):
	proc = get_maxima()
	s = "grind(integrate(" + ",".join(map(str, args)) + ",x))$"
	try:
		proc.stdin.write(s.encode("utf-8"))
		proc.stdin.flush()
		proc.stdout.readline()
		b = proc.stdout.readline().decode("utf-8", "replace")
		if b.startswith("incorrect syntax:"):
			raise SyntaxError(b)
		s = b.strip("$; \r\n")
	except:
		if proc.is_running():
			proc.kill()
		globals().pop("MAXIMA", None)
		raise
	return sympy.sympify(s)

# May integrate a spline
def integrate(*args, **kwargs):
	try:
		if has_maxima:
			try:
				ans = _integrate(*args, **kwargs)
				if not ans:
					raise EOFError
			except:
				pass
			else:
				return ans
		return sympy.integrate(*args, **kwargs)
	except ValueError:
		return sympy.integrate(*plotArgs(args), sympy.Symbol("x"))


def _dsolve(*args, **kwargs):
	proc = get_maxima()
	args = list(args)
	for i, arg in enumerate(args):
		for j in range(8):
			args[i] = arg.replace("_y", "'diff(y,x)")
	s = "grind(contrib_ode(" + ",".join(map(str, args)) + "=0),y,x)$"
	try:
		proc.stdin.write(s.encode("utf-8"))
		proc.stdin.flush()
		proc.stdout.readline()
		b = proc.stdout.readline().decode("utf-8", "replace")
		if b.startswith("incorrect syntax:"):
			raise SyntaxError(b)
		s = b.strip("$; \r\n").replace("%c", "0").replace("%e", "e")
	except:
		if proc.is_running():
			proc.kill()
		globals().pop("MAXIMA", None)
		raise
	return sympy.sympify(s)

def dsolve(*args, **kwargs):
	try:
		if has_maxima:
			try:
				ans = _dsolve(*args, **kwargs)
				if not ans:
					raise EOFError
			except:
				pass
			else:
				return ans
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
	if not os.path.exists("misc/ecm.exe") or os.path.getsize("misc/ecm.exe") < 4096:
		import requests
		with requests.get("https://cdn.discordapp.com/attachments/703579929840844891/1103723891815362600/ecm.exe") as resp:
			b = resp.content
		with open("misc/ecm.exe", "wb") as f:
			f.write(b)
else:
	if not os.path.exists("misc/ecm") or os.path.getsize("misc/ecm") < 4096:
		import requests
		with requests.get("https://cdn.discordapp.com/attachments/703579929840844891/1103729122909376562/ecm") as resp:
			b = resp.content
		with open("misc/ecm", "wb") as f:
			f.write(b)
		subprocess.run(("chmod", "777", "misc/ecm"))
o_factorint = sympy.factorint
_fcache = {}
def _factorint(n, **kwargs):
	timeout = kwargs.pop("timeout", None) or (kwargs["limit"] / 1000 + 1 if kwargs.get("limit") else None)
	try:
		s = str(n)
		if "." in s:
			raise TypeError
		if abs(int(s)) < 1 << 64:
			raise ValueError
	except (TypeError, ValueError):
		return o_factorint(n, **kwargs)
	try:
		return _fcache[s]
	except KeyError:
		pass
	args = ["misc/ecm", s, "2"]
	try:
		proc = subprocess.run(args, stdout=subprocess.PIPE, timeout=timeout)
	except PermissionError:
		if os.name == "nt":
			raise
		subprocess.run(("chmod", "777", "misc/ecm"))
		proc = subprocess.run(args, stdout=subprocess.PIPE, timeout=timeout)
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
sympy.factorint = _factorint

def factorize(*args, **kwargs):
	temp = _factorint(*args, **kwargs)
	return list(itertools.chain(*((k,) * v for k, v in sorted(temp.items()))))

def factors(n, lim=1048576):
	temp = _factorint(n)
	fcount = np.prod([x + 1 for x in temp.values()], dtype=object)
	if fcount > lim:
		raise OverflowError("Too many factors to evaluate.")
	pfact = list(itertools.chain(*((k,) * v for k, v in sorted(temp.items()))))
	facts = set(temp.keys())
	facts.add(1)
	facts.add(n)
	for i in range(2, sum(temp.values())):
		facts.update(np.prod(comb, dtype=object) for comb in itertools.combinations(pfact, i))
	return sorted(facts)

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
		if isinstance(x, (int, sympy.Integer)):
			return x
		y = int(x)
		if x == y:
			return y
	except:
		pass
	return x
round_min = rounder

def _unsafe(ufunc, *args, **kwargs):
	try:
		return ufunc(*args, **kwargs)
	except Exception as ex2:
		ex = ex2
		ex2 = str(ex2)
		if "has no attribute 'dtype'" in ex2:
			try:
				args = [np.asanyarray(a) for a in args]
			except:
				raise ex
			try:
				return ufunc(*args, **kwargs)
			except Exception as ex2:
				ex = ex2
				ex2 = str(ex2)
				pass
		if "casting rule" not in ex2 and "has no callable" not in ex2:
			raise
	kwargs["casting"] = "unsafe"
	try:
		return ufunc(*args, **kwargs)
	except TypeError as ex2:
		ex2 = str(ex2)
		if "unexpected keyword" not in ex2:
			raise ex
	kwargs.pop("casting", None)
	try:
		args = [a if not isinstance(a, np.ndarray) else np.asanyarray(a, np.float64) for a in args]
		kwargs = {k: (v if not isinstance(v, np.ndarray) else np.asanyarray(v, np.float64)) for k, v in kwargs.items()}
		return ufunc(*args, **kwargs)
	except:
		raise ex
autocast = lambda ufunc: lambda *args, **kwargs: _unsafe(ufunc, *args, **kwargs)


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
	"sleep": time.sleep,
	"roman_numerals": roman_numerals,
	"bf": _bf,
	"brainfuck": _bf,
	"random": Random,
	"rand": Random,
	"randint": Random,
	"dice": Random,
	"round_random": round_random,
	"plt": plot,
	"array": array,
	"predict": predict_next,
	"predict_next": predict_next,
	"sec2time": sec2time,
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
	"uniq": np.unique,
	"flip": np.flip,
	"reshape": np.reshape,
	"roll": np.roll,
	"rot90": np.rot90,
	"conjugate": np.conjugate,
	"sum": np.sum,
	"prod": np.prod,
	"max": np.nanmax,
	"min": np.nanmin,
	"argmax": np.argmax,
	"argmin": np.argmin,
	"ptp": np.ptp,
	"mean": np.mean,
	"median": np.median,
	"mode": scipy.stats.mode,
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
	"inv": autocast(np.linalg.inv),
	"matinv": autocast(np.linalg.inv),
	"pinv": autocast(np.linalg.pinv),
	"matpwr": np.linalg.matrix_power,
	"matrix_power": np.linalg.matrix_power,
	"einsum": np.einsum,
	"eig": autocast(np.linalg.eig),
	"eigvals": lambda a: np.asanyarray(sympy.Matrix(a).eigenvals(), dtype=object),
	"eigenvals": lambda a: np.asanyarray(sympy.Matrix(a).eigenvals(), dtype=object),
	"eigvects": lambda a: lambda a: np.asanyarray(sympy.Matrix(a).eigenvects(), dtype=object),
	"eigenvects": lambda a: lambda a: np.asanyarray(sympy.Matrix(a).eigenvects(), dtype=object),
	"svd": autocast(np.linalg.svd),
	"norm": autocast(np.linalg.norm),
	"cond": autocast(np.linalg.cond),
	"det": lambda a, method="bareiss": sympy.Matrix(a).det(method),
	"adj": autocast(np.matrix.getH),
	"adjoint": autocast(np.matrix.getH),
	"adjugate": lambda a: np.asanyarray(sympy.Matrix(a).adjugate(), dtype=object),
	"diagonalise": lambda a: np.asanyarray(sympy.Matrix(a).diagonalize(), dtype=object),
	"diagonalize": lambda a: np.asanyarray(sympy.Matrix(a).diagonalize(), dtype=object),
	"charpoly": lambda a, x="x": sympy.Matrix(a).charpoly(x),
	"cofactor": lambda a, method="berkowitz": np.asanyarray(sympy.Matrix(a).cofactor_matrix(method), dtype=object),
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
	"factorlist": factors,
	"factors": factors,
	"factorise": factorize,
	"factorize": factorize,
	"factor": sympy.factor,
	"factoreq": sympy.factor,
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
	"bool_": np.bool_,
	"uint8": np.uint8,
	"uint16": np.uint16,
	"uint32": np.uint32,
	"uint64": np.uint64,
	"int8": np.int8,
	"int16": np.int16,
	"int32": np.int32,
	"int64": np.int64,
	"float16": np.float16,
	"float32": np.float32,
	"float64": np.float64,
	"complex64": np.complex64,
	"complex128": np.complex128,
})
if hasattr(np, "ulonglong"):
	_globals["uint128"] = np.ulonglong
if hasattr(np, "longlong"):
	_globals["int128"] = np.longlong
if hasattr(np, "longdouble"):
	_globals["float80"] = np.longdouble
if hasattr(np, "clongdouble"):
	_globals["complex160"] = np.clongdouble
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
	random.seed(time.time_ns())
	BF_PREC = sympy.ceiling(int(prec) * 1.25)
	mp.dps = BF_PREC
	sys.set_int_max_str_digits(max(1024, BF_PREC * 256))
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
				i2 = i.doit()
			except:
				continue
			if i == i2 and str(i) == str(i2):
				continue
			try:
				f = f.subs(i, i2)
			except:
				continue
	# If the requested expression evaluates to a plot, return it
	if isinstance(f, Plot) or f is plt or type(f) is str:
		return (f,)
	try:
		f = sympy.simplify(f)
	except:
		pass
	# Solve any sums and round off floats when possible
	for i in sympy.preorder_traversal(f):
		if isinstance(i, (sympy.Number, float, np.floating)) and i != 0 and math.log10(abs(i)) < BF_PREC:
			try:
				f = f.subs(i, rounder(i))
			except:
				pass
		elif isinstance(f, (sympy.Integer, int, np.integer)) and i != 0 and math.log10(abs(i)) > BF_PREC:
			try:
				f = f.subs(i, sympy.N(f, prec))
			except:
				pass
		elif hasattr(i, "doit"):
			try:
				i2 = i.doit()
			except:
				continue
			if i == i2 and str(i) == str(i2):
				continue
			try:
				f = f.subs(i, i2)
			except:
				continue
	# Select list of answers to return based on the desired float precision level
	if isinstance(f, (str, bool, tuple, list, dict, np.ndarray)):
		return [f]
	if isinstance(f, (sympy.Float, sympy.Integer, float, int, np.number)):
		return [f]
	# if isinstance(f, sympy.Rational) and math.log10(max(f.p, f.q)) > prec and math.log10(f) <= prec:
	# 	return [f.evalf(prec, chop=True)]
	if prec:
		try:
			y = f.evalf(prec, chop=False)
		except:
			y = f
		try:
			e = rounder(y)
		except TypeError:
			e = y
			for i in sympy.preorder_traversal(e):
				if isinstance(i, (sympy.Float, float, np.floating)) and i != 0 and math.log10(abs(i)) < BF_PREC:
					e = e.subs(i, rounder(i))
		if r:
			p = prettyAns(f)
			if p == repr(e):
				return [f]
			return [f, p]
		p = prettyAns(f)
		f = repr(e)
		if re.fullmatch(f"-?[0-9]*\.[0-9]*0", f):
			return [f.rstrip(".0")]
		if len(f) > 1 and f[-1].isnumeric() and p.startswith(f[:-1]):
			return [e]
		if p == f:
			return [e]
		return [e, p]
	else:
		p = prettyAns(f)
		if p == repr(f) or len(p) > 1 and p[-1].isnumeric() and repr(f).startswith(p[:-1]):
			return [f]
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
		if isinstance(args, (str, bytes, memoryview)):
			args = literal_eval(base64.b64decode(args))
		resp = evalSym(*args)
		out = procResp(resp)
		sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_result({repr(out)})\n".encode("utf-8"))
	except Exception as ex:
		sys.stdout.buffer.write(f"~PROC_RESP[{ts}].set_exception(pickle.loads({repr(pickle.dumps(ex))}))\n".encode("utf-8"))
		sys.stdout.buffer.write(f"~print({args},{repr(traceback.format_exc())},sep='\\n',end='')\n".encode("utf-8"))
	sys.stdout.flush()


if __name__ == "__main__":
	def ensure_parent():
		parent = psutil.Process(os.getppid())
		while True:
			if not parent.is_running() or parent.status() == "zombie":
				p = psutil.Process()
				for c in p.children(True):
					c.terminate()
					try:
						c.wait(timeout=2)
					except psutil.TimeoutExpired:
						c.kill()
				p.terminate()
				break
			time.sleep(12)
	import threading
	threading.Thread(target=ensure_parent, daemon=True).start()
	while True:
		argv = sys.stdin.readline()
		if not argv:
			raise SystemExit
		argv = argv.rstrip()
		if argv[0] == "~":
			ts, args = argv[1:].split("~", 1)
			evaluate(ts, args.encode("ascii"))