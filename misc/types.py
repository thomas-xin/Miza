import ast
import asyncio
import bisect
import collections.abc
from collections import deque
import concurrent.futures
import contextlib
import copy
import datetime
import fractions
import functools
import io
from itertools import chain, repeat # noqa: F401
from math import ceil, floor, inf, nan
import json
import random
import re
import subprocess
import sys
try:
	sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
	pass
import time
from traceback import print_exc
import numpy as np
import orjson
from misc.ring_vector import RingVector

if not hasattr(time, "time_ns"):
	time.time_ns = lambda: int(time.time() * 1e9)

UNIQUE_TS = 0
def ts_us() -> int:
	global UNIQUE_TS
	ts = max(UNIQUE_TS + 1, time.time_ns())
	UNIQUE_TS = ts
	return ts
def utc():
	return time.time_ns() / 1000000000.0


class MemoryBytes:
	"""A memory-efficient wrapper for byte-like objects that provides bytes-like interface.
	This class wraps byte-like objects (bytes, bytearray, memoryview) and provides
	a consistent interface similar to bytes while maintaining memory efficiency by
	using memoryview internally. It lazily converts to bytes only when necessary.
	The class aims to mirror the immutable bytes API where practical. Operations
	that inherently require materializing new data (e.g. concatenation, repetition)
	will allocate new bytes objects, but read-only / predicate style queries avoid
	unnecessary copies and operate directly on the underlying memoryview.
	Attributes:
		view (memoryview): Direct access to the underlying memoryview object.
	Args:
		data (Union[bytes, bytearray, memoryview, MemoryBytes]): The byte-like data to wrap.
	Raises:
		TypeError: If the input is not a byte-like object.
	Examples:
		>>> mb = MemoryBytes(b'Hello')
		>>> mb[1:3]
		MemoryBytes(b'el')
		>>> mb.upper()
		MemoryBytes(b'HELLO')
		>>> mb.decode()
		'Hello'
	"""

	__slots__ = ("__weakref__", "_mv", "_b")

	def __init__(self, data):
		if not isinstance(data, byte_like):
			if isinstance(data, io.IOBase):
				if hasattr(data, "seek"):
					data.seek(0)
				data = data.read()
			else:
				raise TypeError(f"Expected byte_like, got {type(data).__name__}")
		if isinstance(data, MemoryBytes):
			data = data._mv
		if isinstance(data, bytes):
			self._b = data
		else:
			self._b = None
		if not isinstance(data, memoryview | bytearray):
			data = memoryview(data)
		self._mv = data

	def __getitem__(self, key):
		result = memoryview(self._mv)[key]
		return self.__class__(result) if isinstance(key, slice) else result

	def __setitem__(self, key, value):
		self._mv[key] = value

	def __bool__(self):
		return bool(len(self._mv))

	def __len__(self):
		return len(self._mv)

	def __bytes__(self):
		return self.tobytes()

	def tobytes(self):
		if self._b is None:
			self._b = bytes(self._mv)
		return self._b

	@property
	def view(self):
		return self._mv

	def __int__(self):
		return int(self._mv)

	def __float__(self):
		return float(self._mv)

	def __repr__(self):
		return f"{self.__class__.__name__}({self.tobytes()!r})"

	# --- core protocol helpers ---
	def __iter__(self):
		mv = self._mv
		for i in range(len(mv)):
			yield mv[i]

	def __reversed__(self):
		mv = self._mv
		for i in range(len(mv) - 1, -1, -1):
			yield mv[i]

	def __contains__(self, item):
		if isinstance(item, int):
			return 0 <= item <= 255 and any(b == item for b in self._mv)
		if isinstance(item, byte_like):
			return self.find(item) >= 0
		return False

	def __hash__(self):
		# Hash must be stable & match bytes semantics; rely on cached bytes when possible.
		return hash(self.tobytes())

	def __add__(self, other):
		if isinstance(other, byte_like):
			return self.__class__(self.tobytes() + (other.tobytes() if isinstance(other, MemoryBytes) else bytes(other)))
		return NotImplemented

	def __radd__(self, other):
		if isinstance(other, byte_like):
			return self.__class__((other.tobytes() if isinstance(other, MemoryBytes) else bytes(other)) + self.tobytes())
		return NotImplemented

	def __mul__(self, n):
		if isinstance(n, int):
			return self.__class__(self.tobytes() * n)
		return NotImplemented

	def __rmul__(self, n):
		return self.__mul__(n)

	def __lt__(self, other):
		if isinstance(other, byte_like):
			return self.tobytes() < (other.tobytes() if isinstance(other, MemoryBytes) else bytes(other))
		return NotImplemented

	def __le__(self, other):
		if isinstance(other, byte_like):
			return self.tobytes() <= (other.tobytes() if isinstance(other, MemoryBytes) else bytes(other))
		return NotImplemented

	def __gt__(self, other):
		if isinstance(other, byte_like):
			return self.tobytes() > (other.tobytes() if isinstance(other, MemoryBytes) else bytes(other))
		return NotImplemented

	def __ge__(self, other):
		if isinstance(other, byte_like):
			return self.tobytes() >= (other.tobytes() if isinstance(other, MemoryBytes) else bytes(other))
		return NotImplemented

	def __reduce__(self):
		# For pickling
		return (self.__class__, (self.tobytes(),))

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self._mv == other._mv
		if isinstance(other, byte_like):
			return self._mv == other
		return NotImplemented

	def __ne__(self, other):
		return not self.__eq__(other)

	def startswith(self, prefix, *args):
		return bool(self) and self[:len(prefix)] == prefix

	def endswith(self, suffix, *args):
		return bool(self) and self[-len(suffix):] == suffix

	def removeprefix(self, prefix, *args):
		if self.startswith(prefix):
			return self[len(prefix):]
		return self

	def removesuffix(self, suffix, *args):
		if self.endswith(suffix):
			return self[:-len(suffix)]
		return self

	def find(self, sub, *args):
		# Optimised path avoids full materialisation when possible.
		mv = self._mv
		start = args[0] if len(args) >= 1 else 0
		end = args[1] if len(args) >= 2 else len(mv)
		if start < 0:
			start += len(mv)
		if end < 0:
			end += len(mv)
		start = max(0, start)
		end = min(len(mv), end)
		if isinstance(sub, int):
			for i in range(start, end):
				if mv[i] == sub:
					return i
			return -1
		if self._b is not None:
			return self._b.find(sub, start, end)
		try:
			return sublist_index(mv[start:end], sub) + start
		except ValueError:
			return -1

	def rfind(self, sub, *args):
		mv = self._mv
		start = args[0] if len(args) >= 1 else 0
		end = args[1] if len(args) >= 2 else len(mv)
		if start < 0:
			start += len(mv)
		if end < 0:
			end += len(mv)
		start = max(0, start)
		end = min(len(mv), end)
		if isinstance(sub, int):
			for i in range(end - 1, start - 1, -1):
				if mv[i] == sub:
					return i
			return -1
		if self._b is not None:
			return self._b.rfind(sub, start, end)
		try:
			return sublist_rindex(mv[start:end], sub) + start
		except ValueError:
			return -1

	def index(self, sub, *args):
		if self._b is not None:
			return self._b.index(sub, *args)
		return sublist_index(self._mv, sub, *args)

	def rindex(self, sub, *args):
		if self._b is not None:
			return self._b.index(sub, *args)
		return sublist_rindex(self._mv, sub, *args)
	
	def count(self, sub, *args):
		return self.tobytes().count(sub, *args)

	def replace(self, old, new, count=-1):
		return self.__class__(self.tobytes().replace(old, new, count))

	def translate(self, table, delete=b''):
		return self.__class__(self.tobytes().translate(table, delete))

	def partition(self, sep):
		parts = self.tobytes().partition(sep)
		return tuple(self.__class__(part) for part in parts)

	def rpartition(self, sep):
		parts = self.tobytes().rpartition(sep)
		return tuple(self.__class__(part) for part in parts)

	def split(self, sep=None, maxsplit=-1):
		if self._b is not None or not sep:
			parts = self.tobytes().split(sep, maxsplit)
			return [self.__class__(part) for part in parts]
		parts = []
		i = 0
		while i < len(self) and (maxsplit < 0 or len(parts) < maxsplit):
			try:
				i2 = self.index(sep, i)
			except ValueError:
				break
			if i2 == i:
				i += len(sep)
				continue
			part = self[i:i2]
			parts.append(part)
			i = i2 + len(sep)
		part = self[i:]
		parts.append(part)
		return parts

	def rsplit(self, sep=None, maxsplit=-1):
		parts = self.tobytes().rsplit(sep, maxsplit)
		return [self.__class__(part) for part in parts]

	def splitlines(self, keepends=False):
		lines = self.tobytes().splitlines(keepends)
		return [self.__class__(line) for line in lines]

	def join(self, iterable):
		return self.__class__(self.tobytes().join(iterable))

	def strip(self, chars=None):
		return self.__class__(self.tobytes().strip(chars))

	def lstrip(self, chars=None):
		return self.__class__(self.tobytes().lstrip(chars))

	def rstrip(self, chars=None):
		return self.__class__(self.tobytes().rstrip(chars))

	def expandtabs(self, tabsize=8):
		return self.__class__(self.tobytes().expandtabs(tabsize))

	def capitalize(self):
		return self.__class__(self.tobytes().capitalize())

	def casefold(self):
		return self.__class__(self.tobytes().casefold())

	def center(self, width, fillchar=b' '):
		return self.__class__(self.tobytes().center(width, fillchar))

	def ljust(self, width, fillchar=b' '):
		return self.__class__(self.tobytes().ljust(width, fillchar))

	def rjust(self, width, fillchar=b' '):
		return self.__class__(self.tobytes().rjust(width, fillchar))

	def zfill(self, width):
		return self.__class__(self.tobytes().zfill(width))

	def swapcase(self):
		return self.__class__(self.tobytes().swapcase())

	def title(self):
		return self.__class__(self.tobytes().title())

	def upper(self):
		return self.__class__(self.tobytes().upper())

	def lower(self):
		return self.__class__(self.tobytes().lower())

	# --- predicates (ASCII-focused; fall back to bytes for full semantics) ---
	def isascii(self):
		return all(b < 128 for b in self._mv)

	def isalpha(self):
		mv = self._mv
		if not mv:
			return False
		for b in mv:
			if not (65 <= b <= 90 or 97 <= b <= 122):
				return False
		return True

	def isalnum(self):
		mv = self._mv
		if not mv:
			return False
		for b in mv:
			if not (48 <= b <= 57 or 65 <= b <= 90 or 97 <= b <= 122):
				return False
		return True

	def isdigit(self):
		mv = self._mv
		return bool(mv) and all(48 <= b <= 57 for b in mv)

	def isspace(self):
		# bytes.isspace considers ASCII whitespace characters
		mv = self._mv
		if not mv:
			return False
		for b in mv:
			if b not in (9, 10, 11, 12, 13, 32):
				return False
		return True

	def islower(self):
		mv = self._mv
		cased = False
		for b in mv:
			if 65 <= b <= 90:
				return False
			elif 97 <= b <= 122:
				cased = True
		return cased

	def isupper(self):
		mv = self._mv
		cased = False
		for b in mv:
			if 97 <= b <= 122:
				return False
			elif 65 <= b <= 90:
				cased = True
		return cased

	def istitle(self):
		# Simplistic ASCII titlecase check
		mv = self._mv
		if not mv:
			return False
		words = 0
		in_word = False
		seen_lower = False
		for b in mv:
			if 65 <= b <= 90 or 97 <= b <= 122:
				if not in_word:
					# new word, must start upper
					if not (65 <= b <= 90):
						return False
					in_word = True
					words += 1
				else:
					if 65 <= b <= 90:  # subsequent upper -> invalid
						return False
					seen_lower = True
			else:
				in_word = False
		return words > 0 and (seen_lower or len(mv) == 1)

	# --- convenience ---
	@property
	def nbytes(self):
		return self._mv.nbytes

	def copy(self):
		# Shallow copy shares underlying buffer
		return self.__class__(self._mv)

	def hex(self):
		if self._b is not None:
			return self._b.hex()
		# Avoid caching bytes: hex already produces new str so no need to persist _b
		return self._mv.tobytes().hex()

	@classmethod
	def fromhex(cls, s: str):
		return cls(bytes.fromhex(s))

	def encode(self, encoding=""):
		return self.tobytes()

	def decode(self, encoding="utf-8", errors="strict"):
		return self.tobytes().decode(encoding, errors)


def sublist_index(lst, sub, start=0, end=None):
	if not sub:
		return start
	if end is None:
		end = len(lst)
	if hasattr(lst, "index"):
		i = start
		while i < end - len(sub) + 1:
			try:
				i = lst.index(sub[0], i, end - len(sub) + 1)
			except ValueError:
				break
			if all(lst[i + j] == sub[j] for j in range(1, len(sub))):
				return i
			i += 1
	else:
		for i in range(start, end):
			if all(lst[i + j] == sub[j] for j in range(len(sub))):
				return i
	raise ValueError("Sublist not found.")

def sublist_rindex(lst, sub, start=0, end=None):
	if not sub:
		return start
	if end is None:
		end = len(lst)
	if hasattr(lst, "rindex"):
		i = end - len(sub)
		while i >= start:
			try:
				i = lst.rindex(sub[0], start, i + 1)
			except ValueError:
				break
			if all(lst[i + j] == sub[j] for j in range(1, len(sub))):
				return i
			i -= 1
	else:
		for i in range(end - len(sub), start - 1, -1):
			if all(lst[i + j] == sub[j] for j in range(len(sub))):
				return i
	raise ValueError("Sublist not found.")


class Dummy(BaseException):
	__slots__ = ("__weakref__",)
	def __bool__(self):
		return False

def as_str(s, encoding="utf-8"):
	if callable(getattr(s, "tobytes", None)):
		s = s.tobytes()
	if isinstance(s, (bytes, bytearray)):
		return s.decode(encoding, "replace")
	return str(s)
def as_bytes(b, encoding="utf-8"):
	if isinstance(b, str):
		b = b.encode(encoding)
	elif not isinstance(b, bytes):
		b = bytes(b)
	return b

# Creates a nested tuple from a nested list.
def _nested_tuple(a):
	return tuple(_nested_tuple(i) if isinstance(i, collections.abc.MutableSequence) else i for i in a)
def nested_tuple(a):
	return _nested_tuple(a) if isinstance(a, collections.abc.Sequence) and type(a) not in (str, bytes) and a[0] != a else a
int_like = int | np.integer
object_like = (object, np.object_)

def to_chunks(data, maxsize=None, count=None):
	data = memoryview(data)
	if count is not None:
		maxsize = maxsize or ceil(len(data) / count)
	elif maxsize is None:
		maxsize = len(data)
	while len(data):
		res, data = data[:maxsize], data[maxsize:]
		yield res

# Uses an optional interpolation mode to get a certain position in an iterable.
def get(v, i, mode=1):
	if isinstance(i, int):
		try:
			return v[i]
		except LookupError:
			return v[i % len(v)]
	size = len(v)
	i = i.real + i.imag * size
	if i == int(i) or mode == 0:
		return v[round(i) % size]
	elif mode > 0 and mode < 1:
		return get(v, i, 0) * (1 - mode) + mode * get(v, i, 1)
	elif mode == 1:
		a = floor(i)
		b = i - a
		return v[a % size] * (1 - b) + v[ceil(i) % size] * b
	return get(v, i, floor(mode)) * (1 - mode % 1) + (mode % 1) * get(v, i, ceil(mode))


alist = RingVector
def arange(*args, **kwargs):
	return alist(np.arange(*args, **kwargs, dtype=object))
def afull(size, n=0):
	return alist(np.full(size, n, dtype=object))
def azero(size):
	return alist(np.zeros(size, dtype=object))
def aempty(size):
	return alist(np.empty(size, dtype=object))

byte_like = bytes | bytearray | memoryview | MemoryBytes
list_like = list | tuple | set | frozenset | alist | np.ndarray
string_like = byte_like | str
number = int | float | np.number | fractions.Fraction
json_like = dict | list_like | str | number | bool | None


class cdict(dict):
	"""Class-like dictionary, with attributes corresponding to keys."""

	__slots__ = ("__weakref__",)

	@classmethod
	def from_object(cls, obj):
		return cls((a, getattr(obj, a, None)) for a in dir(obj))

	__init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs) # noqa: E731
	__repr__ = lambda self: self.__class__.__name__ + ("((" + ",".join("(" + ",".join(repr(i) for i in item) + ")" for item in super().items()) + ("," if len(self) == 1 else "") + "))") if self else "()" # noqa: E731
	__str__ = lambda self: super().__repr__() # noqa: E731
	__iter__ = lambda self: iter(tuple(super().__iter__())) # noqa: E731
	__call__ = lambda self, k: self.__getitem__(k) # noqa: E731

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if not k.startswith("__") or not k.endswith("__"):
			try:
				return self.__getitem__(k)
			except KeyError as ex:
				raise AttributeError(*ex.args)
		raise AttributeError(k)

	def __setattr__(self, k, v):
		if k.startswith("__") and k.endswith("__"):
			return object.__setattr__(self, k, v)
		return self.__setitem__(k, v)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(self)
		return data

	def union(self, other=None, **kwargs):
		temp = self.copy()
		if other:
			temp.update(other)
		if kwargs:
			temp.update(kwargs)
		return temp

	@property
	def __dict__(self):
		return self

	def ___repr__(self):
		return super().__repr__()

	def __copy__(self):
		return self.__class__(self)
	copy = __copy__

	def to_dict(self):
		return dict(**self)

	def to_list(self):
		return list(super().values())


class fdict(cdict):
	"""A dict with key-value pairs fed from more dict-like objects."""

	__slots__ = ("_feed",)

	def get_feed(self):
		feed = object.__getattribute__(self, "_feed")
		if callable(feed):
			return feed()
		return feed

	def _keys(self):
		found = set(super().keys())
		for f in self.get_feed():
			found.update(f)
		return found

	def keys(self):
		try:
			self.get_feed()
		except AttributeError:
			return super().keys()
		return self._keys()

	def __len__(self):
		return len(self.keys())

	def __iter__(self):
		return iter(super().keys())

	def _values(self):
		found = set()
		for k, v in super().items():
			found.add(k)
			yield v
		for f in self.get_feed():
			for k, v in f.items():
				if k not in found:
					found.add(k)
					yield v

	def values(self):
		try:
			self.get_feed()
		except AttributeError:
			return super().values()
		return self._values()

	def _items(self):
		found = set()
		for k, v in super().items():
			found.add(k)
			yield k, v
		for f in self.get_feed():
			for k, v in f.items():
				if k not in found:
					found.add(k)
					yield k, v

	def items(self):
		try:
			self.get_feed()
		except AttributeError:
			return super().items()
		return self._items()

	def __getitem__(self, k):
		try:
			return super().__getitem__(k)
		except KeyError:
			pass
		try:
			feed = self.get_feed()
		except AttributeError:
			feed = None
		if not feed:
			raise KeyError(k)
		for f in feed:
			try:
				return f.__getitem__(k)
			except KeyError:
				pass
		raise KeyError(k)

	def __setattr__(self, k, v):
		if k == "_feed" or k.startswith("__") and k.endswith("__"):
			return object.__setattr__(self, k, v)
		return self.__setitem__(k, v)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(self)
		try:
			self.get_feed()
		except AttributeError:
			return data
		for f in self.get_feed():
			data.update(f)
		return data

	def get(self, k, default=None):
		try:
			return self[k]
		except KeyError:
			return default

	def __contains__(self, k):
		if dict.__contains__(self, k):
			return True
		try:
			self.get_feed()
		except AttributeError:
			return False
		for f in self.get_feed():
			if f.__contains__(k):
				return True
		return False


class demap(collections.abc.Mapping):
	"""Double ended mapping, indexable from both sides."""

	__slots__ = ("__weakref__", "a", "b")

	def __init__(self, *args, **kwargs):
		self.a = cdict(*args, **kwargs)
		self.b = cdict(reversed(t) for t in self.a.items())

	def __getitem__(self, k):
		with suppress(KeyError):
			return self.a.__getitem__(k)
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
		with suppress(KeyError):
			return self.__getitem__(k)
		return v

	def pop(self, k, v=None):
		with suppress(KeyError):
			temp = self.__getitem__(k)
			self.__delitem__(k)
			return temp
		return v

	def popitem(self, k, v=None):
		with suppress(KeyError):
			temp = self.__getitem__(k)
			self.__delitem__(k)
			return (k, temp)
		return v

	clear = lambda self: (self.a.clear(), self.b.clear()) # noqa: E731
	__bool__ = lambda self: bool(self.a) # noqa: E731
	__iter__ = lambda self: iter(self.a.items()) # noqa: E731
	__reversed__ = lambda self: reversed(self.a.items()) # noqa: E731
	__len__ = lambda self: self.b.__len__() # noqa: E731
	__str__ = lambda self: self.a.__str__() # noqa: E731
	__repr__ = lambda self: f"{self.__class__.__name__}({self.a.__repr__() if bool(self.b) else ''})" # noqa: E731
	__contains__ = lambda self, k: k in self.a or k in self.b # noqa: E731
			
			
class UniversalSet(collections.abc.Set):
	"""The Universal Set. Contains everything."""

	__slots__ = ("__weakref__",)

	__str__ = lambda self: "ξ" # noqa: E731
	__repr__ = lambda self: f"{self.__class__.__name__}()" # noqa: E731
	__contains__ = lambda self, key: True # noqa: E731
	__bool__ = lambda self: True # noqa: E731
	__iter__ = lambda self: repeat(None) # noqa: E731
	__len__ = lambda self: inf # noqa: E731
	__call__ = lambda self, *args: self # noqa: E731
	__le__ = lambda self, other: type(self) is type(other) # noqa: E731
	__lt__ = lambda self, other: False # noqa: E731
	__eq__ = lambda self, other: type(self) is type(other) # noqa: E731
	__ne__ = lambda self, other: type(self) is not type(other) # noqa: E731
	__gt__ = lambda self, other: type(self) is not type(other) # noqa: E731
	__ge__ = lambda self, other: True # noqa: E731
	__and__ = lambda self, other: other # noqa: E731
	__or__ = lambda self, other: self # noqa: E731
	__sub__ = lambda self, other: self # noqa: E731
	__xor__ = lambda self, other: self # noqa: E731
	index = find = lambda self, obj: 0 # noqa: E731
	isdisjoint = lambda self, other: False # noqa: E731

universal_set = UniversalSet()


def exclusive_range(range, *excluded):
	ex = frozenset(excluded)
	return tuple(i for i in range if i not in ex)

def exclusive_set(range, *excluded):
	ex = frozenset(excluded)
	return frozenset(i for i in range if i not in ex)


def obj2range(r):
	if isinstance(r, number):
		return (r, r + 1)
	if isinstance(r, (range, slice)):
		return (r.start, r.stop)
	return tuple(r)

class RangeSet(collections.abc.Iterable):

	"""
	RangeSet is an ordered, normalized, and mutable collection of non-negative and/or negative integer
	indices represented internally as a union of half-open ranges [start, stop). It behaves like a
	set of integers while also supporting sequence-like operations such as iteration, indexing, and
	slicing across the flattened elements.
	Construction
	- RangeSet(): create an empty set.
	- RangeSet(start, stop): initialize with a single half-open interval [start, stop).
	- RangeSet(seq): if the first argument is list-like, initialize from an iterable of range-like
		objects (each element is converted to a (start, stop) pair via obj2range).
	Core semantics
	- Intervals are half-open: [start, stop), so stop is excluded.
	- Ranges are kept sorted and normalized (no overlaps or adjacency) by merging overlapping or
		touching intervals. Normalization happens via fixup() and is invoked by most mutating methods.
	- Membership:
		- int in rs: True if the integer is contained in any internal range.
		- range/slice in rs: True if the entire target interval is covered by a single internal range.
			Note: non-contiguous coverage across multiple disjoint ranges returns False.
	- Iteration yields contained integers in ascending order.
	- len(rs) returns the count of distinct integers represented by all ranges.
	- bool(rs) is True if the set is non-empty.
	Indexing and slicing
	- rs[i]: return the i-th smallest element (0-based) across the flattened set; raises IndexError
		if out of bounds.
	- rs[i:j:k]: return a new RangeSet consisting of elements selected by positional slicing across
		the flattened sequence of integers (not by numeric value range). The result is normalized.
	Mutating methods (all mutate in place and return self for chaining)
	- add(start, stop=None, fixup=True): add [start, stop). If stop is None, add the single element
		start (i.e., [start, start + 1)). No-op if the added interval is empty or already fully covered.
	- remove(start, stop=None): remove [start, stop) from the set. If stop is None, remove a single
		element at start. Supports splitting ranges when the removal cuts through the middle.
	- update(iterable): bulk-add. Each element can be a singleton (int) or a 2-item range-like
		(start, stop) if list-like.
	- difference_update(iterable): bulk-remove. Each element can be a singleton (int) or a 2-item
		range-like (start, stop) if list-like.
	- fixup(): normalize by merging overlapping or adjacent intervals; typically not needed to be
		called directly unless add(..., fixup=False) was used.
	Class methods
	- RangeSet.parse(slices, size): Build a RangeSet from an iterable of slice-like specifications
		relative to a bounded domain [0, size). Each element in 'slices' can be:
		- [n] or (n,): a single index; supports negative indices like Python.
		- [start, stop]: a half-open interval; None is allowed and interpreted like normal slice bounds;
			negative indices are allowed.
		- [start, stop, step]: a full slice; step of +1 or -1 becomes a contiguous range (after
			normalization of direction). For |step| > 1, the result is a union of singletons (potentially
			merged if they become adjacent due to other operations).
		Out-of-bounds and empty selections are safely ignored/clamped via slice.indices(size).
	Complexity notes
	- Membership check for ints is O(log m + t), where m is the number of stored ranges and t is
		the number of candidate ranges to examine (typically small).
	- add/remove are O(m) in the number of ranges due to insertion and potential merging/splitting.
	- Indexing by position is O(m) in the worst case; slicing by positions is O(N) where N is the
		number of returned elements (plus merging cost).
	Examples
	- Build and mutate:
			rs = RangeSet()
			rs.add(2, 5).add(7).add(5, 7)   # becomes [2, 8)
			assert 3 in rs                   # True
			assert range(2, 8) in rs         # True (covered by a single internal range)
			assert range(2, 9) in rs is False
	- Sequence-like behavior:
			list(rs)                         # [2, 3, 4, 5, 6, 7]
			rs[0]                            # 2
			rs[1:4]                          # RangeSet representing {3, 4, 5}, i.e., [3, 6)
	- Removal and splitting:
			rs.remove(4, 10)                 # leaves [2, 4)
	- Parsing:
			rs = RangeSet.parse([[None, 5], [7, None], [9, 1, -2]], size=10)
			# Interprets slice-like specs relative to [0, 10). Negative indices and None are supported.
	Notes
	- All mutating methods return self for fluent chaining.
	- The containment check for range/slice requires the interval to be fully covered by a single
		internal range, not a union of disjoint ranges.
	"""

	__slots__ = ("ranges",)

	def __init__(self, start=0, stop=0):
		if isinstance(start, list_like):
			self.ranges = list(map(obj2range, start))
		else:
			self.ranges = []
			self.add(start, stop)

	def __repr__(self):
		return self.__class__.__name__ + "(" + repr(self.ranges) + ")"

	def __contains__(self, key):
		if isinstance(key, slice | range):
			start, stop = (key.start, key.stop) if key.step >= 0 else ((k2 := range(key.start, key.stop, key.step)[::-1]).start, k2.stop)
			idx = bisect.bisect_left(self.ranges, start, key=lambda x: x[1])
			for i in range(idx, len(self.ranges)):
				left, right = self.ranges[i]
				if left > start:
					break
				if right > stop:
					return True
				i += 1
			return False
		idx = bisect.bisect_left(self.ranges, key, key=lambda x: x[1])
		for i in range(idx, len(self.ranges)):
			left, right = self.ranges[i]
			if left > key:
				break
			if right > key:
				return True
			i += 1
		return False

	def __len__(self):
		return sum(right - left for left, right in self.ranges)

	def __bool__(self):
		return bool(self.ranges)

	def __iter__(self):
		for left, right in self.ranges:
			if right == left + 1:
				yield left
			else:
				yield from range(left, right)

	def __getitem__(self, k):
		if isinstance(k, slice):
			k = range(*k.indices(len(self)))
			out = self.__class__()
			for i, x in enumerate(self):
				if i in k:
					out.add(x, fixup=False)
			return out.fixup()
		if not isinstance(k, int):
			raise TypeError(k)
		i = 0
		for r in self.ranges:
			if k - i < r[1] - r[0]:
				return k - i + r[0]
			i += r[1] - r[0]
		raise IndexError(k)

	def fixup(self):
		i = 0
		while i < len(self.ranges) - 1:
			l1, r1 = self.ranges[i]
			l2, r2 = self.ranges[i + 1]
			if r1 >= l2:
				self.ranges[i] = (l1, max(r1, r2))
				self.ranges.pop(i + 1)
			else:
				i += 1
		return self

	def add(self, start, stop=None, fixup=True):
		if stop is None:
			stop = start + 1
		assert start <= stop, "Start must be less than or equal to stop."
		if start == stop or range(start, stop) in self:
			return self
		bisect.insort_right(self.ranges, (start, stop))
		if fixup:
			self.fixup()
		return self

	def remove(self, start, stop=None):
		if stop is None:
			stop = start + 1
		assert start <= stop, "Start must be less than or equal to stop."
		if start == stop:
			return self
		i = 0
		while i < len(self.ranges):
			left, right = self.ranges[i]
			if left >= stop:
				break
			if right <= start:
				i += 1
				continue
			if left >= start:
				left = stop
				if left >= right:
					self.ranges.pop(i)
					continue
				self.ranges[i] = (left, right)
				i += 1
				continue
			if right <= stop:
				right = start
				self.ranges[i] = (left, right)
				i += 1
				continue
			l1, r1 = left, start
			l2, r2 = stop, right
			self.ranges[i] = (l1, r1)
			self.ranges.insert(i + 1, (l2, r2))
			i += 2
		return self

	def update(self, others):
		for obj in others:
			if isinstance(obj, list_like):
				self.add(*obj)
			else:
				self.add(obj)
		return self

	def difference_update(self, others):
		for obj in others:
			if isinstance(obj, list_like):
				self.remove(*obj)
			else:
				self.remove(obj)
		return self

	@classmethod
	def parse(cls, slices, size):
		self = cls()
		for spl in slices:
			if not spl:
				continue
			spl = list(spl)
			if spl[0] is None:
				spl[0] = 0
			if len(spl) >= 2 and spl[1] is None:
				spl[1] = size
			if len(spl) == 1:
				n = spl[0]
				if n >= size or n < -size:
					continue
				if n < 0:
					self.add(n + size)
				else:
					self.add(n)
			elif len(spl) >= 3:
				target = range(*slice(*spl).indices(size))
				if target.step == 1:
					self.add(target.start, target.stop)
				elif target.step == -1:
					target = target[::-1]
					self.add(target.start, target.stop)
				elif not self.ranges:
					self.ranges = [(n, n + 1) for n in target]
				else:
					for n in target:
						self.add(n, fixup=False)
					self.fixup()
			else:
				spl = [x if x >= 0 else x + size for x in spl]
				self.add(*slice(*sorted(spl)).indices(size)[:2])
		return self


# Experimental invisible Zero-Width character encoder.
ZeroEnc = "\xad\u061c\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\u2065\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\ufe0f\ufeff"
__zeroEncoder = demap({chr(i + 97): c for i, c in enumerate(ZeroEnc)})
__zeroEncode = "".maketrans(dict(__zeroEncoder.a))
__zeroDecode = "".maketrans(dict(__zeroEncoder.b))
def is_zero_enc(s):
	return s[0] in ZeroEnc if s else None
def zwencode(s):
	return as_str(s).casefold().translate(__zeroEncode)
def zwdecode(s):
	return as_str(s).casefold().translate(__zeroDecode)
__zeroRemover = {c: "" for c in ZeroEnc}
__zeroRemoverTrans = "".maketrans(__zeroRemover)
def zwremove(s):
	return as_str(s).translate(__zeroRemoverTrans)


# Unicode fonts for alphanumeric characters.
UNIFMTS = [
	"𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙",
	"𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝓪𝓫𝓬𝓭𝓮𝓯𝓰𝓱𝓲𝓳𝓴𝓵𝓶𝓷𝓸𝓹𝓺𝓻𝓼𝓽𝓾𝓿𝔀𝔁𝔂𝔃𝓐𝓑𝓒𝓓𝓔𝓕𝓖𝓗𝓘𝓙𝓚𝓛𝓜𝓝𝓞𝓟𝓠𝓡𝓢𝓣𝓤𝓥𝓦𝓧𝓨𝓩",
	"𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝒶𝒷𝒸𝒹𝑒𝒻𝑔𝒽𝒾𝒿𝓀𝓁𝓂𝓃𝑜𝓅𝓆𝓇𝓈𝓉𝓊𝓋𝓌𝓍𝓎𝓏𝒜𝐵𝒞𝒟𝐸𝐹𝒢𝐻𝐼𝒥𝒦𝐿𝑀𝒩𝒪𝒫𝒬𝑅𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵",
	"𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡𝕒𝕓𝕔𝕕𝕖𝕗𝕘𝕙𝕚𝕛𝕜𝕝𝕞𝕟𝕠𝕡𝕢𝕣𝕤𝕥𝕦𝕧𝕨𝕩𝕪𝕫𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ",
	"0123456789𝔞𝔟𝔠𝔡𝔢𝔣𝔤𝔥𝔦𝔧𝔨𝔩𝔪𝔫𝔬𝔭𝔮𝔯𝔰𝔱𝔲𝔳𝔴𝔵𝔶𝔷𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ",
	"0123456789𝖆𝖇𝖈𝖉𝖊𝖋𝖌𝖍𝖎𝖏𝖐𝖑𝖒𝖓𝖔𝖕𝖖𝖗𝖘𝖙𝖚𝖛𝖜𝖝𝖞𝖟𝕬𝕭𝕮𝕯𝕰𝕱𝕲𝕳𝕴𝕵𝕶𝕷𝕸𝕹𝕺𝕻𝕼𝕽𝕾𝕿𝖀𝖁𝖂𝖃𝖄𝖅",
	"０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
	#"0123456789ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘQʀꜱᴛᴜᴠᴡxʏᴢᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘQʀꜱᴛᴜᴠᴡxʏᴢ",
	"⓪①②③④⑤⑥⑦⑧⑨🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉",
	"⓿➊➋➌➍➎➏➐➑➒🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉",
	"⓪①②③④⑤⑥⑦⑧⑨ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ",
	"⓿➊➋➌➍➎➏➐➑➒🅐🅑🅒🅓🅔🅕🅖🅗🅘🅙🅚🅛🅜🅝🅞🅟🅠🅡🅢🅣🅤🅥🅦🅧🅨🅩🅐🅑🅒🅓🅔🅕🅖🅗🅘🅙🅚🅛🅜🅝🅞🅟🅠🅡🅢🅣🅤🅥🅦🅧🅨🅩",
	"0123456789𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡",
	"𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕",
	"𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉",
	"₀₁₂₃₄₅₆₇₈₉ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖqʳˢᵗᵘᵛʷˣʸᶻ🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿",
	"0123456789ᗩᗷᑢᕲᘿᖴᘜᕼᓰᒚҠᒪᘻᘉᓍᕵᕴᖇSᖶᑘᐺᘺ᙭ᖻᗱᗩᗷᑕᗪᗴᖴǤᕼIᒍKᒪᗰᑎOᑭᑫᖇᔕTᑌᐯᗯ᙭Yᘔ",
	"0ƖᘔƐᔭ59Ɫ86ɐqɔpǝɟɓɥᴉſʞןɯuodbɹsʇnʌʍxʎzꓯᗺƆᗡƎℲ⅁HIſꓘ⅂WNOԀΌᴚS⊥∩ΛMX⅄Z",
	"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
]
__umap = {UNIFMTS[k][i]: UNIFMTS[-1][i] for k in range(len(UNIFMTS) - 1) for i in range(len(UNIFMTS[k]))}

__unfont = "".maketrans(__umap)
def unfont(s):
	return str(s).translate(__unfont)

DIACRITICS = {
	"ÀÁÂÃÄÅĀĂĄΑӐӒᎪ": "A",
	"ÆӔ": "AE",
	"ΒᏴᛒ": "B",
	"ÇĆĈĊČϹҪᏟⅭ": "C",
	"ĎĐᎠⅮ": "D",
	"ÈÉÊËĒĔĖĘĚΕЀЁҼҾӖᎬ": "E",
	"Ϝ": "F",
	"ĜĞĠĢԌᏀ": "G",
	"ĤĦΗҢҤԨᎻ": "H",
	"IÌÍÎÏĨĪĬĮİΙЇӀⅠ": "I",
	"Ĳ": "IJ",
	"ĴЈᎫ": "J",
	"ĶΚҚҜҞҠᏦᛕK": "K",
	"ĹĻĽĿŁᏞⅬ": "L",
	"ΜϺМᎷᛖⅯ": "M",
	"ÑŃŅŇŊΝ": "N",
	"ÒÓÔÕÖØŌŎŐΟӦՕ": "O",
	"Œ": "OE",
	"ΡҎᏢ": "P",
	"ႭႳ": "Q",
	"ŔŖŘᏒᚱ": "R",
	"SŚŜŞŠՏႽᏚ𐐠": "S",
	"ŢŤŦΤҬᎢ": "T",
	"ÙÚÛÜŨŪŬŮŰŲԱՍ⋃": "U",
	"ѴᏙⅤ": "V",
	"ŴᎳ": "W",
	"ΧҲӼӾⅩ": "X",
	"ÝŶŸΥЎУҰӮӰӲ": "Y",
	"ŹŻŽΖᏃ": "Z",
	"àáâãäåāăąǎɑαӑӓ": "a",
	"æӕ": "ae",
	"ʙβЬв": "b",
	"çćĉċčϲҫⅽ": "c",
	"ďđԁժḍⅾ": "d",
	"èéêëðēĕėęěѐёҽҿӗ": "e",
	"ĝğġģɡɢ": "g",
	"ĥħʜнңҥһԧԩ": "h",
	"iìíîïĩīĭįıǐɩїاᎥⅰ": "i",
	"ĳ": "ij",
	"ĵϳј": "j",
	"ķĸκқҝҟҡ": "k",
	"ĺļľŀłʟιاⅼ": "l",
	"мⅿ": "m",
	"ñńņňŉŋɴ": "n",
	"òóôõöøōŏőǒοӧ": "o",
	"œ": "oe",
	"þρҏṕ": "p",
	"ŕŗřʀԻ": "r",
	"sśŝşšſ": "s",
	"ß": "ss",
	"ţťŧτтҭ": "t",
	"ùúûüũūŭůűųǔμυ": "u",
	"νѵⅴ": "v",
	"ŵѡ": "w",
	"χҳӽӿⅹ": "x",
	"ýÿŷʏγўүұӯӱӳ": "y",
	"źżž": "z",
}
for i, k in DIACRITICS.items():
	__umap.update({c: k for c in i})
__umap.update({c: "" for c in ZeroEnc})
__umap["\u200a"] = ""
for c in tuple(__umap):
	if c in UNIFMTS[-1]:
		__umap.pop(c)
__trans = "".maketrans(__umap)
zalgo_array = np.array(list(
	RangeSet([
		(768, 880),
		(1155, 1162),
		(1425, 1478),
		(1552, 1560),
		(1619, 1632),
		(1750, 1774),
		(2260, 2304),
		(7616, 7627),
		8432,
		(11744, 11776),
		42607, (42612, 42622), (42654, 42655),
		(65056, 65060),
	]).difference_update([
		1470, 1472, 1475,
		1757, 1758, 1765, 1766, 1769,
		2274,
	])
))
zalgo_map = {n: "" for n in zalgo_array}
__trans.update(zalgo_map)
__unitrans = ["".maketrans({UNIFMTS[-1][x]: UNIFMTS[i][x] for x in range(len(UNIFMTS[-1]))}) for i in range(len(UNIFMTS) - 1)]

# Translates all alphanumeric characters in a string to their corresponding character in the desired font.
def uni_str(s, fmt=0):
	if type(s) is not str:
		s = str(s)
	return s.translate(__unitrans[fmt])

# Translates all alphanumeric characters in unicode fonts to their respective ascii counterparts.
def unicode_prune(s):
	if type(s) is not str:
		s = str(s)
	if s.isascii():
		return s
	return s.translate(__trans)

def word_count(s):
	return 1 + sum(1 for _ in regexp("\\W+").finditer(s))
def single_space(s):
	return regexp("\\s\\s+").sub(" ", s)
def to_alphanumeric(string):
	return single_space(regexp("[^a-z 0-9]+", re.I).sub(" ", unicode_prune(string)))

__qmap = {
	"“": '"',
	"”": '"',
	"„": '"',
	"‘": "'",
	"’": "'",
	"‚": "'",
	"〝": '"',
	"〞": '"',
	"⸌": "'",
	"⸍": "'",
	"⸢": "'",
	"⸣": "'",
	"⸤": "'",
	"⸥": "'",
}
__qtrans = "".maketrans(__qmap)
for i in range(0xe0000, 0xe1000):
	__qtrans[i] = ""

def full_prune(s):
	return unicode_prune(s).translate(__qtrans).casefold()


class fcdict(cdict):
	"""A full-casefold string lookup mapping object."""

	__slots__ = ()

	def __init__(self, *args, **kwargs):
		return super().__init__((full_prune(k), v) for k, v in dict(*args, **kwargs).items())

	def __contains__(self, k):
		return super().__contains__(k) or super().__contains__(full_prune(k))

	def __setitem__(self, k, v):
		return super().__setitem__(full_prune(k), v)

	def __getitem__(self, k):
		return super().__getitem__(full_prune(k))

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if not k.startswith("__") or not k.endswith("__"):
			with suppress(KeyError):
				return super().__getitem__(k)
			return self.__getitem__(k)
		raise AttributeError(k)

	def update(self, other):
		super().update(dict((full_prune(k), v) for k, v in other.items()))

	def get(self, k, default=None):
		try:
			return self[k]
		except KeyError:
			return default

	def pop(self, k, default=Dummy):
		try:
			return super().pop(full_prune(k))
		except KeyError:
			if default is not Dummy:
				return default
			raise

	def popitem(self, k, default=Dummy):
		try:
			return super().popitem(full_prune(k))
		except KeyError:
			if default is not Dummy:
				return default
			raise


class mdict(cdict):
	"""Dictionary with multiple assignable values per key."""

	__slots__ = ()

	def count(self):
		return sum(len(v) for v in super().values())

	def extend(self, k, v):
		try:
			values = super().__getitem__(k)
		except KeyError:
			return super().__setitem__(k, alist(v).uniq(sort=False))
		return values.extend(v).uniq(sort=False)

	def append(self, k, v):
		values = super().setdefault(k, alist())
		if v not in values:
			values.append(v)
	add = insert = append

	def popleft(self, k):
		values = super().__getitem__(k)
		if len(values):
			v = values.popleft()
		else:
			v = None
		if not values:
			super().pop(k)
		return v

	def popright(self, k):
		values = super().__getitem__(k)
		if len(values):
			v = values.popright()
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


class msdict(cdict):
	"""Dictionary with multiple assignable values per key. Uses sets."""

	__slots__ = ()

	def count(self):
		return sum(len(v) for v in super().values())

	def extend(self, k, v):
		try:
			values = super().__getitem__(k)
		except KeyError:
			return super().__setitem__(k, set(v))
		return values.update(v)

	def append(self, k, v):
		values = super().setdefault(k, set())
		if v not in values:
			values.add(v)

	add = append

	def popleft(self, k):
		values = super().__getitem__(k)
		if len(values):
			v = values.pop()
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


def json_default(obj):
	if isinstance(obj, datetime.datetime):
		return obj.strftime("%Y-%m-%dT%H:%M:%S")
	if isinstance(obj, np.number):
		return obj.item()
	if isinstance(obj, (set, frozenset, alist, deque, np.ndarray)):
		return list(obj)
	raise TypeError(obj)

class MultiEncoder(json.JSONEncoder):

	def default(self, obj):
		return json_default(obj)

def json_dumps(obj, *args, **kwargs):
	return orjson.dumps(obj, *args, default=json_default, **kwargs)
def json_dumpstr(obj, *args, **kwargs):
	return orjson.dumps(obj, *args, default=json_default, **kwargs).decode("utf-8", "replace")

class PrettyJSONEncoder(json.JSONEncoder):

	def __init__(self, *args, **kwargs):
		self.indent = kwargs.get("indent") or "\t"
		super().__init__(*args, **kwargs)

	def encode(self, obj, level=0):
		indent = " " * self.indent if type(self.indent) is int else self.indent
		curr_indent = indent * level
		next_indent = indent * (level + 1)
		if isinstance(obj, (list, tuple)):
			if all(not isinstance(x, (tuple, list, dict)) or len(json_dumps(x)) < 10 for x in obj):
				return "[" + ", ".join(json_dumpstr(x) for x in obj) + "]"
			items = [self.encode(x, level=level + 1) for x in obj]
			return "[\n" + next_indent + f",\n{next_indent}".join(item for item in items) + f"\n{curr_indent}" + "]"
		elif isinstance(obj, dict):
			if all(type(x) is str and len(x) <= max(10, len(obj)) for x in obj.values()) and all(type(x) is str and len(x) <= max(10, len(obj)) for x in obj.keys()):
				return json.dumps(obj)
			items = [f"{json_dumpstr(k)}: {self.encode(v, level=level + 1)}" for k, v in obj.items()]
			items.sort()
			return "{\n" + next_indent + f",\n{next_indent}".join(item for item in items) + f"\n{curr_indent}" + "}"
		return json_dumpstr(obj)

	def default(self, obj):
		return json_default(obj)

prettyjsonencoder = PrettyJSONEncoder(indent="\t")
pretty_json = lambda obj: prettyjsonencoder.encode(obj)

def require_hashable(k) -> collections.abc.Hashable:
	if isinstance(k, list_like):
		return tuple(map(require_hashable, k))
	if isinstance(k, collections.abc.Hashable):
		return k
	try:
		return json_dumps(k)
	except orjson.JSONDecodeError:
		pass
	return repr(k)

def hashable_args(func) -> collections.abc.Callable:
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		hashable_args = tuple(map(require_hashable, args))
		hashable_keys = kwargs.keys()
		hashable_values = tuple(map(require_hashable, kwargs.values()))
		hashable_kwargs = dict(zip(hashable_keys, hashable_values))
		return func(*hashable_args, **hashable_kwargs)
	return wrapper

def always_copy(func) -> collections.abc.Callable:
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		return copy.deepcopy(func(*args, **kwargs))
	return wrapper


# Rounds a number to the nearest integer, with a probability determined by the fractional part.
def round_random(x) -> int:
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

def round_min(x) -> number:
	"Casts a number to integer if the conversion would not alter the value."
	if x is None:
		return x
	if isinstance(x, int):
		return x
	if isinstance(x, str):
		if not x:
			return nan
		if "/" in x:
			p, q = x.split("/")
			x = float(p) / float(q)
		elif "." in x:
			x = float(x)
		else:
			try:
				return int(x)
			except ValueError:
				x = float(x)
	if isinstance(x, np.number):
		x = x.item()
	if not hasattr(x, "is_integer"):
		return int(x) if x == int(x) else x
	elif x.is_integer():
		return int(x)
	return x

@functools.lru_cache(maxsize=64)
def try_int(i) -> int | string_like:
	if isinstance(i, str) and not i.isnumeric():
		return i
	try:
		return int(i)
	except (TypeError, ValueError):
		return i

def cast_id(i) -> int:
	try:
		return int(i)
	except (TypeError, ValueError):
		return i.id

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

@functools.lru_cache(maxsize=64)
def safe_eval(s):
	"""Safely evaluates a string as a Python expression. Builtins and private attributes are stripped."""
	a = ast.parse(s, mode="eval")
	class BannedAttributes(ast.NodeTransformer):
		def visit_Attribute(self, node):
			if isinstance(node.attr, str):
				if node.attr.startswith("_") or node.attr.endswith("_"):
					node.attr = node.attr.strip("_")
				return node
			raise AttributeError(node.attr)
	BannedAttributes().visit(a)
	code = compile(a, filename="<math-input>", mode="eval")
	return eval(code, {}, eval_const)

def nop(*void1, **void2) -> None:
	return None
def nofunc(arg, *void1, **void2):
	return arg
def none(it) -> bool:
	return not any(it)

def literal_eval(s):
	return ast.literal_eval(as_str(s).lstrip())

def is_exception(e):
	return isinstance(e, BaseException) or isinstance(e, type) and issubclass(e, BaseException)

def coerce(data, k, cls=None, default=Dummy):
	"""Coerces a value in a dictionary to a specified type. The type should be a callable that returns the coerced value."""
	try:
		v = data[k]
	except LookupError:
		if default is Dummy:
			raise
		v = default
	if v is not cls:
		if not isinstance(cls, type) or not isinstance(v, cls):
			v = data[k] = cls(v)
	return v

def coercedefault(data, k, cls=None, default=Dummy):
	"""Coerces a value in a dictionary to a specified type. The type should be a callable that returns the coerced value. Unlike coerce, this function will also set the default value if the key is not found."""
	try:
		v = data[k]
	except LookupError:
		if default is Dummy:
			raise
		v = data[k] = default
	if v is not cls:
		if not isinstance(cls, type) or not isinstance(v, cls):
			v = data[k] = cls(v)
	return v

def updatedefault(original, updates):
	for k, v in updates.items():
		if isinstance(v, dict) and isinstance(original.get(k), dict):
			updatedefault(original[k], v)
		else:
			original[k] = v
	return original

# For compatibility with versions of asyncio and concurrent.futures that have the exceptions stored in a different module
T0 = TimeoutError
try:
	T1 = asyncio.exceptions.TimeoutError
except AttributeError:
	try:
		T1 = asyncio.TimeoutError
	except AttributeError:
		T1 = TimeoutError
try:
	T2 = concurrent.futures._base.TimeoutError
except AttributeError:
	try:
		T2 = concurrent.futures.TimeoutError
	except AttributeError:
		T2 = TimeoutError

try:
	ISE = asyncio.exceptions.InvalidStateError
except AttributeError:
	ISE = asyncio.InvalidStateError
try:
	ISE2 = concurrent.futures._base.InvalidStateError
except AttributeError:
	ISE2 = concurrent.futures.InvalidStateError
try:
	CE = asyncio.exceptions.CancelledError
except AttributeError:
	CE = asyncio.CancelledError
try:
	CE2 = concurrent.futures._base.CancelledError
except AttributeError:
	CE2 = concurrent.futures.CancelledError
TE = subprocess.TimeoutExpired
CPE = subprocess.CalledProcessError


class ArgumentError(LookupError):
	pass
class DomainError(ArgumentError, OverflowError):
	pass
class EnumError(ArgumentError):
	pass
class TooManyRequests(PermissionError):
	pass
class CommandCancelledError(Exception):
	pass
class DisconnectedChannelError(LookupError):
	pass

AE = ArgumentError
DE = DomainError
EE = EnumError
TMR = TooManyRequests
CCE = CommandCancelledError
DCE = DisconnectedChannelError

def getattr_chain(obj, attrs, default=Dummy):
	if not attrs:
		return obj
	for attr in attrs.split("."):
		obj = getattr(obj, attr, default)
	if obj is Dummy:
		raise AttributeError(attrs)
	return obj
def all_none(*args):
	for obj in args:
		if obj is not None:
			return False
	return True
def no_none(*args):
	for obj in args:
		if obj is None:
			return False
	return True

class T(object):
	"""
	A wrapper class that provides various utility methods for an object.
	Attributes:
		obj: The object being wrapped.
	Methods:
		if_instance(t, func):
			Executes a function if the wrapped object is an instance of the specified type.
		if_not_instance(t, func):
			Executes a function if the wrapped object is not an instance of the specified type.
		if_is(t, func):
			Executes a function if the wrapped object is the specified object.
		if_is_not(t, func):
			Executes a function if the wrapped object is not the specified object.
		get(k, default=None):
			Retrieves an attribute or item from the wrapped object, with a default value if not found.
		__getitem__(k):
			Retrieves an item from the wrapped object using the indexing operator.
		__setitem__(k, v):
			Sets an item in the wrapped object using the indexing operator.
		coerce(k, cls=None, default=Dummy):
			Coerces the value of the specified key to a given class, with a default value if not found.
		coercedefault(k, cls=None, default=Dummy):
			Coerces the value of the specified key to a given class, with a default value if not found.
		updatedefault(other):
			Updates the wrapped object with values from another object, using default values if necessary.
	"""

	__slots__ = ("obj",)

	def __init__(self, obj):
		self.obj = obj

	def if_instance(self, t, func):
		obj = self.obj
		if isinstance(obj, t):
			if isinstance(func, str):
				func = getattr(obj, func)
			func(self)
		return self

	def if_not_instance(self, t, func):
		obj = self.obj
		if not isinstance(obj, t):
			if isinstance(func, str):
				func = getattr(obj, func)
			func(self)
		return self

	def if_is(self, t, func):
		obj = self.obj
		if obj is t:
			if isinstance(func, str):
				func = getattr(obj, func)
			func(self)
		return self

	def if_is_not(self, t, func):
		obj = self.obj
		if obj is not t:
			if isinstance(func, str):
				func = getattr(obj, func)
			func(self)
		return self

	def get(self, k, default=None):
		try:
			getter = self.obj.__getitem__
		except AttributeError:
			pass
		else:
			try:
				return getter(k)
			except LookupError:
				return default
			except TypeError:
				pass
		return getattr(self.obj, k, default)

	def __getitem__(self, k):
		try:
			getter = self.obj.__getitem__
		except AttributeError:
			pass
		else:
			return getter(k)
		return getattr(self.obj, k)

	def __setitem__(self, k, v):
		self.obj[k] = v

	def coerce(self, k, cls=None, default=Dummy):
		return coerce(self, k, cls, default)

	def coercedefault(self, k, cls=None, default=Dummy):
		return coercedefault(self, k, cls, default)

	def updatedefault(self, other):
		return updatedefault(self, other)

class TracebackSuppressor(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager, contextlib.ContextDecorator, collections.abc.Callable):
	"""A context manager that suppresses specified exceptions and optionally traces them.
	This class implements both synchronous and asynchronous context management,
	can be used as a decorator, and can be called to create new instances.
	Args:
		*args: Variable length argument list of exception types to suppress.
		fn (callable, optional): Function to call when an unhandled exception occurs. 
			Defaults to print_exc.
		**kwargs: Arbitrary keyword arguments of additional exception types to suppress.
	Examples:
		>>> # As a context manager
		>>> with TracebackSuppressor(ValueError, ZeroDivisionError):
		...     1/0  # This exception will be suppressed
		>>> # As a decorator
		>>> @TracebackSuppressor(ValueError)
		... def my_function():
		...     raise ValueError()
		>>> # As an async context manager
		>>> async with TracebackSuppressor(ConnectionError):
		...     await async_operation()
	Returns:
		When used as a decorator, returns a wrapped function that suppresses specified exceptions.
		When called with exception types, returns a new TracebackSuppressor instance.
	Notes:
		- Non-Exception subclasses are not suppressed
		- When an unhandled exception occurs, the specified fn (default: print_exc) is called
	"""

	def __init__(self, *args, fn=print_exc, **kwargs):
		self.fn = fn
		self.exceptions = args + tuple(kwargs.values())

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_tb):
		if exc_type and exc_value:
			for exception in self.exceptions:
				if issubclass(exc_type, exception):
					return True
			if not issubclass(exc_type, Exception):
				return
			self.fn()
		return True

	async def __aenter__(self):
		pass

	async def __aexit__(self, *args):
		return self.__exit__(*args)

	def __call__(self, *ins, default=None):
		if len(ins) == 1 and callable(ins[0]) and (not isinstance(ins[0], type) or not issubclass(ins[0], BaseException)):
			def decorator(*args, **kwargs):
				with self:
					return ins[0](*args, **kwargs)
				return default
			return decorator
		return self.__class__(*ins)

tracebacksuppressor = TracebackSuppressor()

class PropagateTraceback(RuntimeError):

	def __init__(self, ex, tb=None):
		super().__init__(*ex.args)
		self.original_traceback = tb

	@classmethod
	def cast(cls, ex, tb=None):
		if not getattr(ex, "original_traceback", None):
			try:
				ex.original_traceback = tb
			except AttributeError:
				return cls(ex, tb)
		return ex

RE = cdict()
def regexp(s, flags=0) -> re.Pattern:
	global RE
	if isinstance(s, re.Pattern):
		return s
	s = as_str(s)
	t = f"{s}\x00{flags}"
	try:
		return RE[t]
	except KeyError:
		RE[t] = re.compile(s, flags)
	return RE[t]

def loop(n):
	return repeat(None, n)

def resume(im, *its):
	yield im
	for it in its:
		yield from it

@hashable_args
@functools.lru_cache(maxsize=256)
def suppress(*args, **kwargs) -> contextlib.suppress:
	if not args and not kwargs:
		return contextlib.suppress(Exception)
	return contextlib.suppress(*args + tuple(kwargs.values()))

def all_subclasses(cls):
	yield cls
	for sub in cls.__subclasses__():
		yield from all_subclasses(sub)


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

@hashable_args
@functools.lru_cache(maxsize=256)
def lim_str(s, maxlen=10, mode="centre") -> str:
	"Limits a string to a maximum length, cutting from the middle and replacing with \"..\" when possible."
	if maxlen is None:
		return s
	if maxlen <= 3:
		return "..."
	if type(s) is not str:
		s = str(s)
	over = (len(s) - maxlen) / 2
	if over > 0:
		if mode == "centre":
			half = len(s) / 2
			s = s[:ceil(half - over - 1)] + ".." + s[ceil(half + over + 1):]
		elif mode == "right":
			s = "..." + s[3 - maxlen:]
		else:
			s = s[:maxlen - 3] + "..."
	return s