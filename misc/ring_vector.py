import bisect
import collections.abc
from collections import deque
import copy
from itertools import chain
from math import ceil, floor, log2, sqrt
import threading
import numpy as np


# Creates a nested tuple from a nested list.
def _nested_tuple(a):
	return tuple(_nested_tuple(i) if isinstance(i, (np.ndarray, collections.abc.MutableSequence)) else i for i in a)
def nested_tuple(a):
	return _nested_tuple(a) if isinstance(a, (np.ndarray, collections.abc.Sequence)) and type(a) not in (str, bytes) and not is_equal(a[0], a) else a
int_like = int | np.integer
object_like = (object, np.object_)
def round_min(x):
	if isinstance(x, int):
		return x
	if isinstance(x, np.integer):
		return x.item()
	y = int(x)
	if x == y:
		if isinstance(y, np.integer):
			return y.item()
		return y
	return x
def is_equal(x, y):
	try:
		eq = bool(x == y)
	except ValueError:
		return False
	try:
		return bool(eq)
	except ValueError:
		return np.all(eq)


class NonThreadBoundRLock:
    def __init__(self):
        import threading
        self._counter = 0
        self._internal_lock = threading.Lock()  # Protect counter
        self._sem = threading.Semaphore(1)
    def acquire(self):
        with self._internal_lock:
            if self._counter == 0:
                self._sem.acquire()
            self._counter += 1
    def release(self):
        with self._internal_lock:
            if self._counter == 0:
                raise RuntimeError("release() called on un-acquired lock")
            self._counter -= 1
            if self._counter == 0:
                self._sem.release()


class ReadWriteLock:
	def __init__(self):
		self.reader_count = 0
		self.reader_lock = threading.RLock()  # Mutex for reader count
		# Use our NonThreadBoundRLock for writer operations.
		self.writer_lock = NonThreadBoundRLock()

	def acquire_read_lock(self):
		with self.reader_lock:
			self.reader_count += 1
			if self.reader_count == 1:
				self.writer_lock.acquire()

	def release_read_lock(self):
		with self.reader_lock:
			self.reader_count = max(self.reader_count - 1, 0)
			if self.reader_count == 0:
				self.writer_lock.release()

	def acquire_write_lock(self):
		self.writer_lock.acquire()

	def release_write_lock(self):
		self.writer_lock.release()


class RingVector(collections.abc.MutableSequence, collections.abc.Callable):
	"""
	A high-performance, thread-safe, mutable sequence implementing a circular buffer.
	This class is built upon a NumPy array to provide the performance of C-level
	data structures with the convenience of a Pythonic API. It combines features
	from Python's `list`, `collections.deque`, and `set`, and supports NumPy-style
	vectorized operations.
	The circular buffer (or ring buffer) design allows for efficient appends and pops
	from both ends (amortized O(1) time complexity) by wrapping data around the
	edges of the underlying array, avoiding costly memory shifts for most operations.
	Key Features:
	- **Performance:** Leverages NumPy arrays for fast, vectorized operations.
	- **Thread-Safety:** Uses a custom ReadWriteLock to allow for safe concurrent
		reading and exclusive writing. Decorators `@reading` and `@writing` simplify
		lock management.
	- **Versatile API:** Implements the `MutableSequence` ABC, providing a rich set
		of methods similar to `list`, `deque`, `set`, and NumPy arrays.
	- **Efficient Memory Management:** Automatically handles resizing of the
		underlying buffer to accommodate new elements, with methods like `reserve()`
		for manual capacity control.
	- **Slicing and Indexing:** Supports integer, float (for interpolation), slice,
		and iterable-based indexing and assignment.
	- **Caching:** Caches results of expensive operations like hashing, creating a
		frozenset, or checking for sortedness, invalidating them upon modification.
	Parameters
	----------
	*args : iterable or elements
			The initial data for the vector. If a single argument is provided, it is
			treated as an iterable. If multiple arguments are given, they are used as
			the elements of the vector.
	dtype : type, optional
			The data type for the underlying NumPy array, by default `object`.
			Using specific types like `np.int32` or `np.float64` can significantly
			improve performance and reduce memory usage.
	device : int, optional
			A placeholder for potential future GPU/device support. Currently unused.
	Attributes
	----------
	length : int
			The number of elements currently in the vector.
	capacity : int
			The total number of elements the underlying buffer can hold before a
			reallocation is needed.
	dtype : numpy.dtype
			The data type of the elements in the vector.
	view : numpy.ndarray
			A contiguous NumPy array view of the elements. May trigger a memory
			reallocation and copy if the internal buffer is wrapped or oversized.
	views : tuple[numpy.ndarray]
			A tuple containing one or two NumPy array views that represent the
			data in the circular buffer. Returns one view if contiguous, two if
			wrapped around the buffer's edge.
	lock : ReadWriteLock
			The lock object used for managing thread-safe access.
	sorted : bool
			A cached property that indicates whether the vector is sorted. The check
			is performed lazily on first access.
	"""

	__slots__ = ("__weakref__", "_lock", "buffer", "offset", "length", "_view", "_margin", "_hash", "_frozenset", "_queries", "_sorted", "_index")

	@property
	def idx_dtype(self):
		if len(self) <= 2147483647:
			return np.int32
		if len(self) <= 9223372036854775807:
			return np.int64
		return object

	@property
	def lock(self) -> ReadWriteLock:
		try:
			return self._lock
		except AttributeError:
			self._lock = ReadWriteLock()
		return self._lock

	def acquire_read_lock(self):
		return self.lock.acquire_read_lock()

	def release_read_lock(self):
		return self.lock.release_read_lock()

	def acquire_write_lock(self):
		return self.lock.acquire_write_lock()

	# Cached attributes are reset whenever the array is modified
	def release_write_lock(self, view=False, order=False, elements=False):
		cleared = {"_view", "_margin", "_hash", "_frozenset", "_queries", "_sorted"}
		if view:
			cleared.difference_update(("_view", "_margin"))
		if order:
			cleared.discard("_sorted")
		if elements:
			cleared.difference_update(("_hash", "_frozenset", "_queries"))
		for k in cleared:
			try:
				delattr(self, k)
			except AttributeError:
				pass
		return self.lock.release_write_lock()

	# For thread-safety: Waits until the list is not busy performing an operation.
	def reading_with():
		def reading(func):
			def call(self, *args, **kwargs):
				self.acquire_read_lock()
				try:
					return func(self, *args, **kwargs)
				finally:
					self.release_read_lock()
			return call
		return reading
	reading = reading_with()

	# For thread-safety: Blocks the list until the operation is complete.
	def writing_with(**keep):
		def writing(func):
			def call(self, *args, **kwargs):
				self.acquire_write_lock()
				try:
					return func(self, *args, **kwargs)
				finally:
					self.release_write_lock(**keep)
			return call
		return writing
	writing = writing_with()

	# Init takes arguments and casts to a deque if possible, else generates as a single value.
	def __init__(self, *args, dtype=object, device=-1):
		if not args:
			self.length = 0
			if dtype not in object_like:
				self.buffer = np.empty(0, dtype=dtype)
			return
		elif len(args) == 1:
			iterable = args[0]
		else:
			iterable = args
		if issubclass(type(iterable), self.__class__) and iterable:
			data = iterable.buffer
			offs = iterable.offset
			size = iterable.length
		elif isinstance(iterable, np.ndarray):
			offs = 0
			size = len(iterable)
			data = np.asanyarray(iterable, dtype=dtype)
		else:
			if not isinstance(iterable, (collections.abc.Sequence, collections.abc.Mapping, np.ndarray)) or type(iterable) in (str, bytes) or isinstance(iterable, set | frozenset | dict):
				try:
					iterable = deque(iterable)
				except TypeError:
					iterable = [iterable]
			size = len(iterable)
			offs = 0
			data = np.empty(size, dtype=dtype)
			data[:] = iterable
		self.initialise(data, offs, size, dtype=dtype)

	@writing
	def initialise(self, buffer, offset=0, length=None, dtype=None):
		if dtype is not None:
			buffer = np.asanyarray(buffer, dtype=dtype)
		self.buffer = buffer
		self.offset = offset
		if length is None:
			length = len(buffer) if buffer is not None else 0
		self.length = length

	def fill(self, other, **keep):
		self.acquire_write_lock()
		try:
			if not isinstance(other, np.ndarray):
				other = self.to_iterable(other, match_length=False)
			if len(other) == 0:
				self.length = 0
				self.buffer = np.empty(0, dtype=self.dtype)
				return self
			self.initialise(other, dtype=self.dtype)
			return self
		finally:
			self.release_write_lock(**keep)

	def __getstate__(self):
		dtype = self.dtype
		dstr = "O" if dtype in object_like else str(dtype)
		data = list(self) if dtype in object_like else self.view
		if getattr(self, "_iter", None):
			return 1, data, dstr, self._index
		return 0, data, dstr

	@writing
	def __setstate__(self, s):
		if isinstance(s, tuple):
			if len(s) == 2:
				if s[0] is None:
					for k, v in s[1].items():
						if k == "data":
							k = "buffer"
						elif k == "offs":
							k = "offset"
						elif k == "size":
							k = "length"
						if k in self.__slots__:
							setattr(self, k, v)
					return
			elif len(s) == 1:
				b = s[0]
				return self.initialise(b)
			if s[0] is None:
				return self.initialise([])
			if not isinstance(s[0], np.ndarray):
				if s[0] == 0:
					view, dtype = s[1:]
					self.initialise(view, dtype=dtype)
				elif s[0] == 1:
					view, dtype, _index = s[1:]
					self._index = _index
					self.initialise(view, dtype=dtype)
				else:
					raise NotImplementedError(f"Unexpected {self.__class__.__name__} version: {s[0]}")
				return
			if len(s) == 4:
				self._index = s[3]
				s = s[:3]
			if len(s) == 3:
				self.initialise(*s)
			return
		elif isinstance(s, dict):
			try:
				spl = s["buffer"], s["offset"], s["length"]
			except KeyError:
				spl = s["data"], s["offs"], s["size"]
			return self.initialise(*spl)
		raise TypeError("Unpickling failed:", s)

	def __getattr__(self, k):
		if k in self.__class__.__slots__ or k in self.__class__.__dict__:
			return self.__getattribute__(k)
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		return getattr(self.__getattribute__("view"), k)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(dir(self.buffer))
		return data

	@property
	def dtype(self):
		try:
			return self.buffer.dtype
		except AttributeError:
			return np.object_

	# Returns an array view representing the items currently "in" the list. Forces a reallocation if not already contiguous, or if the array's capacity is more than 4x the current amount of elements.
	@property
	@reading
	def view(self) -> list | np.ndarray:
		if not self:
			return np.empty(0, dtype=self.dtype)
		try:
			if len(self._view) == 1:
				return self._view[0]
		except AttributeError:
			pass
		if self.capacity > self.length * 4:
			self.reserve()
		elif self.border > self.capacity:
			self.as_contiguous()
		self._view = (self.buffer[self.offset:self.border],)
		return self._view[0]

	def __array__(self):
		return np.asanyarray(self.view, dtype=self.dtype)

	# Returns the array's contiguous view if possible, otherwise raises AttributeError.
	@reading
	def peek(self) -> list | np.ndarray:
		if not self:
			return []
		try:
			if len(self._view) == 1:
				return self._view[0]
		except AttributeError:
			pass
		if self.border <= self.capacity:
			self._view = (self.buffer[self.offset:self.border],)
			return self._view[0]
		raise AttributeError

	# Returns the left and right "slices" of the array. Only returns one slice if array doesn't wrap, returns empty tuple if no elements exist at all.
	@property
	@reading
	def views(self) -> tuple:
		if not self:
			return ()
		try:
			return self._view
		except AttributeError:
			pass
		if self.border <= self.capacity:
			self._view = (self.buffer[self.offset:self.border],)
		else:
			self._view = (self.buffer[self.offset:], self.buffer[:self.margin])
		return self._view

	@property
	def margin(self) -> int:
		try:
			return self._margin
		except AttributeError:
			self._margin = self.border % self.capacity
		return self._margin

	@property
	def border(self) -> int:
		return self.offset + self.length

	def chunk_size(self, scale=256) -> int:
		return ceil(scale * log2(max(2, self.length)))

	# Average O(log n) check for whether the array is sorted
	@reading
	def check_sorted(self) -> bool:
		if self.length < 2:
			return False
		if self.length < 3:
			try:
				return self[0] <= self[1]
			except TypeError:
				return False
		try:
			if self[0] > self[-1] or self[0] > self[1] or self[-2] > self[-1]:
				return False
		except TypeError:
			return False
		try:
			return self._check_sorted(1, self.length - 1, chunk=self.chunk_size(4096))
		except TypeError:
			return False
	def _check_sorted(self, start=0, end=1, chunk=256) -> bool:
		if start + 1 >= end:
			return True
		if start + 2 >= end:
			return self[start] <= self[end - 1]
		if start + chunk >= end:
			if self[start] > self[end - 1] or self[start] > self[start + 1] or self[end - 2] > self[end - 1]:
				return False
			try:
				return np.all(self[start + 1:end - 2] <= self[start + 2:end - 1])
			except TypeError:
				return all(self[i] <= self[i + 1] for i in range(start + 1, end - 2))
		mid = (start + end) // 2
		if not self[mid] <= self[mid + 1]:
			return False
		if not self._check_sorted(start, mid, chunk=chunk):
			return False
		if not self._check_sorted(mid + 1, end, chunk=chunk):
			return False
	@property
	def sorted(self) -> bool:
		if not self:
			return False
		try:
			return self._sorted
		except AttributeError:
			self._sorted = self.check_sorted()
		return self._sorted

	@reading
	def min(self):
		if self.sorted:
			return self[0]
		return min(map(np.min, self.views))
	@reading
	def max(self):
		if self.sorted:
			return self[0]
		return max(map(np.max, self.views))
	@reading
	def sum(self):
		x = sum(map(np.sum, self.views))
		return round_min(x)
	@reading
	def mean(self):
		x = np.mean(list(map(np.mean, self.views)))
		return round_min(x)
	@reading
	def product(self):
		x = np.prod(list(map(np.prod, self.views)))
		return round_min(x)
	prod = product

	@writing_with(order=True, elements=True)
	def as_contiguous(self):
		if not self:
			return self
		left, *spl = self.views
		if not spl:
			return self
		offset = self.capacity - self.length >> 1
		if abs(self.offset - offset) <= self.capacity - self.length:
			self._move_range(self.offset, self.length, offset)
			self.offset = offset
			return self
		new_buffer = np.empty((self.length, *self.buffer.shape[1:]), dtype=self.dtype)
		right = spl[0]
		new_buffer[len(left):] = right
		new_buffer[:len(left)] = left
		self.initialise(new_buffer, dtype=self.dtype)
		return self

	# Reserves enough capacity for `(n + spaces) * 2` elements, the minimum required to make sure all elements can be moved around without additional temporary buffers. Centres data within the buffer to minimise future reallocations.
	@writing_with(order=True, elements=True)
	def reserve(self, spaces=0):
		if not self:
			return self.initialise(np.empty(8, dtype=self.dtype), 3, 0)
		curr_length = self.length + spaces
		new_length = curr_length * 2
		if new_length < self.capacity / 2 or new_length > self.capacity:
			new_buffer = np.empty((new_length, *self.buffer.shape[1:]), dtype=self.dtype)
		else:
			new_buffer = self.buffer
		centre = len(new_buffer) >> 1
		lpos = centre - ceil(curr_length / 2)
		left, *spl = self.views
		if spl:
			right = spl[0]
			new_buffer[lpos + len(left):lpos + self.length] = right
		new_buffer[lpos:lpos + len(left)] = left
		self.initialise(new_buffer, lpos, self.length)
		return self

	def __call__(self, arg=1, *void1, **void2):
		if arg == 1:
			return self.copy()
		return self * arg

	# Returns the hash value of the data in the list.
	def __hash__(self) -> int:
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(self.to_frozenset())
		return self._hash

	# A `frozenset` element is kept cached for future retrieval, and for `__contains__` checks
	@reading
	def to_frozenset(self):
		try:
			return self._frozenset
		except AttributeError:
			self._frozenset = frozenset(self)
		return self._frozenset

	# Basic functions
	@reading
	def __str__(self):
		return "[" + ", ".join(map(repr, self)) + "]"
	@reading
	def __repr__(self):
		return f"{self.__class__.__name__}({tuple(self) if bool(self) else ''})"
	@reading
	def __bool__(self):
		return self.length > 0

	@writing_with(view=True)
	def _mut(self, operation, other, **kwargs):
		i = 0
		for view in self.views:
			operation(view, other[i:i + len(view)], out=view, **kwargs)
			if len(other) > 1:
				i += len(view)
		return self
	@reading
	def _imut(self, operation, other, **kwargs):
		if not self.length:
			return []
		temp = []
		i = 0
		for view in self.views:
			temp.append(operation(view, other[i:i + len(view)], **kwargs))
			if len(other) > 1:
				i += len(view)
		if len(temp) == 1:
			return self.__class__(temp[0])
		return self.__class__(np.concatenate(temp, dtype=temp[0].dtype))
	@reading
	def astype(self, dtype):
		if dtype in object_like:
			if self.dtype in object_like:
				return self
		elif self.dtype == dtype:
			return self
		if not self:
			return self.__class__(dtype=dtype)
		views = [view.astype(dtype) for view in self.views]
		temp = np.concatenate(views) if len(views) > 1 else views[0]
		return self.__class__(temp, dtype=dtype)

	# Arithmetic functions
	def __iadd__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.add, other)
	def __isub__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.subtract, other)
	def __imul__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.multiply, other)
	def __imatmul__(self, other):
		raise NotImplementedError
	def __itruediv__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.true_divide, other)
	def __ifloordiv__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.floor_divide, other)
	def __imod__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.mod, other)
	def __ipow__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.power, other)
	def __ilshift__(self, other):
		other = self.to_iterable(other)
		if all(isinstance(x, int_like) for x in self) and all(isinstance(y, int_like) for y in other):
			return self._mut(np.left_shift, other)
		else:
			return self._mut(np.multiply, np.power(2, other))
	def __irshift__(self, other):
		other = self.to_iterable(other)
		if all(isinstance(x, int_like) for x in self) and all(isinstance(y, int_like) for y in other):
			return self._mut(np.right_shift, other)
		else:
			return self._mut(np.true_divide, np.power(2, other))
	def __iand__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.logical_and, other)
	def __ixor__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.logical_xor, other)
	def __ior__(self, other):
		other = self.to_iterable(other)
		return self._mut(np.logical_or, other)
	def __neg__(self):
		return self.__class__(-self.view)
	def __pos__(self):
		return self
	def __abs__(self):
		return self.__class__(np.abs(self.view))
	def __invert__(self):
		return self.__class__(np.invert(self.view))
	def __add__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.add, other)
	def __sub__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.subtract, other)
	def __mul__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.multiply, other)
	def __matmul__(self, other):
		raise NotImplementedError
	def __truediv__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.true_divide, other)
	def __floordiv__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.floor_divide, other)
	def __mod__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.mod, other)
	def __pow__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.power, other)
	def __lshift__(self, other):
		other = self.to_iterable(other)
		if all(isinstance(x, int_like) for x in self) and all(isinstance(y, int_like) for y in other):
			return self._imut(np.left_shift, other)
		else:
			return self._imut(np.multiply, np.power(2, other))
	def __rshift__(self, other):
		other = self.to_iterable(other)
		if all(isinstance(x, int_like) for x in self) and all(isinstance(y, int_like) for y in other):
			return self._imut(np.right_shift, other)
		else:
			return self._imut(np.true_divide, np.power(2, other))
	def __and__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.logical_and, other)
	def __xor__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.logical_xor, other)
	def __or__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.logical_or, other)
	@writing_with(view=True, order=True)
	def round(self, prec=0):
		if not self:
			return self
		low, high = self.min(), self.max()
		m = max(abs(low), abs(high))
		if m:
			m = log2(m) + log2(10) * prec
		if m > 53:
			temp = [round(x, prec) for x in self]
			return self.fill(temp, order=True)
		if self.dtype in object_like:
			# dtype = np.float64 if m > 24 else np.float32
			temp = self.view.astype(np.float64)
			temp.round(prec, out=temp)
			temp = temp.astype(self.dtype)
			return self.fill(temp, order=True)
		return self._mut(np.round, prec)
	def __round__(self, prec=0):
		return self.__class__(np.round(self.view, prec))
	def __trunc__(self):
		return self.__class__(np.trunc(self.view))
	def __floor__(self):
		self.__class__(np.floor(self.view))
	def __ceil__(self):
		self.__class__(np.ceil(self.view))
	def __index__(self):
		return self.view
	__radd__ = __add__
	def __rsub__(self, other):
		return -self + other
	def __rtruediv__(self, other):
		return self.__class__(other) / self
	def __rfloordiv__(self, other):
		return self.__class__(other) // self
	def __rmod__(self, other):
		return self.__class__(other) % self
	def __rpow__(self, other):
		return self.__class__(other) ** self
	def __rlshift__(self, other):
		return self.__class__(other) << self
	def __rrshift__(self, other):
		return self.__class__(other) >> self
	__rand__ = __and__
	__rxor__ = __xor__
	__ror__ = __or__

	# Comparison operations
	def __lt__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.less, other)
	def __le__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.less_equal, other)
	def __eq__(self, other):
		try:
			if self is other:
				return True
			if self.length != len(other):
				return False
			if isinstance(other, (set, frozenset, dict)):
				return self.to_frozenset() == other
			other = self.to_iterable(other)
			return np.all(self._imut(np.equal, other))
		except (TypeError, IndexError):
			return False
	def __ne__(self, other):
		try:
			if self is other:
				return False
			if self.length != len(other):
				return True
			if isinstance(other, (set, frozenset, dict)):
				return self.to_frozenset() != other
			other = self.to_iterable(other)
			return np.any(self._imut(np.not_equal, other))
		except (TypeError, IndexError):
			return True
	def eq(self, other):
		if self is other:
			return np.ones(self.length, dtype=bool)
		other = self.to_iterable(other)
		return self._imut(np.equal, other)
	equal = eq
	def ne(self, other):
		if self is other:
			return np.zeros(self.length, dtype=bool)
		other = self.to_iterable(other)
		return self._imut(np.not_equal, other)
	not_equal = ne
	def isclose(self, other, dtype=None, rtol=1e-05, atol=1e-08, equal_nan=False):
		if not self:
			return np.empty(0, dtype=self.dtype)
		if self is other:
			return np.ones(len(other), dtype=bool)
		if self.dtype in object_like:
			if dtype is None:
				try:
					dtype = other.dtype
					if dtype in object_like:
						raise AttributeError
				except AttributeError:
					dtype = np.float64
			return np.isclose(self.view.astype(dtype), other)
		other = self.to_iterable(other)
		return self._imut(np.isclose, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
	def __gt__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.greater, other)
	def __ge__(self, other):
		other = self.to_iterable(other)
		return self._imut(np.greater_equal, other)

	# Takes ints, floats, slices and iterables for indexing
	@reading
	def __getitem__(self, k):
		if isinstance(k, float):
			x, y = floor(k), ceil(k)
			if x == y:
				return self[x]
			if y >= self.length:
				y = self.length - 1
			return self[x] * (y - k) + self[y] * (k - x)
		try:
			view = self.peek()
		except AttributeError:
			pass
		else:
			if isinstance(k, slice):
				return self.__class__(view[k])
			return view[k]
		if isinstance(k, slice):
			start, stop, step = k.indices(self.length)
			if step == 1:
				temp = self.__class__()
				temp.initialise(self.buffer, (self.offset + start) % self.capacity, stop - start)
				return temp
			# Non-contiguous memory cannot be sliced mutably; force a reserve
			return self.__class__(self.view[k])
		elif not isinstance(k, collections.abc.Iterable):
			if k < -self.length or k >= self.length:
				raise IndexError(f"Index {k} out of bounds for size {self.length} vector.")
			if k < 0:
				k += self.length
			i = (self.offset + k % self.length) % self.capacity
			return self.buffer[i]
		return self.view[k]

	# Takes ints, floats, slices and iterables for indexing
	@writing_with(view=True)
	def __setitem__(self, k, v):
		if isinstance(k, (float, np.floating)):
			x, y = floor(k), ceil(k)
			if x == y:
				self[x] = v
				return
			if y >= self.length:
				y = self.length - 1
			a = self[x] * (k - x) + v * (y - k)
			b = self[y] * (y - k) + v * (k - x)
			self[x] = a
			self[y] = b
			return
		try:
			view = self.peek()
		except AttributeError:
			pass
		else:
			view[k] = v
			return
		if isinstance(k, slice):
			try:
				left, *spl = self.views
				if spl:
					right = spl[0]
					lstart, lstop, lstep = k.indices(len(left))
					lpos = (len(left) - lstart) // lstep
					left[lstart:lstop:lstep] = v[:lpos]
					rstart, rstop, rstep = k.indices(self.length)
					rstart = max(0, rstart - len(left))
					rstop = max(0, rstop - len(left))
					right[rstart:rstop:rstep] = v[lpos:]
				else:
					left[k] = v
			except ValueError as ex:
				raise IndexError(*ex.args)
			return
		elif not isinstance(k, collections.abc.Iterable):
			if k < -self.length or k >= self.length:
				raise IndexError(f"Index {k} out of bounds for size {self.length} vector.")
			if k < 0:
				k += self.length
			i = (self.offset + k) % self.capacity
			self.buffer[i] = v
			return
		self.view[k] = v

	# Takes ints and slices for indexing
	@writing_with(order=True)
	def __delitem__(self, key):
		if isinstance(key, slice):
			s = key.indices(self.length)
			return self.pops(range(*s))
		if isinstance(key, int_like):
			return self.pop(key)
		return self.pops(key)

	# Basic sequence functions
	def __len__(self):
		return object.__getattribute__(self, "length")
	__length_hint__ = qsize = __len__
	@property
	@reading
	def shape(self):
		return (self.length,)
	@property
	@reading
	def capacity(self):
		return len(self.buffer)
	def fsize(self):
		return self.capacity
	def __iter__(self):
		return chain.from_iterable(self.views)
	def __reversed__(self):
		return chain.from_iterable(map(reversed, reversed(self.views)))

	@reading
	def next(self):
		try:
			self._index = (self._index + 1) % self.length
		except AttributeError:
			self._index = 0
		return self[self._index]

	@reading
	def __bytes__(self):
		view = self.astype(np.uint8).view
		return view.tobytes()
	tobytes = __bytes__

	@reading
	def __contains__(self, item):
		if not self:
			return False
		q = getattr(self, "_queries", 0)
		if q > log2(self.length) + 1:
			return item in self.to_frozenset()
		try:
			return item in self._frozenset
		except AttributeError:
			self._queries = q + 1
		try:
			return any(item in arr for arr in self.views)
		except ValueError:
			return any(any(is_equal(e, item) for e in arr) for arr in self.views)

	def __copy__(self):
		return self.copy()

	def __deepcopy__(self, memo):
		return self.__class__([copy.deepcopy(v, memo) for v in self])

	# Creates an iterable from an iterator or item, making sure the shape matches.
	def to_iterable(self, other, match_length=True, dtype=None):
		dtype = dtype if dtype is not None else self.dtype
		if not isinstance(other, (collections.abc.Sequence, np.ndarray)) or isinstance(other, collections.abc.Mapping):
			try:
				other = list(other)
			except TypeError:
				other = [other]
		if match_length and len(other) not in (1, self.length) and self.length:
			raise IndexError(f"Unable to broadcast operation on objects with size {self.length} and {len(other)}.")
		if isinstance(other, self.__class__):
			other = other.view
		if hasattr(other, "__array__"):
			return np.asanyarray(other, dtype=dtype)
		if isinstance(other, collections.abc.Iterable):
			return np.fromiter(other, dtype=dtype)
		x = np.empty(len(other), dtype=dtype)
		x[:] = other
		return x

	@writing
	def clear(self):
		self.length = 0
		if self.dtype in object_like:
			try:
				del self.buffer
			except AttributeError:
				pass
		else:
			self.buffer = np.empty(0, dtype=self.dtype)
		return self

	@reading
	def copy(self, deep=False):
		if deep:
			return self.__class__(copy.deepcopy(self.view))
		return self.__class__(self.view.copy())

	@writing_with(order=True, elements=True)
	def sort(self, *args, key=None, reverse=False, **kwargs):
		try:
			if key is not None:
				view = sorted(self.view, *args, key=key, reverse=reverse, **kwargs)
				self.fill(view, elements=True)
			else:
				self.view.sort(kind="stable")
				if reverse:
					self.reverse()
		except ValueError:
			view = sorted(nested_tuple(self.view), *args, key=key, reverse=reverse, **kwargs)
			self.fill(view, elements=True)
		self._sorted = True
		return self

	@writing_with(view=True, elements=True)
	def shuffle(self, *args, **kwargs):
		# This can be optimised at some point for non-contiguous arrays, but shuffling is O(n) already
		view = self.view
		self.acquire_write_lock()
		try:
			np.random.shuffle(view, *args, **kwargs)
		finally:
			self.release_write_lock()
		return self

	@writing_with(elements=True)
	def reverse(self):
		view = self.view[::-1]
		return self.fill(view, elements=True)

	# Rotates the list a certain amount of steps.
	@writing_with(elements=True)
	def rotate(self, steps):
		s = self.length
		if not s:
			return self
		steps %= s
		if steps > s >> 1:
			steps -= s
		if abs(steps) <= max(8, sqrt(s)):
			while steps > 0:
				self.appendleft(self.popright())
				steps -= 1
			while steps < 0:
				self.appendright(self.popleft())
				steps += 1
			return self
		temp = self.view.copy()
		if steps > 0:
			self.buffer[steps:s] = temp[:s - steps]
			self.buffer[:steps] = temp[s - steps:]
		else:
			steps = -steps
			self.buffer[:s - steps] = temp[steps:]
			self.buffer[s - steps:] = temp[:steps]
		self.offset = 0
		return self
	rotateright = rotate

	def rotateleft(self, steps):
		return self.rotate(-steps)

	# For compatibility with dict.get
	@reading
	def get(self, key, default=None):
		try:
			return self[key]
		except (TypeError, LookupError):
			return default

	@writing_with(order=True)
	def popleft(self):
		if not self:
			raise IndexError("Pop from empty Ring Vector.")
		temp = self.buffer[self.offset]
		self.offset = (self.offset + 1) % self.capacity
		self.length -= 1
		return temp
	dequeue = popleft

	@writing_with(order=True)
	def popright(self):
		if not self:
			raise IndexError("Pop from empty Ring Vector.")
		temp = self.buffer[self.margin - 1]
		self.length -= 1
		return temp

	# Moves a range of elements in the buffer. Uses vectorised numpy slicing while maintaining move order (i.e. elements will not improperly overwrite each other in intermediate steps, and the array will wrap correctly around the edges).
	def _move_range(self, source, size, dest):
		buffer = self.buffer
		bufsize = len(buffer)
		source %= bufsize
		dest %= bufsize
		if source == dest:
			return
		size %= bufsize
		if size == 0:
			return
		source_end = source + size
		dest_end = dest + size
		if source_end <= bufsize:
			if dest_end <= bufsize:
				buffer[dest:dest_end] = buffer[source:source_end]
				return
			# Dest wraps only
			right = dest_end - bufsize
			left = size - right
			if dest_end - bufsize > source:
				buffer[bufsize - left:bufsize] = buffer[source:source + left]
				buffer[0:right] = buffer[source + left:source_end]
			else:
				buffer[0:right] = buffer[source + left:source_end]
				buffer[bufsize - left:bufsize] = buffer[source:source + left]
			return
		if dest_end <= bufsize:
			# Source wraps only
			right = source_end - bufsize
			left = size - right
			if source_end - bufsize > dest:
				buffer[dest + left:dest_end] = buffer[0:right]
				buffer[dest:dest + left] = buffer[bufsize - left:bufsize]
			else:
				buffer[dest:dest + left] = buffer[bufsize - left:bufsize]
				buffer[dest + left:dest_end] = buffer[0:right]
			return
		if dest > source:
			# Source-dest wrap overlap, moving data right
			distance = dest - source # Always positive
			right = source_end - bufsize
			buffer[distance:right + distance] = buffer[0:right]
			centre = bufsize - distance
			buffer[0:distance] = buffer[centre:bufsize]
			buffer[source + distance:bufsize] = buffer[source:centre]
			return
		# Source-dest wrap overlap, moving data left
		distance = source - dest # Always positive
		centre = bufsize - source
		buffer[dest:dest + centre] = buffer[dest + distance:bufsize]
		buffer[dest + centre:bufsize] = buffer[0:distance]
		right = dest_end - bufsize
		buffer[0:right] = buffer[distance:right + distance]

	# Removes an item from the list. O(n) time complexity apart from the ends.
	@writing_with(order=True)
	def pop(self, index=None, *args):
		if not self:
			if args:
				return args[0]
			raise IndexError("Pop from empty Ring Vector.")
		if index is None:
			return self.popright()
		if index >= self.length or index <= -self.length:
			if args:
				return args[0]
			print(index, self.length)
			raise IndexError("Ring Vector index out of range.")
		index %= self.length
		if index == self.length - 1:
			return self.popright()
		if index == 0:
			return self.popleft()
		temp = self[index]
		if self.length > self.capacity:
			self.reserve(1)
		if index >= self.length >> 1:
			# index is on the right; shift everything >index left
			self._move_range(self.offset + index + 1, self.length - index - 1, self.offset + index)
		else:
			# index is on the left; shift everything <index right
			self._move_range(self.offset, index, self.offset + 1)
			self.offset = (self.offset + 1) % self.capacity
		self.length -= 1
		return temp

	# Inserts an item into the list. O(n) time complexity apart from the ends.
	@writing
	def insert(self, index, value):
		if not self:
			return self.fill((value,))
		if index >= self.length:
			return self.append(value)
		index %= self.length
		if index == 0:
			return self.appendleft(value)
		if self.length + 1 > self.capacity / 2:
			self.reserve(1)
		self.length += 1
		if index >= self.length >> 1:
			# index is on the right; shift everything >=index right
			self._move_range(self.offset + index, self.length - index - 1, self.offset + index + 1)
		else:
			# index is on the left; shift everything <index left
			self.offset = (self.offset - 1) % self.capacity
			self._move_range(self.offset + 1, index, self.offset)
		self.buffer[(self.offset + index) % self.capacity] = value
		return self
	ins = insert

	@reading
	def searchsorted(self, value):
		left, *spl = self.views
		if spl and value > left[-1]:
			right = spl[0]
			return np.searchsorted(right, value) + len(left)
		return np.searchsorted(left, value)

	# Insertion sort using a binary search to find target position. O(n) time complexity unless not already sorted, in which case automatically sorts the list with O(n log n) complexity.
	@writing_with(order=True)
	def insort(self, value, key=None):
		try:
			if not self:
				return self.fill((value,))
			if not self.sorted:
				self.sort(key=key)
			if key is None:
				index = self.searchsorted(value)
				self.insert(index, value)
				return self
			bisect.insort_left(self, value, key=key)
		finally:
			self._sorted = True
		return self

	# Removes up to `count` instances of a certain value from the list.
	@writing_with(order=True)
	def remove(self, value, count=1, key=None, last=False, passthrough=False):
		if not self:
			if passthrough:
				return self
			raise ValueError("Ring Vector is empty.")
		pops = self.search(value, count=count, key=key)
		if count:
			if last:
				pops = pops[-count:]
			else:
				pops = pops[:count]
		if len(pops):
			self.pops(pops)
		elif not passthrough:
			raise ValueError(f"Element {value} not in Ring Vector.")
		return self
	rm = remove
	def discard(self, value, count=None, key=None, last=False):
		return self.remove(value, count=count, key=key, last=last, passthrough=True)

	# Removes all duplicate values from the list. Makes use of frozensets and numpy.unique when maintaining order is not required.
	@writing_with(order=True)
	def dedup(self, sort=None, key=None):
		if not self:
			return self
		if not key and sort:
			try:
				temp = np.unique(self.as_contiguous())
			except TypeError:
				temp = sorted(self.to_frozenset())
			self._sorted = True
		elif not key and sort is None:
			temp = self.to_frozenset()
			self._sorted = False
		else:
			temp = deque()
			found = set()
			for x in self:
				y = (key or nested_tuple)(x)
				if y not in found:
					found.add(y)
					temp.append(x)
		self.fill(temp, order=True)
		return self
	uniq = unique = dedup

	# Returns first matching value in list.
	def index(self, value, key=None):
		return self.search(value, count=1, key=key)[0]
	i = index

	# Returns last matching value in list.
	def rindex(self, value, key=None):
		return self.search(value, key=key)[-1]

	# Returns indices representing positions for all instances of the target found in list, using binary search for O(log n) complexity when applicable.
	@reading
	def search(self, value, count=None, key=None):
		if not self or count is not None and count <= 0:
			return []
		if key is None:
			if self.sorted:
				i = self.searchsorted(value)
				if i not in range(self.length) or self[i] != value:
					raise ValueError(f"{value} not found.")
				indices = [i]
				if count is None or count > 1:
					indices = deque(indices)
					for x in range(i + self.offset - 1, self.offset - 1, -1):
						if self.buffer[x % self.capacity] == value:
							indices.appendleft(x - self.offset)
						else:
							break
					for x in range(i + self.offset + 1, self.border):
						if self.buffer[x % self.capacity] == value:
							indices.append(x - self.offset)
						else:
							break
					indices = np.asanyarray(indices, dtype=self.idx_dtype)
					indices %= self.length
			else:
				view = self.view
				other = self.to_iterable(value)
				chunk = increment = self.chunk_size(256)
				indices = []
				offset = 0
				while offset < self.length and (count is None or len(indices) < count):
					check = other[offset:offset + chunk] if len(other) > 1 else other
					temp = np.nonzero(view[offset:offset + chunk] == check)[0]
					temp += offset
					indices.extend(temp)
					offset += chunk
					chunk += increment
		else:
			if not count:
				indices = [i for i, v in enumerate(map(key, self)) if v == value]
			else:
				indices = []
				for i, v in enumerate(map(key, self)):
					if v == value:
						indices.append(i)
						if len(indices) >= count:
							break
		if not len(indices):
			raise ValueError(f"{value} not found.")
		if count is not None and len(indices) > count:
			return indices[:count]
		return indices
	find = findall = search

	# Counts the amount of instances of the target within the list.
	@reading
	def count(self, value, key=None):
		if not self:
			return 0
		if key is None:
			other = self.to_iterable(value)
			mask = self._imut(np.equal, other)
			return mask.sum()
		return sum(key(i) == value for i in self)

	@reading
	def concat(self, value):
		if not self:
			return []
		temp = np.concatenate([*self.views, value], dtype=self.dtype)
		return self.__class__(temp)

	# Appends item at the start of the list, reallocating when necessary.
	@writing
	def appendleft(self, value):
		if not self:
			return self.fill((value,))
		if self.length >= self.capacity:
			self.reserve(1)
		self.offset = (self.offset - 1) % self.capacity
		self.buffer[self.offset] = value
		self.length += 1
		return self

	# Appends item at the end of the list, reallocating when necessary.
	@writing
	def append(self, value):
		if not self:
			return self.fill((value,))
		if self.length >= self.capacity:
			self.reserve(1)
		self.buffer[self.margin] = value
		self.length += 1
		return self
	appendright = app = enqueue = append

	@writing
	def add(self, value):
		if value not in self:
			return self.append(value)
		return self

	# Appends iterable at the start of the list, reallocating when necessary.
	@writing
	def extendleft(self, value, reverse=True):
		value = self.to_iterable(value, match_length=False)
		if reverse:
			value = value[::-1]
		if not self:
			return self.fill(value)
		if self.length + len(value) > self.capacity:
			temp = np.concatenate([value, *self.views])
			return self.fill(temp)
		for v in reversed(value):
			self.appendleft(v)
		return self

	# Appends iterable at the end of the list, reallocating when necessary.
	@writing
	def extend(self, value, reverse=False):
		value = self.to_iterable(value, match_length=False)
		if reverse:
			value = value[::-1]
		if not self:
			return self.fill(value)
		if self.length + len(value) > self.capacity:
			temp = np.concatenate([*self.views, value])
			return self.fill(temp)
		for v in value:
			self.append(v)
		return self
	extendright = ext = extend

	# Removes items according to an array of indices.
	@writing_with(order=True)
	def delitems(self, indices, keep=True):
		indices = self.to_iterable(indices, match_length=False, dtype=self.idx_dtype)
		if len(indices) < self.chunk_size(4):
			if len(indices) < 1:
				if keep:
					return self.__class__()
				return self
			if len(indices) == 1:
				temp = self.pop(indices[0])
				if keep:
					return self.__class__((temp,))
				return self
			indices.sort()
			if keep:
				temp = self.view[indices]
			for i in reversed(indices):
				self.pop(i)
			if keep:
				return self.__class__((temp,))
			return self
		left, *spl = self.views
		if spl:
			indices.sort()
			right = spl[0]
			i = np.searchsorted(indices, len(left))
			left2 = np.delete(left, indices[:i])
			indices[i:] -= len(left)
			right2 = np.delete(right, indices[i:])
			temp = np.concatenate([left2, right2])
			if keep:
				to_keep = np.concatenate([left[indices[:i]], right[indices[i:]]])
		else:
			temp = np.delete(left, indices)
			if keep:
				to_keep = left[indices]
		self.fill(temp, order=True)
		if keep:
			return to_keep
		return self
	pops = delitems

	# Inserts iterable at the selected index, always reallocating.
	@writing
	def splice(self, index, value):
		value = self.to_iterable(value, match_length=False)
		if not self:
			return self.fill((value,))
		if index <= 0:
			temp = np.concatenate([value, *self.views])
		elif index >= self.length:
			temp = np.concatenate([*self.views, value])
		else:
			left, *spl = self.views
			if spl:
				right = spl[0]
				if index >= len(left):
					temp = np.concatenate([left, right[:index - len(left)], value, right[index - len(left):]])
				else:
					temp = np.concatenate([left[:index], value, left[index:], right])
			else:
				temp = np.concatenate([left[:index], value, left[index:]])
		return self.fill(temp)
	exti = extendi = splice

	@reading
	def split(self, value, splits=None):
		indices = self.search(value, count=splits)
		view = self.view
		out = []
		pos = 0
		for i in indices:
			out.append(self.__class__(view[pos:i]))
			pos = i + 1
		out.append(self.__class__(view[pos:]))
		return out

	# Similar to str.join().
	@reading
	def join(self, iterable):
		iterable = self.to_iterable(iterable, match_length=False)
		temp = []
		for i, v in enumerate(iterable):
			if isinstance(v, collections.abc.Iterable):
				temp.append(v)
			else:
				temp.append([v])
			if i != len(iterable) - 1:
				temp.append(self)
		temp = np.concatenate(temp, dtype=self.dtype)
		return self.__class__(temp)

	# Similar to str.replace().
	@writing
	def replace(self, original, new):
		other = self.to_iterable(original)
		mask = self._imut(np.equal, other)
		self[mask] = new
		return self

	# Similar to str.strip().
	@writing_with(order=True)
	def strip(self, *values):
		pops = 0
		for e in self:
			if e in values:
				pops += 1
			else:
				break
		self.offset += pops
		self.length -= pops
		pops = 0
		for e in reversed(self):
			if e in values:
				pops += 1
			else:
				break
		self.length -= pops
		return self

	# For compatibility with dict() attributes.
	@reading
	def keys(self):
		return range(self.length)
	@reading
	def values(self):
		return iter(self)
	@reading
	def items(self):
		return enumerate(self)

	# For compatibility with set() attributes.
	@reading
	def isdisjoint(self, other):
		if type(other) not in (set, frozenset):
			other = frozenset(other)
		return self.to_frozenset().isdisjoint(other)

	@reading
	def issubset(self, other):
		if type(other) not in (set, frozenset):
			other = frozenset(other)
		return self.to_frozenset().issubset(other)

	@reading
	def issuperset(self, other):
		if type(other) not in (set, frozenset):
			other = frozenset(other)
		return self.to_frozenset().issuperset(other)

	@reading
	def union(self, *others):
		args = deque()
		for other in others:
			if type(other) not in (set, frozenset):
				other = frozenset(other)
			args.append(other)
		return self.to_frozenset().union(*args)

	@reading
	def intersection(self, *others):
		args = deque()
		for other in others:
			if type(other) not in (set, frozenset):
				other = frozenset(other)
			args.append(other)
		return self.to_frozenset().intersection(*args)

	@reading
	def difference(self, *others):
		args = deque()
		for other in others:
			if type(other) not in (set, frozenset):
				other = frozenset(other)
			args.append(other)
		return self.to_frozenset().difference(*args)

	@reading
	def symmetric_difference(self, other):
		if type(other) not in (set, frozenset):
			other = frozenset(other)
		return self.to_frozenset().symmetric_difference(other)

	@writing
	def update(self, *others, uniq=True):
		for other in others:
			if isinstance(other, collections.abc.Mapping):
				other = other.values()
			self.extend(other)
		if uniq:
			self.uniq()
		return self

	@writing
	def intersection_update(self, *others, uniq=True):
		pops = set()
		for other in others:
			if isinstance(other, collections.abc.Mapping):
				other = other.values()
			if type(other) not in (set, frozenset):
				other = frozenset(other)
			for i, v in enumerate(self):
				if v not in other:
					pops.add(i)
		self.pops(pops)
		if uniq:
			self.uniq()
		return self

	@writing
	def difference_update(self, *others, uniq=False):
		pops = set()
		for other in others:
			if isinstance(other, collections.abc.Mapping):
				other = other.values()
			if type(other) not in (set, frozenset):
				other = frozenset(other)
			for i, v in enumerate(self):
				if v in other:
					pops.add(i)
		self.pops(pops)
		if uniq:
			self.uniq()
		return self

	@writing
	def symmetric_difference_update(self, other):
		data = set(self)
		if isinstance(other, collections.abc.Mapping):
			other = other.values()
		if type(other) not in (set, frozenset):
			other = frozenset(other)
		data.symmetric_difference_update(other)
		self.fill(data)
		self._frozenset = data
		return self

	# Clips all values in list to input boundaries.
	@writing_with(view=True, order=True)
	def clip(self, a, b=None):
		if b is None:
			b = -a
		if a > b:
			a, b = b, a
		for arr in self.views:
			np.clip(arr, a, b, out=arr)
		return self

	# Casting values to various types.
	@reading
	def real(self):
		if not self:
			return []
		temp = np.real(np.concatenate(self.views))
		return self.__class__(temp)

	@reading
	def imag(self):
		if not self:
			return []
		temp = np.imag(self.complex().view())
		return self.__class__(temp)

	@reading
	def float(self):
		if not self:
			return []
		temp = np.concatenate(self.imag, dtype=np.float64)
		return self.__class__(temp)

	@reading
	def complex(self):
		if not self:
			return []
		temp = np.concatenate(self.imag, dtype=np.complex128)
		return self.__class__(temp)

	@reading
	def mpf(self):
		import mpmath.mpf as mpf
		return self.__class__(map(mpf, self))