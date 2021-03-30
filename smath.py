"""
Adds many useful math-related functions.
"""

import contextlib, concurrent.futures

# A context manager that enables concurrent imports.
class MultiThreadedImporter(contextlib.AbstractContextManager, contextlib.ContextDecorator):

    def __init__(self, glob=None):
        self.glob = glob
        self.exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        self.out = {}

    def __enter__(self):
        return self

    def __import__(self, *modules):
        for module in modules:
            self.out[module] = self.exc.submit(__import__, module)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type and exc_value:
            raise exc_value

    def close(self):
        for k in tuple(self.out):
            self.out[k] = self.out[k].result()
        glob = self.glob if self.glob is not None else globals()
        glob.update(self.out)
        self.exc.shutdown(True)

with MultiThreadedImporter() as importer:
    importer.__import__(
        "sys",
        "collections",
        "traceback",
        "time",
        "datetime",
        "pytz",
        "ast",
        "copy",
        "pickle",
        "io",
        "random",
        "math",
        "cmath",
        "fractions",
        "mpmath",
        "sympy",
        "shlex",
        "numpy",
        "colorsys",
        "re",
        "hashlib",
        "base64",
        "dateutil",
        "itertools",
        "colormath",
    )

from dateutil import parser as tparser
from sympy.parsing.sympy_parser import parse_expr
from colormath import color_objects, color_conversions


suppress = lambda *args, **kwargs: contextlib.suppress(BaseException) if not args and not kwargs else contextlib.suppress(*args + tuple(kwargs.values()))
closing = contextlib.closing
repeat = itertools.repeat


print_exc = lambda *args: sys.stdout.write(("\n".join(as_str(i) for i in args) + "\n" if args else "") + traceback.format_exc())

class Dummy(BaseException):
    __slots__ = ()
    __bool__ = lambda: False


loop = lambda x: repeat(None, x)

def try_int(i):
    try:
        return int(i)
    except:
        return i

np = numpy
array = np.array
deque = collections.deque

random.seed(random.randint(0, (1 << 32) - 1) - time.time_ns())
mp = mpmath.mp
mp.dps = 128

math.round = round

mpf = mpmath.mpf
mpf.__floordiv__ = lambda x, y: int(x / y)
mpf.__rfloordiv__ = lambda y, x: int(x / y)
mpf.__lshift__ = lambda x, y: x * (1 << y if type(y) is int else 2 ** y)
mpf.__rshift__ = lambda x, y: x // (1 << y if type(y) is int else 2 ** y)
mpf.__rlshift__ = lambda y, x: x * (1 << y if type(y) is int else 2 ** y)
mpf.__rrshift__ = lambda y, x: x * (1 << y if type(y) is int else 2 ** y)
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

# Not completely safe, but much safer than regular eval
safe_eval = lambda s: eval(as_str(s).replace("__", ""), {}, eval_const)

def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)

literal_eval = lambda s: ast.literal_eval(as_str(s).lstrip())

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


# s = "class Real(mpf):"
# for op in ("add", "sub", "mul", "truediv", "floordiv", "mod", "divmod", "pow"):
#     s += f"\n\t__{op}__=lambda self, x: super().__{op}__(mpf(x))"
# exec(s)


def exclusive_range(range, *excluded):
    ex = frozenset(excluded)
    return tuple(i for i in range if i not in ex)

def exclusive_set(range, *excluded):
    ex = frozenset(excluded)
    return frozenset(i for i in range if i not in ex)


class UniversalSet(collections.abc.Set):

    __slots__ = ()

    __str__ = lambda self: "ξ"
    __repr__ = lambda self: f"{self.__class__.__name__}()"
    __contains__ = lambda self, key: True
    __bool__ = lambda self: True
    __iter__ = lambda self: repeat(None)
    __len__ = lambda self: inf
    __le__ = lambda self, other: type(self) is type(other)
    __lt__ = lambda self, other: False
    __eq__ = lambda self, other: type(self) is type(other)
    __ne__ = lambda self, other: type(self) is not type(other)
    __gt__ = lambda self, other: type(self) is not type(other)
    __ge__ = lambda self, other: True
    __and__ = lambda self, other: other
    __or__ = lambda self, other: self
    __sub__ = lambda self, other: self
    __xor__ = lambda self, other: self
    index = lambda self, obj: 0
    isdisjoint = lambda self, other: False

universal_set = UniversalSet()


class alist(collections.abc.MutableSequence, collections.abc.Callable):

    """
custom list-like data structure that incorporates the functionality of numpy arrays but allocates more space on the ends in order to have faster insertion."""

    maxoff = (1 << 24) - 1
    minsize = 9
    __slots__ = ("hash", "block", "offs", "size", "data", "frozenset", "queries", "_index")

    # For thread-safety: Waits until the list is not busy performing an operation.
    def waiting(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force and type(self.block) is concurrent.futures.Future:
                self.block.result(timeout=12)
            return func(self, *args, **kwargs)
        return call

    # For thread-safety: Blocks the list until the operation is complete.
    def blocking(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force and type(self.block) is concurrent.futures.Future:
                self.block.result(timeout=12)
            self.block = concurrent.futures.Future()
            self.hash = None
            self.frozenset = None
            try:
                del self.queries
            except AttributeError:
                pass
            try:
                output = func(self, *args, **kwargs)
            except:
                try:
                    self.block.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass
                raise
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            return output
        return call

    # Init takes arguments and casts to a deque if possible, else generates as a single value. Allocates space equal to 3 times the length of the input iterable.
    def __init__(self, *args, fromarray=False, **void):
        fut = getattr(self, "block", None)
        self.block = concurrent.futures.Future()
        self.hash = None
        self.frozenset = None
        if fut:
            try:
                del self.queries
            except AttributeError:
                pass
            try:
                del self._index
            except AttributeError:
                pass
        if not args:
            self.offs = 0
            self.size = 0
            self.data = None
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            if fut:
                try:
                    fut.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass
            return
        elif len(args) == 1:
            iterable = args[0]
        else:
            iterable = args
        if issubclass(type(iterable), self.__class__) and iterable:
            self.offs = iterable.offs
            self.size = iterable.size
            if fromarray:
                self.data = iterable.data
            else:
                self.data = iterable.data.copy()
        elif fromarray:
            self.offs = 0
            self.size = len(iterable)
            self.data = iterable
        else:
            if not issubclass(type(iterable), collections.abc.Sequence) or issubclass(type(iterable), collections.abc.Mapping) or type(iterable) in (str, bytes):
                try:
                    iterable = deque(iterable)
                except TypeError:
                    iterable = [iterable]
            self.size = len(iterable)
            size = max(self.minsize, self.size * 3)
            self.offs = size // 3
            self.data = np.empty(size, dtype=object)
            self.view[:] = iterable
        if not fut or fut.done():
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            if fut:
                try:
                    fut.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass

    def __getstate__(self):
        if self.size <= 0:
            self.clear()
            self.data = None
            self.offs = 0
        return (self.view,)

    def __setstate__(self, s):
        if type(s) is tuple:
            if len(s) == 2:
                if s[0] is None:
                    for k, v in s[1].items():
                        setattr(self, k, v)
                    self.block = None
                    return
            elif len(s) == 3:
                self.data, self.offs, self.size = s
                self.hash = None
                self.frozenset = None
                try:
                    del self.queries
                except AttributeError:
                    pass
                self.block = None
                return
            elif len(s) == 1:
                self.data = s[0]
                self.offs = 0
                self.size = len(self.data) if self.data is not None else 0
                self.hash = None
                self.frozenset = None
                try:
                    del self.queries
                except AttributeError:
                    pass
                self.block = None
                return
        raise TypeError("Unpickling failed:", s)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        return getattr(self.__getattribute__("view"), k)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(dir(self.view))
        return data

    # Returns a numpy array representing the items currently "in" the list.
    @property
    def view(self):
        data = self.__getattribute__("data")
        if data is None:
            return []
        offs, size = [self.__getattribute__(i) for i in ("offs", "size")]
        return data[offs:offs + size]

    @waiting
    def __call__(self, arg=1, *void1, **void2):
        if arg == 1:
            return self.copy()
        return self * arg

    # Returns the hash value of the data in the list.
    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.view.tobytes())
        return self.hash

    def to_frozenset(self):
        if self.frozenset is None:
            self.frozenset = frozenset(self)
        return self.frozenset

    # Basic functions
    __str__ = lambda self: "[" + ", ".join(repr(i) for i in iter(self)) + "]"
    __repr__ = lambda self: f"{self.__class__.__name__}({tuple(self) if self.__bool__() else ''})"
    __bool__ = lambda self: self.size > 0

    # Arithmetic functions

    @blocking
    def __iadd__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.add(arr, iterable, out=arr)
        return self

    @blocking
    def __isub__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.subtract(arr, iterable, out=arr)
        return self

    @blocking
    def __imul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.multiply(arr, iterable, out=arr)
        return self

    @blocking
    def __imatmul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        temp = np.matmul(arr, iterable)
        self.size = len(temp)
        arr[:self.size] = temp
        return self

    @blocking
    def __itruediv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.true_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __ifloordiv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.floor_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __imod__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.mod(arr, iterable, out=arr)
        return self

    @blocking
    def __ipow__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.power(arr, iterable, out=arr)
        return self

    @blocking
    def __ilshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.left_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.multiply(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __irshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.right_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.divide(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __iand__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_and(arr, iterable, out=arr)
        return self

    @blocking
    def __ixor__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_xor(arr, iterable, out=arr)
        return self

    @blocking
    def __ior__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_or(arr, iterable, out=arr)
        return self

    @waiting
    def __neg__(self):
        return self.__class__(-self.view)

    @waiting
    def __pos__(self):
        return self

    @waiting
    def __abs__(self):
        d = self.data
        return self.__class__(np.abs(self.view))

    @waiting
    def __invert__(self):
        return self.__class__(np.invert(self.view))

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
        temp1 = self.view
        temp2 = self.to_iterable(other)
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
        temp = np.round(self.view, prec)
        return self.__class__(temp)

    @waiting
    def __trunc__(self):
        temp = np.trunc(self.view)
        return self.__class__(temp)

    @waiting
    def __floor__(self):
        temp = np.floor(self.view)
        return self.__class__(temp)

    @waiting
    def __ceil__(self):
        temp = np.ceil(self.view)
        return self.__class__(temp)

    __index__ = lambda self: self.view
    __radd__ = __add__
    __rsub__ = lambda self, other: -self + other
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @waiting
    def __rtruediv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.true_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rfloordiv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.floor_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rmod__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.mod(iterable, arr, out=arr)
        return temp

    @waiting
    def __rpow__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.power(iterable, arr, out=arr)
        return temp

    @waiting
    def __rlshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.left_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.multiply(iterable, np.power(2, arr), out=arr)
        return temp

    @waiting
    def __rrshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.right_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.divide(iterable, np.power(2, arr), out=arr)
        return temp

    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    # Comparison operations

    @waiting
    def __lt__(self, other):
        it = self.to_iterable(other)
        return self.view < other

    @waiting
    def __le__(self, other):
        it = self.to_iterable(other)
        return self.view <= other

    @waiting
    def __eq__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view == other
        except (TypeError, IndexError):
            return

    @waiting
    def __ne__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view != other
        except (TypeError, IndexError):
            return

    @waiting
    def __gt__(self, other):
        it = self.to_iterable(other)
        return self.view > other

    @waiting
    def __ge__(self, other):
        it = self.to_iterable(other)
        return self.view >= other

    # Takes ints, floats, slices and iterables for indexing
    @waiting
    def __getitem__(self, *args):
        if len(args) == 1:
            key = args[0]
            if type(key) in (float, complex):
                return get(self.view, key, 1)
            if type(key) is int:
                try:
                    key = key % self.size
                except ZeroDivisionError:
                    raise IndexError("Array List index out of range.")
                return self.view.__getitem__(key)
            if type(key) is slice:
                if key.step in (None, 1):
                    start = key.start
                    if start is None:
                        start = 0
                    stop = key.stop
                    if stop is None:
                        stop = self.size
                    if start >= self.size or stop <= start and (stop >= 0 or stop + self.size <= start):
                        return self.__class__()
                    temp = self.__class__(self, fromarray=True)
                    if start < 0:
                        if start < -self.size:
                            start = 0
                        else:
                            start %= self.size
                    if stop < 0:
                        stop %= self.size
                    elif stop > self.size:
                        stop = self.size
                    temp.offs += start
                    temp.size = stop - start
                    if not temp.size:
                        return self.__class__()
                    return temp
            return self.__class__(self.view.__getitem__(key), fromarray=True)
        return self.__class__(self.view.__getitem__(*args), fromarray=True)

    # Takes ints, slices and iterables for indexing
    @blocking
    def __setitem__(self, *args):
        if len(args) == 2:
            key = args[0]
            if type(key) is int:
                try:
                    key = key % self.size
                except ZeroDivisionError:
                    raise IndexError("Array List index out of range.")
            return self.view.__setitem__(key, args[1])
        return self.view.__setitem__(*args)

    # Takes ints and slices for indexing
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

    # Basic sequence functions
    __len__ = lambda self: self.size
    __length_hint__ = __len__
    __iter__ = lambda self: iter(self.view)
    __reversed__ = lambda self: iter(np.flip(self.view))

    def next(self):
        try:
            self._index = (self._index + 1) % self.size
        except AttributeError:
            self._index = 0
        return self[self._index]

    @waiting
    def __bytes__(self):
        return bytes(round(i) & 255 for i in self.view)

    def __contains__(self, item):
        try:
            if self.queries >= 8:
                return item in self.to_frozenset()
            if self.frozenset is not None:
                return item in self.frozenset
            self.queries += 1
        except AttributeError:
            self.queries = 1
        return item in self.view

    __copy__ = lambda self: self.copy()

    # Creates an iterable from an iterator, making sure the shape matches.
    def to_iterable(self, other, force=False):
        if not issubclass(type(other), collections.abc.Sequence) or issubclass(type(other), collections.abc.Mapping):
            try:
                other = list(other)
            except TypeError:
                other = [other]
        if len(other) not in (1, self.size) and not force:
            raise IndexError(f"Unable to perform operation on objects with size {self.size} and {len(other)}.")
        return other

    @blocking
    def clear(self):
        self.size = 0
        if self.data is not None:
            self.offs = len(self.data) // 3
        else:
            self.offs = 0
        return self

    @waiting
    def copy(self):
        return self.__class__(self.view)

    @waiting
    def sort(self, *args, **kwargs):
        return self.__class__(sorted(self.view, *args, **kwargs))

    @waiting
    def shuffle(self, *args, **kwargs):
        return self.__class__(shuffle(self.view, *args, **kwargs))

    @waiting
    def reverse(self):
        return self.__class__(np.flip(self.view))

    # Rotates the list a certain amount of steps, using np.roll for large rotate operations.
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
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = np.roll(self.view, steps)
        return self

    @blocking
    def rotateleft(self, steps):
        return self.rotate(-steps, force=True)

    rotateright = rotate

    # Re-initializes the list if the positional offsets are too large or if the list is empty.
    @blocking
    def isempty(self):
        if self.size:
            if abs(len(self.data) // 3 - self.offs) > self.maxoff:
                self.reconstitute(force=True)
            return False
        if len(self.data) > 4096:
            self.data = None
            self.offs = 0
        elif self.data is not None:
            self.offs = len(self.data) // 3
        return True

    # For compatibility with dict.get
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

    # Removes an item from the list. O(n) time complexity.
    @blocking
    def pop(self, index=None, *args):
        try:
            if index is None:
                return self.popright(force=True)
            if index >= len(self.data):
                return self.popright(force=True)
            elif index == 0:
                return self.popleft(force=True)
            index %= self.size
            temp = self.data[index + self.offs]
            if index > self.size >> 1:
                self.view[index:-1] = self.data[self.offs + index + 1:self.offs + self.size]
            else:
                self.view[1:index + 1] = self.data[self.offs:self.offs + index]
                self.offs += 1
            self.size -= 1
            return temp
        except LookupError:
            if not args:
                raise
            return args[0]

    # Inserts an item into the list. O(n) time complexity.
    @blocking
    def insert(self, index, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if index >= self.size:
            return self.append(value, force=True)
        elif index == 0:
            return self.appendleft(value, force=True)
        index %= self.size
        if index > self.size >> 1:
            if self.size + self.offs + 1 >= len(self.data):
                self.reconstitute(force=True)
            self.size += 1
            self.view[index + 1:] = self.view[index:-1]
        else:
            if self.offs < 1:
                self.reconstitute(force=True)
            self.size += 1
            self.offs -= 1
            self.view[:index] = self.view[1:index + 1]
        self.view[index] = value
        return self

    # Insertion sort using a binary search to find target position. O(n) time complexity.
    @blocking
    def insort(self, value, key=None, sorted=True):
        if self.data is None:
            self.__init__((value,))
            return self
        if not sorted:
            self.__init__(sorted(self, key=key))
        if key is None:
            return self.insert(np.searchsorted(self.view, value), value, force=True)
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

    # Removes all instances of a certain value from the list.
    @blocking
    def remove(self, value, key=None, sorted=False):
        pops = self.search(value, key, sorted, force=True)
        if pops:
            self.pops(pops, force=True)
        return self

    discard = remove

    # Removes all duplicate values from the list.
    @blocking
    def removedups(self, sorted=True):
        if sorted:
            try:
                temp = np.unique(self.view)
            except:
                temp = sorted(set(self.view))
        elif sorted is None:
            temp = tuple(set(self.view))
        else:
            temp = {}
            for x in self.view:
                if x not in temp:
                    temp[x] = None
            temp = tuple(temp.keys())
        self.size = len(temp)
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = temp
        return self

    uniq = unique = removedups

    # Returns first matching value in list.
    @waiting
    def index(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[0]

    # Returns last matching value in list.
    @waiting
    def rindex(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[-1]

    # Returns indices representing positions for all instances of the target found in list, using binary search when applicable.
    @waiting
    def search(self, value, key=None, sorted=False):
        if key is None:
            if sorted and self.size > self.minsize:
                i = np.searchsorted(self.view, value)
                if self.view[i] != value:
                    raise IndexError(f"{value} not found.")
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
                return self.__class__(np.arange(self.size, dtype=np.uint32)[self.view == value])
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
            pops = self.__class__(i for i, x in enumerate(self.view) if key(x) == value)
        if not pops:
            raise IndexError(f"{value} not found.")
        return pops

    find = findall = search

    # Counts the amount of instances of the target within the list.
    @waiting
    def count(self, value, key=None):
        if key is None:
            return sum(self.view == value)
        return sum(1 for i in self if key(i) == value)

    concat = lambda self, value: self.__class__(np.concatenate([self.view, value]))

    # Appends item at the start of the list, reallocating when necessary.
    @blocking
    def appendleft(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs <= 0:
            self.reconstitute(force=True)
        self.offs -= 1
        self.size += 1
        self.data[self.offs] = value
        return self

    # Appends item at the end of the list, reallocating when necessary.
    @blocking
    def append(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs + self.size >= len(self.data):
            self.reconstitute(force=True)
        self.data[self.offs + self.size] = value
        self.size += 1
        return self

    add = lambda self, value: object.__getattribute__(self, ("append", "appendleft")[random.randint(0, 1)])(value)
    appendright = append

    # Appends iterable at the start of the list, reallocating when necessary.
    @blocking
    def extendleft(self, value):
        if self.data is None or not self.size:
            self.__init__(reversed(value))
            return self
        value = self.to_iterable(reversed(value), force=True)
        if self.offs >= len(value):
            self.data[self.offs - len(value):self.offs] = value
            self.offs -= len(value)
            self.size += len(value)
            return self
        self.__init__(np.concatenate([value, self.view]), fromarray=True)
        return self

    # Appends iterable at the end of the list, reallocating when necessary.
    @blocking
    def extend(self, value):
        if self.data is None or not self.size:
            self.__init__(value)
            return self
        value = self.to_iterable(value, force=True)
        if len(self.data) - self.offs - self.size >= len(value):
            self.data[self.offs + self.size:self.offs + self.size + len(value)] = value
            self.size += len(value)
            return self
        self.__init__(np.concatenate([self.view, value]), fromarray=True)
        return self

    extendright = extend

    # Similar to str.join
    @waiting
    def join(self, iterable):
        iterable = self.to_iterable(iterable)
        temp = deque()
        for i, v in enumerate(iterable):
            try:
                temp.extend(v)
            except TypeError:
                temp.append(v)
            if i != len(iterable) - 1:
                temp.extend(self.view)
        return self.__class__(temp)

    # Similar to str.replace().
    @blocking
    def replace(self, original, new):
        view = self.view
        for i, v in enumerate(view):
            if v == original:
                view[i] = new
        return self

    # Fills list with value(s).
    @blocking
    def fill(self, value):
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = value
        return self

    # For compatibility with dict() attributes.
    keys = lambda self: range(len(self))
    values = lambda self: iter(self)
    items = lambda self: enumerate(self)

    # For compatibility with set() attributes.
    @waiting
    def isdisjoint(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().isdisjoint(other)

    @waiting
    def issubset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issubset(other)

    @waiting
    def issuperset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issuperset(other)

    @waiting
    def union(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().union(*args)

    @waiting
    def intersection(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().intersection(*args)

    @waiting
    def difference(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().difference(*args)

    @waiting
    def symmetric_difference(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().symmetric_difference(other)

    @blocking
    def update(self, *others, uniq=True):
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            self.extend(other, force=True)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def intersection_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v not in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def difference_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def symmetric_difference_update(self, other):
        data = set(self)
        if issubclass(other, collections.abc.Mapping):
            other = other.values()
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        data.symmetric_difference_update(other)
        self.__init__(data)
        return self

    # Clips all values in list to input boundaries.
    @blocking
    def clip(self, a, b=None):
        if b is None:
            b = -a
        if a > b:
            a, b = b, a
        arr = self.view
        np.clip(arr, a, b, out=arr)
        return self

    # Casting values to various types.

    @waiting
    def real(self):
        return self.__class__(np.real(self.view))

    @waiting
    def imag(self):
        return self.__class__(np.imag(self.view))

    @waiting
    def float(self):
        return self.__class__(float(i.real) for i in self.view)

    @waiting
    def complex(self):
        return self.__class__(complex(i) for i in self.view)

    @waiting
    def mpf(self):
        return self.__class__(mpf(i.real) for i in self.view)

    @waiting
    def sum(self):
        return np.sum(self.view)

    @waiting
    def mean(self):
        return np.mean(self.view)

    @waiting
    def product(self):
        return np.prod(self.view)

    prod = product

    # Reallocates list.
    @blocking
    def reconstitute(self, data=None):
        self.__init__(data if data is not None else self.view)
        return self

    # Removes items according to an array of indices.
    @blocking
    def delitems(self, iterable):
        iterable = self.to_iterable(iterable, force=True)
        if len(iterable) < 1:
            return self
        if len(iterable) == 1:
            return self.pop(iterable[0], force=True)
        temp = np.delete(self.view, np.asarray(iterable, dtype=np.int32))
        self.size = len(temp)
        if self.data is not None:
            self.offs = (len(self.data) - self.size) // 3
            self.view[:] = temp
        else:
            self.reconstitute(temp, force=True)
        return self

    pops = delitems

hlist = alist
arange = lambda *args, **kwargs: alist(range(*args, **kwargs))
afull = lambda size, n=0: alist(repeat(n, size))
azero = lambda size: alist(repeat(0, size))


# Class-based dictionary, with attributes corresponding to keys.
class cdict(dict):

    __slots__ = ()

    @classmethod
    def from_object(cls, obj):
        return cls((a, getattr(obj, a, None)) for a in dir(obj))

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: self.__class__.__name__ + ("((" + ",".join("(" + ",".join(repr(i) for i in item) + ")" for item in super().items()) + ("," if len(self) == 1 else "") + "))") if self else "()"
    __str__ = lambda self: super().__repr__()
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __call__ = lambda self, k: self.__getitem__(k)

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

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


# A dict with key-value pairs fed from more dict-like objects.
class fdict(cdict):

    __slots__ = ("_feed",)

    def get_feed(self):
        feed = object.__getattribute__(self, "_feed")
        if callable(feed):
            return feed()
        return feed

    def _keys(self):
        found = set()
        for k in super().keys():
            found.add(k)
            yield k
        for f in self.get_feed():
            for k in f:
                if k not in found:
                    found.add(k)
                    yield k

    def keys(self):
        try:
            self.get_feed()
        except AttributeError:
            return super().keys()
        return self._keys()

    __iter__ = lambda self: iter(super().keys())

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

    def _len_(self):
        size = len(self)
        try:
            self.get_feed()
        except AttributeError:
            return size
        for f in self.get_feed():
            try:
                size += f._len_()
            except AttributeError:
                size += len(f)
        return size

    def __getitem__(self, k):
        try:
            return super().__getitem__(k)
        except KeyError:
            pass
        try:
            self.get_feed()
        except AttributeError:
            raise KeyError(k)
        for f in self.get_feed():
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

# A full-casefold string lookup mapping object.
class fcdict(cdict):

    __slots__ = ()

    __init__ = lambda self, *args, **kwargs: super().__init__((full_prune(k), v) for k, v in dict(*args, **kwargs).items())
    __contains__ = lambda self, k: super().__contains__(k) or super().__contains__(full_prune(k))

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

# Dictionary with multiple assignable values per key.
class mdict(cdict):

    __slots__ = ()

    count = lambda self: sum(len(v) for v in super().values())

    def extend(self, k, v):
        try:
            values = super().__getitem__(k)
        except KeyError:
            return super().__setitem__(k, alist(v).uniq(sorted=False))
        return values.extend(v).uniq(sorted=False)

    def append(self, k, v):
        values = set_dict(super(), k, alist())
        if v not in values:
            values.append(v)

    add = append

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

# Dictionary with multiple assignable values per key. Uses sets.
class msdict(cdict):

    __slots__ = ()

    count = lambda self: sum(len(v) for v in super().values())

    def extend(self, k, v):
        try:
            values = super().__getitem__(k)
        except KeyError:
            return super().__setitem__(k, set(v))
        return values.update(v)

    def append(self, k, v):
        values = set_dict(super(), k, set())
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

# Double ended mapping, indexable from both sides.
class demap(collections.abc.Mapping):

    __slots__ = ("a", "b")

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

    clear = lambda self: (self.a.clear(), self.b.clear())
    __bool__ = lambda self: bool(self.a)
    __iter__ = lambda self: iter(self.a.items())
    __reversed__ = lambda self: reversed(self.a.items())
    __len__ = lambda self: self.b.__len__()
    __str__ = lambda self: self.a.__str__()
    __repr__ = lambda self: f"{self.__class__.__name__}({self.a.__repr__() if bool(self.b) else ''})"
    __contains__ = lambda self, k: k in self.a or k in self.b


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
    if type(x) is complex:
        return round(x.real, y) + round(x.imag, y) * 1j
    try:
        return math.round(x, y)
    except:
        pass
    return x

# Rounds a number to the nearest integer, with a probability determined by the fractional part.
def round_random(x):
    y = round_min(x)
    if type(y) is int:
        return y
    x, y = divmod(x, 1)
    if random.random() <= y:
        x += 1
    return int(x)

# Rounds x to the nearest multiple of y.
round_multiple = lambda x, y=1: round_min(math.round(x / y) * y) if y else x
# Randomly rounds x to the nearest multiple of y.
round_random_multiple = lambda x, y=1: round_min(round_random(x / y) * y) if y else x

# Returns integer ceiling value of x, for all complex x.
def ceil(x):
    with suppress(Exception):
        return math.ceil(x)
    if type(x) is complex:
        return ceil(x.real) + ceil(x.imag) * 1j
    with suppress(Exception):
        return math.ceil(x)
    return x

# Returns integer floor value of x, for all complex x.
def floor(x):
    with suppress(Exception):
        return math.floor(x)
    if type(x) is complex:
        return floor(x.real) + floor(x.imag) * 1j
    with suppress(Exception):
        return math.floor(x)
    return x

# Returns integer truncated value of x, for all complex x.
def trunc(x):
    with suppress(Exception):
        return math.trunc(x)
    if type(x) is complex:
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
    return random.randint(floor(min(x, y)), ceil(max(x, y)) - 1) + z

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
    if type(x) is str:
        if "." in x:
            x = x.strip("0")
            if len(x) > 8:
                x = mpf(x)
            else:
                x = float(x)
        else:
            try:
                return int(x)
            except ValueError:
                return float(x)
    if type(x) is int:
        return x
    if type(x) is not complex:
        if is_finite(x):
            if type(x) is globals().get("mpf", None):
                y = int(x)
                if x == y:
                    return y
                f = float(x)
                if str(x) == str(f):
                    return f
            else:
                y = math.round(x)
                if x == y:
                    return int(y)
        return x
    else:
        if x.imag == 0:
            return round_min(x.real)
        else:
            return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)


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


def _predict_next(seq):
	if len(seq) < 2:
		return
	if np.min(seq) == np.max(seq):
		return seq[0]
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
	seq = np.asarray(seq, dtype=np.float64)
	for i in range(8, 1 + max(8, min(len(seq), limit))):
		temp = _predict_next(seq[-i:])
		if temp is not None:
			return temp


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
    while n > ratio:
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
def lim_str(s, maxlen=10):
    if maxlen is None:
        return s
    if type(s) is not str:
        s = str(s)
    over = (len(s) - maxlen) / 2
    if over > 0:
        half = len(s) / 2
        s = s[:ceil(half - over - 1)] + ".." + s[ceil(half + over + 1):]
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


# Uses an optional interpolation mode to get a certain position in an iterable.
def get(v, i, mode=1):
    size = len(v)
    i = i.real + i.imag * size
    if i == int(i) or mode == 0:
        return v[round(i) % size]
    elif mode > 0 and mode < 1:
        return get(v, i, 0) * (1 - mode) + mode * get(v, i, 1)
    elif mode == 1:
        return v[floor(i) % size] * (1 - i % 1) + v[ceil(i) % size] * (i % 1)
    return get(v, i, floor(mode)) * (1 - mode % 1) + (mode % 1) * get(v, i, ceil(mode))


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
    t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
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
def iter2str(it, key=None, limit=1728, offset=0, left="[", right="]", sep=" "):
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
        if type(t) is dict:
            t = int_key(t)
        c[k] = t
    return c


# Time functions
class DynamicDT:

    __slots__ = ("_dt", "_offset", "_ts")

    def __getstate__(self):
        return self.timestamp(), getattr(self, "tzinfo", None)

    def __setstate__(self, s):
        if len(s) == 2:
            ts, tzinfo = s
            offs, ots = divmod(ts, 12622780800)
            self._dt = datetime.datetime.fromtimestamp(ots)
            if tzinfo:
                self._dt = self._dt.replace(tzinfo=tzinfo)
            self._offset = round(offs * 400)
            return
        raise TypeError("Unpickling failed:", s)

    def __init__(self, *args, **kwargs):
        if type(args[0]) is bytes:
            self._dt = datetime.datetime(args[0])
            return
        offs, y = divmod(args[0], 400)
        y += 2000
        offs *= 400
        offs -= 2000
        self._dt = datetime.datetime(y, *args[1:], **kwargs)
        self.set_offset(offs)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        return getattr(self._dt, k)

    def __str__(self):
        y = self.year_repr()
        return y + str(self._dt)[4:]

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
            return self.__class__.fromtimestamp(self.timestamp() + other)
        ts = (self._dt + other).timestamp()
        if abs(self.offset()) >= 25600:
            ts = round(ts)
        return self.__class__.fromtimestamp(ts + self.offset() * 31556952)
    __radd__ = __add__

    def __sub__(self, other):
        if type(other) not in (datetime.timedelta, datetime.datetime, datetime.date, self.__class__):
            return self.__class__.fromtimestamp(self.timestamp() - other)
        out = (self._dt - other)
        ts = getattr(out, "timestamp", None)
        if ts is None:
            return self.__class__.fromdatetime(out).set_offset(self.offset())
        ts = ts()
        if abs(self.offset()) >= 25600:
            ts = round(ts)
        return self.__class__.fromtimestamp(ts + self.offset() * 31556952)
    __rsub__ = __sub__

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
        offs, ots = divmod(ts, 12622780800)
        if abs(offs) >= 64:
            ots = round(ots)
            ts = round(ts)
        d = utc_ft(ots)
        dt = cls(*d.timetuple()[:6], d.microsecond)
        dt._offset += round(offs * 400)
        dt._ts = ts
        return dt

    @classmethod
    def fromtimestamp(cls, ts):
        offs, ots = divmod(ts, 12622780800)
        if abs(offs) >= 4:
            ots = round(ots)
            ts = round(ts)
        d = datetime.datetime.fromtimestamp(ots)
        dt = cls(*d.timetuple()[:6], d.microsecond)
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


ts_us = lambda: time.time_ns() // 1000
utc = lambda: time.time_ns() / 1e9
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
    elif month == 3:
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
    seconds = round(getattr(t2, "second", 0) - getattr(t1, "second", 0) + (getattr(t2, "microsecond", 0) - getattr(t1, "microsecond", 0)) / 1000000, 6)
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

# Experimental invisible Zero-Width character encoder.
ZeroEnc = "\xad\u061c\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\u2065\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\ufe0f\ufeff"
__zeroEncoder = demap({chr(i + 97): c for i, c in enumerate(ZeroEnc)})
__zeroEncode = "".maketrans(dict(__zeroEncoder.a))
__zeroDecode = "".maketrans(dict(__zeroEncoder.b))
is_zero_enc = lambda s: (s[0] in ZeroEnc) if s else None
zwencode = lambda s: as_str(s).casefold().translate(__zeroEncode)
zwdecode = lambda s: as_str(s).casefold().translate(__zeroDecode)
__zeroRemover = {c: "" for c in ZeroEnc}
__zeroRemoverTrans = "".maketrans(__zeroRemover)
zwremove = lambda s: as_str(s).translate(__zeroRemoverTrans)


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
unfont = lambda s: str(s).translate(__unfont)

DIACRITICS = {
    "ÀÁÂÃÄÅĀĂĄ": "A",
    "Æ": "AE",
    "ÇĆĈĊČ": "C",
    "ĎĐ": "D",
    "ÈÉÊËĒĔĖĘĚ": "E",
    "ĜĞĠĢ": "G",
    "ĤĦ": "H",
    "ÌÍÎÏĨĪĬĮİ": "I",
    "Ĳ": "IJ",
    "Ĵ": "J",
    "Ķ": "K",
    "ĹĻĽĿŁ": "L",
    "ÑŃŅŇŊ": "N",
    "ÒÓÔÕÖØŌŎŐ": "O",
    "Œ": "OE",
    "ŔŖŘ": "R",
    "ŚŜŞŠ": "S",
    "ŢŤŦ": "T",
    "ÙÚÛÜŨŪŬŮŰŲ": "U",
    "Ŵ": "W",
    "ÝŶŸ": "Y",
    "ŹŻŽ": "Z",
    "àáâãäåāăą": "a",
    "æ": "ae",
    "çćĉċč": "c",
    "ďđ": "d",
    "èéêëðēĕėęě": "e",
    "ĝğġģ": "g",
    "ĥħ": "h",
    "ìíîïĩīĭįı": "i",
    "ĳ": "ij",
    "ĵ": "j",
    "ķĸ": "k",
    "ĺļľŀł": "l",
    "ñńņňŉŋ": "n",
    "òóôõöøōŏő": "o",
    "œ": "oe",
    "þ": "p",
    "ŕŗř": "r",
    "śŝşšſ": "s",
    "ß": "ss",
    "ţťŧ": "t",
    "ùúûüũūŭůűų": "u",
    "ŵ": "w",
    "ýÿŷ": "y",
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
extra_zalgos = (
    range(768, 880),
    range(1155, 1162),
    exclusive_range(range(1425, 1478), 1470, 1472, 1475),
    range(1552, 1560),
    range(1619, 1632),
    exclusive_range(range(1750, 1774), 1757, 1758, 1765, 1766, 1769),
    exclusive_range(range(2260, 2304), 2274),
    range(7616, 7627),
    (8432,),
    range(11744, 11776),
    (42607,), range(42612, 42622), (42654, 42655),
    range(65056, 65060),
)
zalgo_array = np.concatenate(extra_zalgos)
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

full_prune = lambda s: unicode_prune(s).translate(__qtrans).casefold()


# A fuzzy substring search that returns the ratio of characters matched between two strings.
def fuzzy_substring(sub, s, match_start=False, match_length=True):
    if not match_length and s in sub:
        return 1
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
    ratio = max(0, min(1, match / len(s)))
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
hex2bytes = lambda b: bytes.fromhex(as_str(b))

# Converts a bytes object to a base64 string.
def bytes2b64(b, alt_char_set=False):
    if type(b) is str:
        b = b.encode("utf-8")
    b = base64.b64encode(b)
    if alt_char_set:
        b = b.replace(b"=", b"-").replace(b"/", b".")
    return b

# Converts a base 64 string to a bytes object.
def b642bytes(b, alt_char_set=False):
    if type(b) is str:
        b = b.encode("utf-8")
    if alt_char_set:
        b = b.replace(b"-", b"=").replace(b".", b"/")
    b = base64.b64decode(b)
    return b

if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
    randbytes = random.randbytes
else:
    randbytes = lambda size: (np.random.random_sample(size) * 256).astype(np.uint8).tobytes()

# SHA256 operations: base64 and base16.
shash = lambda s: as_str(base64.b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).replace(b"/", b"-").rstrip(b"="))
hhash = lambda s: bytes2hex(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), space=False)
ihash = lambda s: int.from_bytes(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest(), "little") % 4294967296 - 2147483648

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