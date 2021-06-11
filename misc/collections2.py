import numpy, itertools, collections, concurrent.futures
np = numpy
from itertools import repeat
from collections import deque


class Dummy(BaseException):
    __slots__ = ()
    __bool__ = lambda: False

def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)


# Creates a nested tuple from a nested list.
_nested_tuple = lambda a: tuple(_nested_tuple(i) if isinstance(i, collections.abc.MutableSequence) else i for i in a)
nested_tuple = lambda a: _nested_tuple(a) if isinstance(a, collections.abc.Sequence) and type(a) not in (str, bytes) and a[0] != a else a


import numpy, itertools, collections, concurrent.futures
np = numpy
from itertools import repeat
from collections import deque


# Creates a nested tuple from a nested list.
_nested_tuple = lambda a: tuple(_nested_tuple(i) if isinstance(i, collections.abc.MutableSequence) else i for i in a)
nested_tuple = lambda a: _nested_tuple(a) if isinstance(a, collections.abc.Sequence) and type(a) not in (str, bytes) and a[0] != a else a


class alist(collections.abc.MutableSequence, collections.abc.Callable):

    """Custom list-like data structure that incorporates the functionality of numpy arrays, but allocates more space on the ends in order to have faster insertion."""

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
    def __init__(self, *args, fromarray=True, **void):
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
        elif fromarray and isinstance(iterable, np.ndarray):
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
            if fromarray:
                size = self.size
                self.offs = 0
            else:
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
        return self.data, self.offs, self.size

    def __setstate__(self, s):
        if type(s) is tuple:
            if len(s) == 2:
                if s[0] is None:
                    for k, v in s[1].items():
                        setattr(self, k, v)
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
            return self.pops(range(*s))
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
        return self.__class__(self.view.copy())

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
    def insort(self, value, key=None, sort=True):
        if self.data is None:
            self.__init__((value,))
            return self
        if not sort:
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
    def remove(self, value, count=None, key=None, sort=False, last=False):
        pops = self.search(value, key, sort, force=True)
        if count:
            if last:
                pops = pops[-count:]
            else:
                pops = pops[:count]
        if pops:
            self.pops(pops, force=True)
        return self

    discard = remove

    # Removes all duplicate values from the list.
    @blocking
    def removedups(self, sort=True):
        if self.data is not None:
            if sort:
                try:
                    temp = np.unique(self.view)
                except:
                    temp = sorted(set(map(nested_tuple, self.view)))
            elif sort is None:
                temp = tuple(set(self.view))
            else:
                temp = deque()
                found = set()
                for x in self.view:
                    y = nested_tuple(x)
                    if y not in found:
                        found.add(y)
                        temp.append(x)
            self.size = len(temp)
            self.offs = (len(self.data) - self.size) // 3
            self.view[:] = temp
        return self

    uniq = unique = removedups

    # Returns first matching value in list.
    @waiting
    def index(self, value, key=None, sort=False):
        return self.search(value, key, sort, force=True)[0]

    # Returns last matching value in list.
    @waiting
    def rindex(self, value, key=None, sort=False):
        return self.search(value, key, sort, force=True)[-1]

    # Returns indices representing positions for all instances of the target found in list, using binary search when applicable.
    @waiting
    def search(self, value, key=None, sort=False):
        if key is None:
            if sort and self.size > self.minsize:
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
        if sort:
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
        self.__init__(np.concatenate([value, self.view]))
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
        self.__init__(np.concatenate([self.view, value]))
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
        self.__init__(data if data is not None else self.view, fromarray=False)
        return self

    # Removes items according to an array of indices.
    @blocking
    def delitems(self, iterable, keep=False):
        iterable = self.to_iterable(iterable, force=True)
        if len(iterable) < 1:
            if keep:
                return self.__class__()
            return self
        if len(iterable) == 1:
            temp = self.pop(iterable[0], force=True)
            if keep:
                return self.__class__((temp,))
            return self
        indices = np.asarray(iterable, dtype=np.int32)
        if keep:
            temp2 = self.view[indices].copy()
        temp = np.delete(self.view, indices)
        self.size = len(temp)
        if self.data is not None:
            self.offs = (len(self.data) - self.size) // 3
            self.view[:] = temp
        else:
            self.reconstitute(temp, force=True)
        if keep:
            return self.__class__(temp2)
        return self

    pops = delitems

hlist = alist
arange = lambda *args, **kwargs: alist(range(*args, **kwargs))
afull = lambda size, n=0: alist(repeat(n, size))
azero = lambda size: alist(repeat(0, size))


class cdict(dict):
    
    """Class-based dictionary, with attributes corresponding to keys."""

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


class fdict(cdict):

    """A dict with key-value pairs fed from more dict-like objects."""

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
        

class demap(collections.abc.Mapping):
    
    """Double ended mapping, indexable from both sides."""

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
            
            
class UniversalSet(collections.abc.Set):
    
    """The Universal Set. Contains everything."""

    __slots__ = ()

    __str__ = lambda self: "ξ"
    __repr__ = lambda self: f"{self.__class__.__name__}()"
    __contains__ = lambda self, key: True
    __bool__ = lambda self: True
    __iter__ = lambda self: repeat(None)
    __len__ = lambda self: inf
    __call__ = lambda self, *args: self
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
    index = find = lambda self, obj: 0
    isdisjoint = lambda self, other: False

universal_set = UniversalSet()


def exclusive_range(range, *excluded):
    ex = frozenset(excluded)
    return tuple(i for i in range if i not in ex)

def exclusive_set(range, *excluded):
    ex = frozenset(excluded)
    return frozenset(i for i in range if i not in ex)


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


class fcdict(cdict):

    """A full-casefold string lookup mapping object."""

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


class mdict(cdict):

    """Dictionary with multiple assignable values per key."""

    __slots__ = ()

    count = lambda self: sum(len(v) for v in super().values())

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


class msdict(cdict):

    """Dictionary with multiple assignable values per key. Uses sets."""

    __slots__ = ()

    count = lambda self: sum(len(v) for v in super().values())

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
