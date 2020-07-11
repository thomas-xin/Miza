from smath import *

size = int(1e8)
x = range(size)
print(f"Starting test with numbers 0 to {size - 1}...")

def f(func, *args, **kwargs):
    t = utc()
    temp = func(*args, **kwargs)
    s = str(round(utc() - t, 4))
    print(s, end="\t" if len(s) < 8 else "")
    return temp

print("Initializing Lists...", end="\t")
a = f(list, x)
b = f(deque, x)
c = f(np.array, x, dtype=object)
d = f(hlist, x)

print("\nGet first item", end="\t\t")
f(a.__getitem__, 0)
f(b.__getitem__, 0)
f(c.__getitem__, 0)
f(d.__getitem__, 0)

print("\nGet last item", end="\t\t")
f(a.__getitem__, -1)
f(b.__getitem__, -1)
f(c.__getitem__, -1)
f(d.__getitem__, -1)

print("\nGet middle item", end="\t\t")
f(a.__getitem__, size >> 1)
f(b.__getitem__, size >> 1)
f(c.__getitem__, size >> 1)
f(d.__getitem__, size >> 1)

print("\nAppend at start", end="\t\t")
f(a.insert, 0, -1)
f(b.appendleft, -1)
c = f(np.insert, c, 0, -1)
f(d.appendleft, -1)

print("\nAppend at end", end="\t\t")
f(a.append, size)
f(b.append, size)
c = f(np.append, c, size)
f(d.append, size)

print("\nAppend in middle", end="\t")
f(a.insert, size >> 1, 0)
f(b.insert, size >> 1, 0)
c = f(np.insert, c, size >> 1, 0)
f(d.insert, size >> 1, 0)

print("\nDivide by 2", end="\t\t")
a = f(list, (i / 2 for i in a))
b = f(deque, (i / 2 for i in b))
c = f(np.true_divide, c, 2)
f(d.__itruediv__, 2)

print("\nPower of 2", end="\t\t")
a = f(list, (i * i for i in a))
b = f(deque, (i * i for i in b))
c = f(np.power, c, 2)
f(d.__ipow__, 2)