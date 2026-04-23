import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc import x_math
import sympy

cases = [
    (sympy.Rational(1, 3),  "1/3"),
    (sympy.Rational(1, 7),  "1/7"),
    (sympy.Rational(1, 6),  "1/6"),
    (sympy.Rational(1, 13), "1/13"),
    (sympy.Rational(1, 97), "1/97 (primitive root)"),
    (sympy.Rational(1, 50), "1/50 (terminating)"),
    (sympy.Rational(1, 2),  "1/2 (terminating)"),
]
for r, desc in cases:
    t0 = time.perf_counter()
    out = x_math.simplify_recurring(r, prec=200)
    dt = (time.perf_counter() - t0) * 1000
    s = str(out)[:80]
    print(f"{desc:28s} {dt:8.2f}ms -> {s}")

# Pathological: smooth lambda(q)
q = 10**12 - 1  # period exactly 12, but lambda has MANY divisors
r = sympy.Rational(1, q)
t0 = time.perf_counter()
out = x_math.simplify_recurring(r, prec=200)
dt = (time.perf_counter() - t0) * 1000
print(f"1/(10^12-1) {' ':16s} {dt:8.2f}ms -> {str(out)[:80]}")

q = 3 * 7 * 11 * 13 * 37
r = sympy.Rational(1, q)
t0 = time.perf_counter()
out = x_math.simplify_recurring(r, prec=400)
dt = (time.perf_counter() - t0) * 1000
print(f"1/{q:<26d} {dt:8.2f}ms -> {str(out)[:80]}")

# Really nasty one: denominator with smooth lambda and large period
q = 9999999999999  # 10^13 - 1 = 53 * 79 * 265371653
r = sympy.Rational(1, q)
t0 = time.perf_counter()
out = x_math.simplify_recurring(r, prec=50)
dt = (time.perf_counter() - t0) * 1000
print(f"1/(10^13-1) short prec {' ':5s} {dt:8.2f}ms -> {str(out)[:80]}")
