import math
from functools import reduce

def compute_gcd(arr):
    if not arr:
        return 0
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    return reduce(gcd, arr)

def approximate_gcd(arr, min_value=8):
    if not arr:
        return 0
    
    # Check if any element is >= min_value
    has_element_above_min = any(x >= min_value for x in arr)
    if not has_element_above_min:
        return compute_gcd(arr)
    
    # Collect non-zero elements
    non_zero = [x for x in arr if x != 0]
    if not non_zero:
        return 0  # All elements are zero
    
    # Generate all possible divisors >= min_value from non-zero elements
    divisors = set()
    for x in non_zero:
        x_abs = abs(x)
        # Find all divisors of x_abs
        for i in range(1, int(math.isqrt(x_abs)) + 1):
            if x_abs % i == 0:
                if i >= min_value:
                    divisors.add(i)
                counterpart = x_abs // i
                if counterpart >= min_value:
                    divisors.add(counterpart)
    
    # If there are no divisors >= min_value, return the GCD of all elements
    if not divisors:
        return compute_gcd(arr)
    
    # Sort divisors in descending order
    sorted_divisors = sorted(divisors, reverse=True)
    
    max_count = 0
    candidates = []
    
    # Find the divisor(s) with the maximum count of divisible elements
    for d in sorted_divisors:
        count = 0
        for x in arr:
            if x % d == 0:
                count += 1
        if count > max_count:
            max_count = count
            candidates = [d]
        elif count == max_count:
            candidates.append(d)
    
    # Now find the maximum GCD among the candidates
    max_gcd = 0
    for d in candidates:
        elements = [x for x in arr if x % d == 0]
        current_gcd = compute_gcd(elements)
        if current_gcd > max_gcd:
            max_gcd = current_gcd
    
    return max_gcd if max_gcd >= min_value else compute_gcd(arr)


def approximate_gcd_2(arr, min_value=8):
    N = len(arr)
    # A set to store all candidate divisors (d) we might consider.
    candidate_divisors = set()
    
    # For each number in the array (skip 0 for now since 0 is divisible by any d)
    for x in arr:
        if x == 0:
            continue
        # Get all divisors of |x|
        divisors = get_divisors(abs(x))
        # Only add those that are at least min_value.
        for d in divisors:
            if d >= min_value:
                candidate_divisors.add(d)
    
    best_d = None
    best_outlier_count = N  # worst-case: drop everything
    # Iterate over candidate divisors
    for d in candidate_divisors:
        # Count how many numbers are divisible by d.
        support = 0
        for x in arr:
            # Note: by definition, 0 is considered divisible by any nonzero d.
            if x == 0 or x % d == 0:
                support += 1
        outliers = N - support
        # Check if this candidate is better:
        #  (a) fewer outliers, or (b) same outliers but a larger d.
        if outliers < best_outlier_count or (outliers == best_outlier_count and d > best_d):
            best_outlier_count = outliers
            best_d = d
            
    return best_d

def get_divisors(n):
    # Returns all positive divisors of n.
    divisors = set()
    i = 1
    while i*i <= n:
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
        i += 1
    return divisors




import time, random

# Test inputs
inputs = [[random.randint(10, 10000) for i in range(1000)] for j in range(100)]

def compare_functions(func1, func2, inputs, num_runs=10):
    differences = []

    # Compare outputs
    for input_value in inputs:
        output1 = func1(input_value)
        output2 = func2(input_value)
        if output1 != output2:
            differences.append((input_value, output1, output2))
    
    # Measure execution time
    def measure_time(func, inputs):
        start_time = time.time()
        for _ in range(num_runs):
            for input_value in inputs:
                func(input_value)
        end_time = time.time()
        return end_time - start_time

    time1 = measure_time(func1, inputs)
    time2 = measure_time(func2, inputs)
    time_difference = time1 - time2

    return differences, time_difference

# Run the comparison
differences, time_difference = compare_functions(approximate_gcd, approximate_gcd_2, inputs)

# Print the results
if differences:
    print("Differences found:")
    for input_value, output1, output2 in differences:
        print(f"Input: {input_value}, Function1 Output: {output1}, Function2 Output: {output2}")
else:
    print("No differences found.")

print(f"Difference in execution time: {time_difference:.6f} seconds")
