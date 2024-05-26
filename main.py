import pandas as pd
import itertools
import math
import time
import xlsxwriter
from sympy import isprime

def read_random_numbers(filename):
    """Read random numbers from an Excel file."""
    df = pd.read_excel(filename)
    return df['Random Numbers'].tolist()

def save_factorizations_to_excel(data, filename):
    """Save the prime factorizations to an Excel file."""
    df = pd.DataFrame(data, columns=['Number', 'Prime Factors', 'Correct'])
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    print(f"{filename} has been saved.")

def smooth_factor(q, factor_base, num_primes):
    """Checks if a number is smooth relative to a given base of prime numbers."""
    exponent_vector = [0] * num_primes
    for i, p in enumerate(factor_base[1:], 1):  # Skip -1
        while q % p == 0:
            q //= p
            exponent_vector[i] += 1
    if q == 1:  # q should be fully factorized by the factor base
        return exponent_vector
    return None

def z2_gaussian_elimination(matrix):
    """Performs Gaussian elimination over Z/2Z."""
    row, col = 0, 0
    while row < len(matrix) and col < len(matrix[0]):
        pivot = row
        for i in range(row + 1, len(matrix)):
            if matrix[i][col]:
                pivot = i
                break
        if matrix[pivot][col]:
            matrix[pivot], matrix[row] = matrix[row], matrix[pivot]
            for i in range(len(matrix)):
                if i != row and matrix[i][col]:
                    for j in range(col, len(matrix[i])):
                        matrix[i][j] ^= matrix[row][j]
            row += 1
        col += 1
    return matrix

def jacobi_symbol(a, n):
    """Computes the Jacobi symbol (a/n)."""
    if n % 2 == 0:
        raise ValueError("n must be odd")
    if a == 0:
        return 0
    result = 1
    if a < 0:
        a = -a
        if n % 4 == 3:
            result = -result
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == n % 4 == 3:
            result = -result
        a %= n
    if n == 1:
        return result
    else:
        return 0

def integer_sqrt(n):
    """Computes the integer square root."""
    if n < 0:
        raise ValueError("square root of negative number")
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def gcd(a, b):
    """Greatest common divisor (GCD)."""
    while b != 0:
        a, b = b, a % b
    return a

def primes_sieve(limit):
    """Generate prime numbers up to a specified limit."""
    primes = []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, limit + 1):
        if sieve[p]:
            primes.append(p)
            for i in range(p * p, limit + 1, p):
                sieve[i] = False
    return primes

def cfrac(n, k=1):
    """Continued fraction factorization algorithm."""
    def find_multiplier():
        """Finds a multiplier for input to the factorization algorithm."""
        choices = {}
        prime_list = primes_sieve(1000)
        for k in range(1, 1000):
            if jacobi_symbol(k * n, 3) >= 0 and jacobi_symbol(k * n, 5) >= 0:
                count = 0
                for p in prime_list:
                    if p <= 5:
                        continue
                    if p > 31:
                        break
                    if jacobi_symbol(k * n, p) >= 0:
                        count += 1
                if count not in choices:
                    choices[count] = [k]
                else:
                    choices[count].append(k)
        max_count = max(choices)
        return min(choices[max_count])

    def cfrac_aq_pairs(n):
        """Generate tuples (i, A_{i - 1}, Q_i) for i > 0."""
        g = integer_sqrt(n)
        A0, A1 = 0, 1
        Q0, Q1 = n, 1
        P0 = 0
        r0 = g

        for i in itertools.count():
            if Q1 == 0:
                break
            q = (g + P0) // Q1
            r1 = g + P0 - q * Q1
            A2 = (q * A1 + A0) % n
            P1 = g - r1
            Q2 = Q0 + q * (r1 - r0)

            if i > 0:
                yield (i, A1, Q1)

            A0, A1 = A1, A2
            Q0, Q1 = Q1, Q2
            P0 = P1
            r0 = r1

    B = int(0.5 * math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) + 1
    prime_list = primes_sieve(B)

    if k == 0:
        k = find_multiplier()
    kn = k * n

    factors = []
    for p in prime_list:
        while n % p == 0:
            factors.append(p)
            n //= p

    factor_base = [-1]
    for p in prime_list:
        if p == 2 or jacobi_symbol(kn, p) >= 0:
            factor_base.append(p)

    num_primes = len(factor_base)

    exponent_matrix = []
    relations = []
    a_list = []

    aq_pairs = cfrac_aq_pairs(kn)

    while len(relations) <= num_primes:
        try:
            (i, a, q) = next(aq_pairs)
        except StopIteration:
            break

        if i % 2 == 1:
            q *= -1

        exponent_vector = smooth_factor(q, factor_base, num_primes)
        if exponent_vector:
            exponent_matrix.append(exponent_vector)
            relations.append((a, exponent_vector))
            a_list.append(a)

    kernel = z2_gaussian_elimination(exponent_matrix)

    for i in range(len(kernel)):
        y = 1
        x2_exponents = [0] * num_primes
        for j in range(len(kernel[i])):
            if kernel[i][j]:
                y = (a_list[j] * y) % n
                for f in range(num_primes):
                    x2_exponents[f] += exponent_matrix[j][f]

        x = 1
        for j in range(num_primes):
            x *= factor_base[j] ** (x2_exponents[j] // 2)

        for val in [x - y, x + y]:
            d = gcd(val, n)
            if 1 < d < n:
                factors.append(d)
                n //= d  # Reduce n
    if n > 1:
        factors.append(n)  # Add the remaining prime number
    return factors

def verify_factors(number, factors):
    """Verify if the product of the factors equals the original number."""
    product = 1
    for factor in factors:
        product *= factor
    return product == number

def process_numbers_and_save(filename):
    random_numbers = read_random_numbers('random_numbers10000.xlsx')
    factorization_results = []

    for number in random_numbers:
        if number < 2:
            factors = [number]
        elif isprime(number):
            factors = [number]
        else:
            factors = cfrac(number)
        is_correct = verify_factors(number, factors)
        factorization_results.append([number, factors, is_correct])

    save_factorizations_to_excel(factorization_results, filename)

if __name__ == "__main__":
    start_time = time.time()
    process_numbers_and_save('factorized_numbers10000.xlsx')
    #print(cfrac(338031018))
    print("       %s seconds       " % (time.time() - start_time))


