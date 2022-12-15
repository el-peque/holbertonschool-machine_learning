#!/usr/bin/env python3
"""class Binomial"""


class Binomial:
    """Represents a binomial distribution"""
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, n=1, p=0.5):
        """Initiates Binomial"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = sum([(x - mean) ** 2 for x in data]) / len(data)
            self.p = ((1 - (var / mean)))
            self.n = round(mean / self.p)
            self.p = float(mean / self.n)

    def cdf(self, x):
        """Calculates the value of the CDF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(x + 1)])

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of 'successes'"""
        k = int(k)
        if k < 0:
            return 0
        p = self.p
        n = self.n
        n_f = Binomial.factorial(n)
        k_f = Binomial.factorial(k)
        nk_f = Binomial.factorial(n - k)
        return (n_f / (k_f * nk_f)) * (p ** k) * ((1 - p) ** (n - k))

    @staticmethod
    def factorial(n):
        """Calculates the factorial of a number"""
        if n <= 1:
            return 1
        else:
            return (n * Binomial.factorial(n - 1))
