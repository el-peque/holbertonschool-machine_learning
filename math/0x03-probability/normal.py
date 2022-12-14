#!/usr/bin/env python3
"""class Normal"""


class Normal:
    """Represents a normal distribution"""
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initiates Normal"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            stddev = sum([(i - self.mean) ** 2 for i in data]) / len(data)
            stddev = stddev ** 0.5
            self.stddev = stddev

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of 'successes'"""
        k = int(k)
        if k < 0:
            return 0

        λ = self.lambtha
        return ((self.e ** (-λ)) * (λ ** k) / Poisson.factorial(k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0
        cdf = sum([self.pmf(i) for i in range(k + 1)])
        return cdf

    @staticmethod
    def factorial(n):
        """Calculates the factorial of a number"""
        if n <= 1:
            return 1
        else:
            return (n * Poisson.factorial(n - 1))
