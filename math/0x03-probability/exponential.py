#!/usr/bin/env python3
"""class Exponential"""


class Exponential:
    """Represents an exponential distribution"""
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Initiates poisson"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

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
