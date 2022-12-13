#!/usr/bin/env python3
"""class Poisson"""


class Poisson:
    """Represents a poisson distribution"""
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Initiates poisson"""
        if data is None:
            self.lambtha = float(lambtha)
        elif lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of 'successes'"""
        k = int(k)
        λ = self.lambtha
        return ((self.e ** (-λ)) * (λ ** k) / Poisson.factorial(k))

    @staticmethod
    def factorial(n):
        if n == 1:
            return 1
        else:
            return (n * Poisson.factorial(n - 1))
