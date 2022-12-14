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
            self.p = (sum(data) / len(data) ** 2) * 2
            self.n = round(len(data) / 2)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        res = (1 / ((2 * self.pi * (self.stddev ** 2)) ** 0.5))
        res *= self.e ** ((-1 / 2) * ((((x - self.mean) / self.stddev)) ** 2))
        return res

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        return totient(x) * ((x - self.mean) / self.stddev)

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of 'successes'"""
        k = int(k)
        if k < 0:
            return 0

        λ = self.lambtha
        return ((self.e ** (-λ)) * (λ ** k) / Poisson.factorial(k))

    @staticmethod
    def factorial(n):
        """Calculates the factorial of a number"""
        if n <= 1:
            return 1
        else:
            return (n * Poisson.factorial(n - 1))
