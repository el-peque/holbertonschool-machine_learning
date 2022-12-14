#!/usr/bin/env python3
"""class Normal"""


class Normal:
    """Represents a normal distribution"""
    pi = 3.1415926536
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
        mean, stddev = self.mean, self.stddev
        return 0.5 * (1 + self.erf((x - mean) / (stddev * (2 ** 0.5))))

    @staticmethod
    def erf(x):
        """Calculates the error function of x"""
