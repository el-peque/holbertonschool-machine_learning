#!/usr/bin/env python3
"""class Poisson"""


class Poisson:
    """Represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initiates poisson"""
        if data is None:
            self.lambtha = lambtha
        elif lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(set(data)) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
