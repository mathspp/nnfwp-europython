import numpy as np


class Layer:
    """Class representing the connections between two layers of neurons."""

    def __init__(self, W, b):
        self._W = W
        self._b = b

    def forward_pass(self, x):
        return np.dot(self._W, x) + self._b
