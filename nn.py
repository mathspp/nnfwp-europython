import numpy as np


def create_weights_matrix(nrows, ncols):
    """Generate normally distributed random numbers."""
    return np.random.default_rng().normal(
        loc=0,
        scale=1/(nrows*ncols),
        size=(nrows, ncols),
    )


def create_bias_vector(length):
    return create_weights_matrix(length, 1)


class Layer:
    """Class representing the connections between two layers of neurons."""

    def __init__(self, inps, outs):
        self._W = create_weights_matrix(outs, inps)
        self._b = create_bias_vector(outs)

    def forward_pass(self, x):
        return np.dot(self._W, x) + self._b
