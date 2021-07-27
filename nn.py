from abc import abstractmethod
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


class ActivationFunction:
    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def df(self, x):
        pass


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha):
        self._alpha = alpha

    def f(self, x):
        return np.maximum(x, self._alpha*x)

    def df(self, x):
        return np.maximum(self._alpha, x > 0)


def mean_squared_error(outs, targets):
    return np.mean(np.power(outs - targets, 2))


def dmean_square_error(outs, targets):
    return 2*(outs - targets)/outs.size


class Layer:
    """Class representing the connections between two layers of neurons."""

    def __init__(self, inps, outs, act_func):
        self._W = create_weights_matrix(outs, inps)
        self._b = create_bias_vector(outs)
        self._f = act_func

    def forward_pass(self, x):
        return self._f(
            np.dot(self._W, x) + self._b
        )


class Network:
    """Class representing a sequence of compatible layers."""

    def __init__(self, layers):
        self._layers = layers

    def forward_pass(self, x):
        out = x
        for layer in self._layers:
            out = layer.forward_pass(out)
        return out


if __name__ == "__main__":
    layers = [
        Layer(3, 7, leaky_relu),
        Layer(7, 6, leaky_relu),
        Layer(6, 2, leaky_relu),
    ]

    net = Network(layers)
    print(
        net.forward_pass(np.array([1, 2, 3]).reshape((3, 1)))
    )
