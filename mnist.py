# MNIST
# “Hello, world!” of machine learning.

import csv
import numpy as np
from nn import Layer, Network, LeakyReLU, MSE


if __name__ == "__main__":
    layers = [
        Layer(784, 16, LeakyReLU(0.1)),
        Layer(16, 16, LeakyReLU(0.1)),
        Layer(16, 10, LeakyReLU(0.1)),
    ]
    net = Network(layers, 0.001, MSE())

    # Test the network.

    # Train the network.

    # Test the network again.
