import math
import random

import numpy as np


class Neuron:
    def __init__(self, mean: np.ndarray, std: float):
        self.weight = random.random()
        self.mean = mean
        self.std = std
        self.y = 0

    def basis_function(self, x: np.ndarray) -> None:
        g = ((x - self.mean) ** 2).sum() / (2 * self.std ** 2)
        self.y = math.exp(-g)

    def update(self, eta: float, error: float, x: np.ndarray) -> None:
        delta = eta * error * self.y
        w, m, s = self.weight, self.mean.copy(), self.std
        self.weight += delta
        self.mean += delta * w * s ** -2 * (x - m)
        self.std += delta * w * s ** -3 * ((x - m) ** 2).sum()
