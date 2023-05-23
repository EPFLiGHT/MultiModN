from multimodn.encoders import MLPEncoder
from typing import Callable
from torch import sigmoid

class SLPEncoder(MLPEncoder):
    """Single Layer Perceptron encoder"""

    def __init__(
        self,
        state_size: int,
        n_features: int,
        activation: Callable = sigmoid,
    ):
        super().__init__(state_size, n_features, (), activation)

class LinearEncoder(SLPEncoder):
    """Linear encoder"""

    def __init__(
        self,
        state_size: int,
        n_features: int,
    ):
        super().__init__(state_size, n_features, lambda x: x)

class LogisticEncoder(SLPEncoder):
    """Logistic encoder"""

    def __init__(
        self,
        state_size: int,
        n_features: int,
    ):
        super().__init__(state_size, n_features, sigmoid)