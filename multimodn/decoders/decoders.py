import torch

from multimodn.decoders.multimod_decoder import MultiModDecoder
from torch import Tensor, sigmoid
from typing import Callable, Optional
import torch.nn as nn


class ClassDecoder(MultiModDecoder):
    """Classifier for MultiModN"""

    def __init__(self, state_size: int, n_classes: int, activation: Callable,
                 device: Optional[torch.device] = None):
        super().__init__(state_size)
        self.n_classes = n_classes
        self.fc = nn.Linear(state_size, n_classes, device=device)
        self.activation = activation

    def forward(self, state: Tensor) -> Tensor:
        return self.activation(self.fc(state))


class LogisticDecoder(ClassDecoder):
    """Logistic decoder for MultiModN"""

    def __init__(self, state_size: int, device: Optional[torch.device] = None):
        super().__init__(state_size, 2, sigmoid, device)
