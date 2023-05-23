from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn


class MultiModEncoder(nn.Module, ABC):
    """Abstract encoder for MultiModN"""

    def __init__(self, state_size: int):
        super(MultiModEncoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        pass
