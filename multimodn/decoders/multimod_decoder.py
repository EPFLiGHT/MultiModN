from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn

class MultiModDecoder(nn.Module, ABC):
    """Abstract decoder for MultiModN"""

    def __init__(self, state_size: int):
        super(MultiModDecoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        pass
