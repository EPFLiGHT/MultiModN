from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn

class MoMoDecoder(nn.Module, ABC):
    """Abstract decoder for MoMoNet"""

    def __init__(self, state_size: int):
        super(MoMoDecoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        pass
