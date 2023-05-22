from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn


class MoMoEncoder(nn.Module, ABC):
    """Abstract encoder for MoMoNet"""

    def __init__(self, state_size: int):
        super(MoMoEncoder, self).__init__()
        self.state_size = state_size

    @abstractmethod
    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        pass
