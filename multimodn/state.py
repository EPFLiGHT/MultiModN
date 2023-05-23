from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import nn, Tensor
from itertools import cycle

class InitState(nn.Module, ABC):
    """Trainable initial state"""

    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size
        
    @abstractmethod
    def forward(self, batch_size) -> Tensor:
        pass

class TrainableInitState(InitState):
    """Trainable initial state"""

    def __init__(self, state_size: int, device: Optional[torch.device] = None):
        super().__init__(state_size)
        self.device = device
        self.state_value = nn.Parameter(
            torch.randn((1, state_size), requires_grad=True, device=self.device)
        )
    
    def forward(self, batch_size) -> Tensor:
        init_tensor = torch.tile(self.state_value, [batch_size, 1])

        return init_tensor

class StaticInitState(InitState):
    """Initial state given from list"""

    def __init__(self, states: List[Tensor]):
        state_size = states[0].size(0)
        super().__init__(state_size)

        self.state_iterator = cycle(states)
    
    def forward(self, batch_size) -> Tensor:
        states = [torch.reshape(next(self.state_iterator), (1, -1)) for _ in range(batch_size)]

        state = torch.cat(states, dim=0)

        return state.detach()