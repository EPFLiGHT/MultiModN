import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
from multimodn.encoders import MultiModEncoder


class MLPEncoder(MultiModEncoder):
    """Multi-layer perceptron encoder"""

    def __init__(
            self,
            state_size: int,
            n_features: int,
            hidden_layers: Tuple[int],
            activation: Callable = F.relu,
            device: Optional[torch.device] = None,
    ):
        super().__init__(state_size)

        self.activation = activation

        dim_layers = [n_features] + list(hidden_layers) + [self.state_size, ]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers) - 2:
                self.layers.append(
                    nn.Linear(in_dim + self.state_size, out_dim, device=device))
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, device=device))

    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        for layer in self.layers[0:-1]:
            x = self.activation(layer(x))

        output = self.layers[-1](torch.cat([x, state], dim=1))

        return output


class MLPFeatureEncoder(MLPEncoder):
    """Feature encoder"""

    def __init__(
            self,
            state_size: int,
            hidden_size: int,
            activation: Callable = F.relu,
            device: Optional[torch.device] = None,
    ):
        super().__init__(state_size, 1, (hidden_size,), activation, device)

    def forward(self, state: Tensor, x: float) -> Tensor:
        return super().forward(state, Tensor(x))
