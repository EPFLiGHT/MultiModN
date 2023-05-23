import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from multimodn.encoders import MultiModEncoder

class LSTMEncoder(MultiModEncoder):
    """LSTM encoder"""

    def __init__(
        self,
        state_size: int,
        n_features: int,
        hidden_layers: Tuple[int],
        activation: Callable = F.relu,
    ):
        super().__init__(state_size)

        self.activation = activation

        dim_layers = [n_features] + list(hidden_layers) + [self.state_size,]

        self.layers = nn.ModuleList()
        for i, (inDim, outDim) in enumerate(zip(dim_layers, dim_layers[1:])):
            # The state is concatenated to the input of the last layer
            if i == len(dim_layers)-2:
                self.layers.append(nn.LSTM(inDim + self.state_size, outDim, batch_first=True))
            else:
                self.layers.append(nn.LSTM(inDim, outDim, batch_first=True))

    def forward(self, state: Tensor, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            out, tups = layer(x)
            x = self.activation(out)

        output, tups = self.layers[-1](torch.cat([x, state], dim=1))

        return output

class LSTMFeatureEncoder(LSTMEncoder):
    """Feature encoder"""

    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        activation: Callable = F.relu,
    ):
        super().__init__(state_size, 1, (hidden_size,), activation)
    
    def forward(self, state: Tensor, x: float) -> Tensor:
        return super().forward(state, Tensor(x))