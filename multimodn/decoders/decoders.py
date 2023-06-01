import torch

from multimodn.decoders.multimod_decoder import MultiModDecoder
from torch import Tensor, sigmoid
from typing import Callable, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F

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
    
class MLPDecoder(MultiModDecoder):
    """Multi-layer perceptron decoder"""
    def __init__(
            self,
            state_size: int,            
            hidden_layers: Tuple[int],
            n_classes: int = 2,
            output_activation: Callable = sigmoid,
            hidden_activation: Callable = F.relu,
            device: Optional[torch.device] = None,
    ):
        super().__init__(state_size)
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.n_classes = n_classes
        dim_layers = [self.state_size] + list(hidden_layers) + [n_classes, ]
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dim_layers, dim_layers[1:])):  
            self.layers.append(nn.Linear(in_dim, out_dim, device=device))

    def forward(self, x: Tensor) -> Tensor:       
        for layer in self.layers[0:-1]:
            x = self.hidden_activation((layer(x)))  
        output = self.output_activation(self.layers[-1](x))
        return output    


class LogisticDecoder(ClassDecoder):
    """Logistic decoder for MultiModN"""

    def __init__(self, state_size: int, device: Optional[torch.device] = None):
        super().__init__(state_size, 2, sigmoid, device)
