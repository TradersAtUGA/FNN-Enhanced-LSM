import torch
import torch.nn as nn
from typing import List

# Feedforward Neural Network (FNN), also known as a Multilayer Perceptron (MLP)
class LSMContinuationNN(nn.Module):
    def __init__(self, dimensions: int, nn_layers: List[int]):
        super().__init__()

        # assume layers = 3
        first_layer = nn_layers[0]
        second_layer = nn_layers[1]
        third_layer = nn_layers[2]

        self.model = nn.Sequential(
            nn.Linear(dimensions, first_layer),
            nn.LeakyReLU(0.01),
            nn.Linear(first_layer, second_layer),
            nn.LeakyReLU(0.01),
            nn.Linear(second_layer, third_layer),
            nn.LeakyReLU(0.01),
            nn.Linear(third_layer, 1)
        )
    
    def forward(self, x):
        return self.model(x)