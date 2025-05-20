import torch
import torch.nn as nn

# Feedforward Neural Network (FNN), also known as a Multilayer Perceptron (MLP)
class LSMContinuationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.model(x)