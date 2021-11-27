import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'linear'

        # Define the network layers in order.
        # Input is 2D.
        # Output is a single value.
        # Single linear layer.
        self.layers = nn.Sequential(
            nn.Linear(3, 1)
        )
    
    def forward(self, batch):
        # Process batch using the layers.
        x = self.layers(batch)
        # Final sigmoid activation to obtain a probability.
        return torch.sigmoid(x)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'mlp'

        # Define the network layers in order.
        # Input is 2D.
        # Output is a single value.
        # Multiple linear layers each followed by a ReLU non-linearity (apart from the last).
        lin = 2
        hln = 16
        lout = 1
        self.layers = nn.Sequential(
            nn.Linear(lin, hln),
            nn.ReLU(),
            nn.Linear(hln, lout)
        )
    
    def forward(self, batch):
        # Process batch using the layers.
        x = self.layers(batch)
        # Final sigmoid activation to obtain a probability.
        return torch.sigmoid(x)
