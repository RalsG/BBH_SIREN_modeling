import torch
import torch.nn as nn
import math


class SirenLayer(nn.Module):
    """A single SIREN layer with sine activation."""
    def __init__(self, in_features, out_features, bias=True, 
                 is_first=False, w0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        
        # Create a standard linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Perform the special SIREN initialization
        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer initialization
                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layer initialization
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        # Apply linear transformation and sine activation
        return torch.sin(self.w0 * self.linear(x))
    

class Siren(nn.Module):
    """The full SIREN model."""
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, w0=30.0):
        super().__init__()
        
        layers = []
        # First layer
        layers.append(SirenLayer(in_features, hidden_features, is_first=True, w0=w0))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features, is_first=False, w0=w0))
            
        # Final output layer
        final_linear = nn.Linear(hidden_features, out_features)
        # Init final layer with smaller bound per SIREN paper for stability
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / w0
            final_linear.weight.uniform_(-bound, bound)
            if final_linear.bias is not None:
                final_linear.bias.zero()
        layers.append(final_linear)
        

        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)