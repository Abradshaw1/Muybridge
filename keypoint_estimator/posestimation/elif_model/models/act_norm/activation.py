import torch.nn as nn
import torch

class SiLU(nn.Module):
    """Sigmoid Weighted Liner Unit."""

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs) -> torch.Tensor:
        if self.inplace:
            return inputs.mul_(torch.sigmoid(inputs))
        else:
            return inputs * torch.sigmoid(inputs)