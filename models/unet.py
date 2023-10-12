import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self) -> None:
        super(DoubleConv, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
