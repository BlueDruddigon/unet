import torch
import torch.nn as nn


class AttentionUNet(nn.Module):
    def __init__(self) -> None:
        super(AttentionUNet, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
