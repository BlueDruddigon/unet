from typing import Tuple, List

import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    """2-Layer Convolution followed by BatchNorm and Activation"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        layers = nn.ModuleList([
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        ])
        
        super(DoubleConv, self).__init__(*layers)


class ContractComponent(nn.Module):
    """Single Component of Contracting Path, which is a DoubleConv and a 2x2 MaxPool"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ContractComponent, self).__init__()
        
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Include a conv feature map in the result for ExpandPath"""
        x = self.conv(x)
        return self.pool(x), x


class ContractPath(nn.Module):
    """Contracting Path with last DoubleConv without down sampling"""
    def __init__(self, in_channels: int = 1, out_channels: int = 1024, num_levels: int = 4) -> None:
        super(ContractPath, self).__init__()
        
        self.input_layers = [in_channels, 64, 128, 256]
        self.output_layers = [64, 128, 256, 512]
        
        assert len(self.input_layers) == num_levels
        assert len(self.output_layers) == num_levels
        
        self.blocks = nn.ModuleList([
          ContractComponent(self.input_layers[i], self.output_layers[i]) for i in range(num_levels)
        ])
        
        self.last = DoubleConv(512, out_channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Resulting output feature map with a list of previous feature maps for ExpandPath"""
        feats = []
        for blk in self.blocks:
            x, conv_feat = blk(x)
            feats.append(conv_feat)
        
        return self.last(x), feats


class CenterCrop:
    """Center Crop Feature Map for Concatenation"""
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x1.size()
        pad_x = (x2.size()[3] - x1.size()[3]) // 2
        pad_y = (x2.size()[2] - x1.size()[2]) // 2
        return x2[:, :, pad_y:pad_y + h, pad_x:pad_x + w].clone()


class ExpandComponent(nn.Module):
    """Up-Convolution followed by CenterCrop and DoubleConv"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super(ExpandComponent, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2)
        self.crop = CenterCrop()
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        y = self.crop(x, y)
        x = torch.cat([y, x], dim=1)
        x = self.conv(x)
        return x


class ExpandPath(nn.Module):
    """Complete ExpandPath with the final output feature map"""
    def __init__(self, in_channels: int = 1024, out_channels: int = 2, num_levels: int = 4) -> None:
        super(ExpandPath, self).__init__()
        
        self.input_layers = [in_channels, 512, 256, 128]
        self.output_layers = [512, 256, 128, 64]
        
        self.blocks = nn.ModuleList([
          ExpandComponent(
            self.input_layers[i],
            self.output_layers[i],
            # add a specific output channel of last block
            out_channels=out_channels if i == num_levels - 1 else self.output_layers[i]
          ) for i in range(num_levels)
        ])
    
    def forward(self, x: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            x = blk(x, y[i])
        
        return x


class UNet(nn.Module):
    """U-Net architecture"""
    def __init__(self, in_channels: int = 1, hidden_channels: int = 1024, out_channels: int = 2) -> None:
        super(UNet, self).__init__()
        
        self.contract = ContractPath(in_channels, hidden_channels)
        self.expand = ExpandPath(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_feats = self.contract(x)
        conv_feats = conv_feats[::-1]  # Reverse the list of previous conv feats
        x = self.expand(x, conv_feats)
        return x
