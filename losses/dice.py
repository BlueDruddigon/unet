from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
      self,
      sigmoid: bool = False,
      softmax: bool = True,
      reduction: str = 'mean',
      eps: float = 1e-7,
    ) -> None:
        super(DiceLoss, self).__init__()
        
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            inputs = F.sigmoid(inputs)
        
        num_classes = inputs.shape[1]
        if self.softmax:
            assert num_classes != 1, 'single channel prediction, `softmax=True` is ignored'
            inputs = F.softmax(inputs, dim=1)
        
        if num_classes != 1:
            if targets.shape[1] == 1 or targets.ndim == 4:
                targets = targets.squeeze(1)
            targets = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2)
        
        assert inputs.shape == targets.shape, \
            f'Different shape between inputs ({inputs.shape}) and targets ({targets.shape})'
        
        reduce_axis: List[int] = torch.arange(2, len(inputs.shape)).tolist()
        
        intersection = torch.sum(targets * inputs, dim=reduce_axis)
        denominator = torch.sum(inputs, dim=reduce_axis) + torch.sum(targets, dim=reduce_axis)
        loss: torch.Tensor = 1. - (2.*intersection + self.eps) / (denominator + self.eps)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(f'Unsupported reduction method {self.reduction}. Available: ["sum", "mean"].')
        
        return loss


class DiceCELoss(nn.Module):
    def __init__(
      self,
      num_classes: int,
      gamma: float = 0.4,
      sigmoid: bool = False,
      softmax: bool = True,
      reduction: str = 'mean',
      eps: float = 1e-7
    ) -> None:
        super(DiceCELoss, self).__init__()
        
        self.num_classes = num_classes
        self.reduction = reduction
        self.eps = eps
        self.gamma = gamma
        
        self.dice = DiceLoss(sigmoid=sigmoid, softmax=softmax, reduction=self.reduction, eps=self.eps)
        self.cross_entropy = nn.CrossEntropyLoss(
          reduction=self.reduction, label_smoothing=self.eps
        ) if self.num_classes > 1 else nn.BCEWithLogitsLoss(reduction=self.reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch) -> torch.Tensor:
        ce_loss = self.cross_entropy(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        total_loss: torch.Tensor = self.gamma * dice_loss + (1 - self.gamma) * ce_loss
        return total_loss
