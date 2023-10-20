import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolynormialLRWarmup(_LRScheduler):
    """Polynomial learning rate scheduler with warmup.
    Based on `https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/lr_scheduler.py`
    """
    def __init__(
      self,
      optimizer: Optimizer,
      warmup_iters: int,
      total_iters: int = 5,
      power: float = 1.0,
      last_epoch: int = -1,
      verbose: bool = False
    ) -> None:
        super().__init__(optimizer, last_epoch, verbose)
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters
    
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn(
              "To get the last learning rate computed by the scheduler, "
              "please use `get_last_lr()`.", UserWarning
            )
        
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch <= self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        l = self.last_epoch
        w = self.warmup_iters
        t = self.total_iters
        decay_factor = ((1.0 - (l-w) / (t-w)) / (1.0 - (l-1-w) / (t-w))) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:
            return [(
              base_lr * (
                1.0 - (min(self.total_iters, self.last_epoch) - self.warmup_iters) /
                (self.total_iters - self.warmup_iters)
              ) ** self.power
            ) for base_lr in self.base_lrs]
