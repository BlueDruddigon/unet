from typing import Callable

import torch


class EarlyStopping:
    """Early Stopping simple implementation in PyTorch"""
    
    mode_dict = {'min': torch.lt, 'max': torch.gt}
    
    def __init__(self, min_delta: float = 1e-4, patience: int = 5, mode: str = 'min') -> None:
        """
        :param min_delta: minimum change in the monitored quantity to qualify as an improvement.
        :param patience: number of checks with no improvement.
        :param mode: one of `"min", "max"`. Defaults to `"min"`
            In `min` mode, training will stop when the quantity monitored has stopped decreasing.
            And in `max` mode, it will stop when the quantity monitored has stopped increasing.
        """
        super(EarlyStopping, self).__init__()
        
        self.patience = patience
        self.min_delta = torch.tensor(min_delta)
        self.counter = 0
        
        self.mode = mode
        if self.mode not in self.mode_dict:
            raise NotImplementedError
        
        self.best_score = torch.tensor(torch.inf) if self.mode == 'min' else torch.tensor(-torch.inf)
    
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    
    def step(self, metric: torch.Tensor) -> bool:
        """
        :param metric: metric quantity to be monitored.
        :return: a boolean flag to manage if the training process should be stopped.
        """
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric)
        if self.monitor_op(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        elif self.monitor_op(self.min_delta, metric - self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
