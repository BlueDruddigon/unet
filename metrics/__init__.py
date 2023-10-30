from typing import Tuple

import numpy as np

from .dice_coeff import dice_coef
from .hausdorff_distance import hd95

__all__ = ['compute_metrics']


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    return dice_coef(y_true, y_pred), hd95(y_true, y_pred)
