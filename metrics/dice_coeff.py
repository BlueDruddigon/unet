import numpy as np
from medpy import metric

__all__ = ['dice_coef']


def dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true[y_true > 0] = 1
    y_pred[y_pred > 0] = 1
    if y_true.sum() > 0 and y_pred.sum() > 0:
        return metric.binary.dc(y_true, y_pred)
    elif y_true.sum() == 0 and y_pred.sum() > 0:
        return 1
    return 0
