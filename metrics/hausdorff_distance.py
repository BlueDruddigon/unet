import numpy as np
from medpy import metric

__all__ = ['hd95']


def hd95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true[y_true > 0] = 1
    y_pred[y_pred > 0] = 1
    if y_true.sum() > 0 and y_pred.sum() > 0:
        return metric.binary.hd95()
    return 0
