from typing import Tuple

import numpy as np
from scipy import ndimage


def random_rot_flip(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = np.random.randint(0, 4)  # direction to rotate
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)  # axis for flipping
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
