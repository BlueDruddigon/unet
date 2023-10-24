import glob
import os
import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .transformations import random_rot_flip, random_rotate


class RandomGenerator:
    def __init__(self, output_size: Union[List[int], Tuple[int, int]]) -> None:
        self.output_size = output_size
    
    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[torch.Tensor, torch.LongTensor]:
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        height, width = image.shape
        if height != self.output_size[0] or width != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / height, self.output_size[1] / width), order=3)
            label = zoom(label, (self.output_size[0] / height, self.output_size[1] / width), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return image, label.long()


class SynapseDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, is_train: bool = True) -> None:
        super(SynapseDataset, self).__init__()
        
        self.root = root
        self.phase = 'train' if is_train else 'test'
        self.transform = transform or RandomGenerator((224, 224))
        
        self.ids = glob.glob(os.path.join(self.root, self.phase, '*.npz'))
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        data_path = self.ids[index]
        data = np.load(data_path)
        image, label = data['image'], data['label']
        data.close()
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
