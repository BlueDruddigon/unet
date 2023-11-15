import glob
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.misc import seed_everything


class SynapseDataset(Dataset):
    phases = ['train', 'valid', 'test']
    
    def __init__(self, root: str, phase: str = 'train', transform: Optional[Dict[str, Callable]] = None) -> None:
        super(SynapseDataset, self).__init__()
        
        assert phase in self.phases
        
        self.root = root
        self.transform = transform[phase] if transform is not None else None
        self.ids = glob.glob(os.path.join(self.root, phase, '*.npz'))
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        assert index <= len(self)
        data_path = self.ids[index]
        
        data = np.load(data_path)
        image, label = data['image'], data['label']
        data.close()
        
        # convert to Tensor
        image = torch.from_numpy(image).to(dtype=torch.float32).unsqueeze(0)  # (1, IMG_SIZE, IMG_SIZE)
        label = torch.from_numpy(label).to(dtype=torch.float32).unsqueeze(0)  # (1, IMG_SIZE, IMG_SIZE)
        
        if self.transform is not None:
            seed = np.random.randint(0, 1362947)
            seed_everything(seed)
            image = self.transform(image)
            seed_everything(seed)
            label = self.transform(label)
        
        return image, label.squeeze(0)
