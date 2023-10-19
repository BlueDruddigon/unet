import argparse

import torch
from torch.utils.data import (
    BatchSampler, DataLoader, DistributedSampler, random_split, RandomSampler, SequentialSampler,
)

from .synapse import RandomGenerator, SynapseDataset
from .transformations import *

__all__ = ['SynapseDataset', 'RandomGenerator']


def build_dataset(args: argparse.Namespace):
    # dataset and splits
    dataset = SynapseDataset(args.root, is_train=True)
    train_set, valid_set, test_set = random_split(
      dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(0)
    )
    
    # samplers
    if args.distributed:
        train_sampler = DistributedSampler(train_set)
        valid_sampler = DistributedSampler(valid_set, shuffle=False)
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = RandomSampler(train_set)
        valid_sampler = SequentialSampler(valid_set)
        test_sampler = SequentialSampler(test_set)
    
    train_batch_sampler = BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=True)
    
    # loader
    train_loader = DataLoader(
      train_set, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
      valid_set,
      sampler=valid_sampler,
      drop_last=True,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      pin_memory=True
    )
    test_loader = DataLoader(
      test_set,
      sampler=test_sampler,
      drop_last=True,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader
