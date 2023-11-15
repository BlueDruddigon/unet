import argparse

from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .synapse import SynapseDataset
from .transformations import get_default_transformations

__all__ = ['SynapseDataset', 'get_default_transformations', 'build_dataset']


def build_dataset(args: argparse.Namespace):
    # transformer, dataset and splits
    transformer = get_default_transformations(args.image_size)
    
    train_set = SynapseDataset(args.data_root, phase='train', transform=transformer)
    valid_set = SynapseDataset(args.data_root, phase='valid', transform=transformer)
    test_set = SynapseDataset(args.data_root, phase='test', transform=transformer)
    
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
