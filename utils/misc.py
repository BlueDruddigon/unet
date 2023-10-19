import argparse
import random

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_distributed_mode(args: argparse.Namespace) -> None:
    dist.init_process_group(backend='nccl')
    rank = get_rank()
    args.device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(args.device_id)
    torch.cuda.empty_cache()
    print(f'| dist init rank: {rank} |')
    dist.barrier()
    setup_for_distributed(rank == 0)


def is_dist_available_and_initialized() -> bool:
    return dist.is_initialized() if dist.is_available() else False


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master: bool) -> None:
    """This function disables printing when not in the primary process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs) -> None:
        force = kwargs.get('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def cleanup_dist() -> None:
    dist.destroy_process_group()


class AverageMeter:
    def __init__(self) -> None:
        self.val = None
        self.sum = None
        self.count = None
        self.avg = None
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
    
    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
