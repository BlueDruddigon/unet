import argparse
from typing import Tuple

import torch.nn as nn
import yaml

from .focal_unet import FocalUNet
from .swin_unet import SwinUNet
from .unet import UNet

__all__ = ['SwinUNet', 'UNet', 'build_model']


def build_model(args: argparse.Namespace) -> Tuple[nn.Module, argparse.Namespace]:
    config_path = args.config
    args.num_channels = 3 if args.rgb else 1
    # Load config from dataset
    if config_path:
        with open(config_path) as f:
            hyper_params = yaml.safe_load(f)
        args.n_classes = hyper_params['DATASET']['N_CLASSES'] or args.num_classes
        args.n_channels = hyper_params['DATASET']['N_CHANNELS'] or args.num_channels
        args.image_size = hyper_params['DATASET']['IMAGE_SIZE']
    if args.model_name == 'unet':
        model = UNet(in_channels=args.num_channels, out_channels=args.num_classes)
        if config_path:
            model = UNet(in_channels=args.n_channels, out_channels=args.n_classes)
    elif args.model_name == 'swin_unet':
        model = SwinUNet()
        if config_path:
            with open(config_path) as f:
                hyper_params = yaml.safe_load(f)
            
            drop_path_rate = hyper_params['MODEL']['DROP_PATH_RATE']
            patch_size = hyper_params['MODEL']['SWIN']['PATCH_SIZE']
            embed_dim = hyper_params['MODEL']['SWIN']['EMBED_DIM']
            depths = hyper_params['MODEL']['SWIN']['DEPTHS']
            num_heads = hyper_params['MODEL']['SWIN']['NUM_HEADS']
            window_size = hyper_params['MODEL']['SWIN']['WINDOW_SIZE']
            final_upsample = hyper_params['MODEL']['SWIN']['FINAL_UPSAMPLE']
            
            model = SwinUNet(
              img_size=args.image_size,
              patch_size=patch_size,
              in_channels=args.n_channels,
              num_classes=args.n_classes,
              embed_dim=embed_dim,
              depths=depths,
              num_heads=num_heads,
              window_size=window_size,
              drop_path_rate=drop_path_rate,
              final_upsample=final_upsample
            )
    else:
        raise NotImplementedError
    return model, args
