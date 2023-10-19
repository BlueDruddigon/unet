import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from engine import evaluate, train_one_epoch, validation_epoch
from models import UNet
from utils import cleanup_dist, init_distributed_mode, save_on_master, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data-root', required=True, type=str, default='', help='Path to the data root directory.')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes in the dataset.')
    parser.add_argument('--rgb', action='store_true', help='Whether using RGB mode or not')
    
    # Distributed Training
    parser.add_argument('--distributed', action='store_false', help='Whether using distributed data parallel')
    parser.add_argument('--amp', action='store_false', help='Whether using Mixed Average Precision')
    parser.add_argument('--seed', type=int, default=1245, help='Seed number')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--device-id', type=int, default=0, help='Device id if not using DDP')
    
    # Model and Training Parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-8, help='Weight decay rule for Optimizer')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum for SGD')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='Number of training epochs')
    
    # Utilities
    parser.add_argument('--valid-freq', type=int, default=10, help='Frequency of validation')
    parser.add_argument('--save-freq', type=int, default=5, help='Frequency of saving checkpoint')
    parser.add_argument('--save-dir', default='weights', type=str, help='Path to save checkpoint')
    parser.add_argument('--log-dir', default='runs', type=str, help='Path to log dir')
    parser.add_argument('--resume', default='', type=str, help='Checkpointing to resume from')
    
    return parser.parse_args()


def main(args: argparse.Namespace):  # sourcery skip: extract-method
    if args.distributed:
        init_distributed_mode(args)
    
    # seeding
    seed_everything(args.seed)
    
    args.n_channels = 3 if args.rgb else 1
    
    # Model and Loss fn
    model = UNet(in_channels=args.n_channels, out_channels=args.classes)
    criterion = nn.CrossEntropyLoss() if args.classes > 1 else nn.BCEWithLogitsLoss()
    model = model.to(args.device_id)
    criterion = criterion.to(args.device_id)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.device_id])
        model.register_comm_hook(None, fp16_compress_hook)
    
    # Optimizers
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.amp)
    
    # Dataset and Loader
    train_loader, valid_loader, test_loader = build_dataset(args)
    
    # Create weights and logs dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Tensorboard logger
    tensorboard_dir = os.path.join(args.log_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    
    # Resume from checkpoint
    args.start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device_id)
        if args.distributed:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f'Resume from epoch {args.start_epoch}')
    else:
        print('Training from scratch.')
    
    # Training process
    for epoch in range(args.start_epoch, args.epochs + 1):
        loss_value = train_one_epoch(
          model, criterion, scaler=scaler, optimizer=optimizer, loader=train_loader, epoch=epoch, args=args
        )
        writer.add_scalar('train/loss', loss_value, epoch)
        
        # Validation Process
        if epoch % args.valid_freq == 0 and epoch > 0:
            valid_loss = validation_epoch(model, criterion, loader=valid_loader, epoch=epoch, args=args)
            writer.add_scalar('valid/loss', valid_loss, epoch)
            # Update scheduler's state
            scheduler.step(valid_loss)
        
        # Save Checkpoint
        if epoch % args.save_freq == 0 and epoch > 0:
            if args.distributed:
                save_on_master({
                  'state_dict': model.module.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'epoch': epoch
                }, f'{args.save_dir}/{epoch}.pth')
            else:
                torch.save({
                  'state_dict': model.module.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'epoch': epoch
                }, f'{args.save_dir}/{epoch}.pth')
    
    # Test phase
    print('End of training. Start evaluation on test set.')
    score = evaluate(model, loader=test_loader, args=args)
    print(f'Test score: {score:.4f}')
    
    cleanup_dist()


if __name__ == '__main__':
    args = parse_args()
    main(args)
