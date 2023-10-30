import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import compute_metrics
from utils import AverageMeter


def train_one_epoch(
  model: nn.Module,
  criterion: nn.Module,
  scaler: GradScaler,
  loader: DataLoader,
  optimizer: optim.Optimizer,
  epoch: int,
  args: argparse.Namespace,
  max_norm: float = 1.0
) -> float:
    model.train()
    criterion.train()
    
    pbar = tqdm(enumerate(loader), total=len(loader))
    
    # metric logger
    running_loss = AverageMeter()
    batch_timer = AverageMeter()
    
    end = time.time()
    for idx, (image, label) in pbar:
        image = image.to(device=args.device_id, dtype=torch.float32)
        label = label.to(device=args.device_id, dtype=torch.long)
        
        with autocast(enabled=args.amp):
            pred = model(image)
            loss = criterion(pred, label)
        
        # Back-prop
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # update metrics
        running_loss.update(loss.item(), n=1)
        batch_timer.update(time.time() - end)
        end = time.time()
        
        # update pbar's status
        s = f'Epoch [{epoch}][{idx + 1}/{len(loader)}] ' \
            f'Time/b: {batch_timer.val:.2f} ({batch_timer.avg:.2f})s ' \
            f'Loss: {running_loss.val:.4f} ({running_loss.avg:.4f})'
        pbar.set_description(s)
    
    return running_loss.avg


@torch.no_grad()
def validation_epoch(
  model: nn.Module, criterion: nn.Module, loader: DataLoader, epoch: int, args: argparse.Namespace
) -> float:
    model.eval()
    criterion.eval()
    
    # status bar
    pbar = tqdm(enumerate(loader), total=len(loader))
    
    # metric logger
    validation_loss = AverageMeter()
    batch_timer = AverageMeter()
    
    end = time.time()
    for idx, (image, label) in pbar:
        image = image.to(device=args.device_id, dtype=torch.float32)
        label = label.to(device=args.device_id, dtype=torch.long)
        
        with autocast(enabled=args.amp):
            pred = model(image)
            loss = criterion(pred, label)
        
        # Update Loss value
        validation_loss.update(loss.item(), 1)
        batch_timer.update(time.time() - end)
        end = time.time()
        
        # Update Status Bar
        s = f'Validation [{epoch}][{idx + 1}/{len(loader)}] ' \
            f'Time/b: {batch_timer.val:.2f} ({batch_timer.avg:.2f})s ' \
            f'Loss: {validation_loss.val:.4f} ({validation_loss.avg:.4f})'
        pbar.set_description(s)
    
    return validation_loss.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, args: argparse.Namespace) -> float:
    model.eval()
    
    pbar = tqdm(enumerate(loader), total=len(loader))
    
    # metric logger
    running_time = AverageMeter()
    metrics = 0.
    
    with autocast(enabled=args.amp):
        for idx, (image, label) in pbar:
            start = time.time()
            image = image.to(args.device_id, dtype=torch.float32)
            label = label.to(args.device_id, dtype=torch.long)
            
            # Feed forward to the network and take the softmax
            output = model(image)
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            # turn output and label to numpy format
            y_true = label.squeeze(0).detach().cpu().numpy()
            y_pred = output.squeeze(0).detach().cpu().numpy()
            
            metric_list = [compute_metrics(y_true == i, y_pred == i) for i in range(args.num_classes)]
            
            end = time.time()
            # update metrics
            running_time.update(end - start)
            metrics += np.array(metric_list)
            
            # Update status bar
            s = f'Evaluation [{idx+1}/{len(loader)}] ' \
                f'Time/b: {running_time.val:.2f} ({running_time.avg:.2f})s ' \
                f'Mean Dice: {np.mean(metrics, axis=0)[0]} Mean HD: {np.mean(metrics, axis=0)[1]}'
            pbar.set_description(s)
    
    return np.mean(metrics, axis=0)
