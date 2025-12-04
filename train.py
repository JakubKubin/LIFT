"""
Training script for LIFT model.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import warnings

# Suppress torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Import datasets
from dataset import (
    Vimeo15Dataset,
    X4K1000FPSDataset,
    UCF101Dataset,
    collate_fn
)

# Import Model, Loss and Config
from model import LIFT, LIFTLoss
from configs.default import Config

# Import Logger
from utils.tensorboard_logger import TensorBoardLogger

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def get_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )


def get_lr_scheduler(optimizer, config, steps_per_epoch):
    def lr_lambda(step):
        if step < config.lr_warmup_steps:
            return step / config.lr_warmup_steps
        else:
            total_steps = config.num_epochs * steps_per_epoch
            progress = (step - config.lr_warmup_steps) / (total_steps - config.lr_warmup_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return cosine_decay * (1.0 - config.lr_min / config.learning_rate) + config.lr_min / config.learning_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_psnr(pred, target) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return -10 * torch.log10(mse)


def train_epoch(
    model: LIFT,
    dataloader,
    optimizer,
    scheduler,
    loss_fn: LIFTLoss,
    device,
    epoch,
    config,
    logger: TensorBoardLogger,
    global_step,
    scaler
):
    model.train()
    model.set_epoch(epoch)

    total_losses = {k: 0.0 for k in ['total', 'l1', 'lap', 'flow_smooth', 'occlusion']}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    start_time = time.time()

    for step, batch in enumerate(pbar):
        frames = batch['frames'].to(device, non_blocking=True)
        ref_frames = batch['ref_frames'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)
        timestep = batch['timestep'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast('cuda', enabled=config.mixed_precision):
            outputs = model(frames, ref_frames, timestep[0].item())
            pred = outputs['prediction']

            losses = loss_fn(
                pred, gt,
                flow1=outputs['flows']['flow_7'],
                flow2=outputs['flows']['flow_9'],
                logit_occ1=outputs['occlusions']['logit_occ_7'],
                logit_occ2=outputs['occlusions']['logit_occ_9'],
                warped1=outputs['warped']['warped_7'],
                warped2=outputs['warped']['warped_9']
            )
            loss = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_time = time.time() - start_time
        start_time = time.time()

        logger.log_training_step(
            step=global_step,
            losses=losses,
            lr=scheduler.get_last_lr()[0],
            outputs=outputs,
            gt=gt,
            ref_frames=ref_frames,
            batch_time=batch_time
        )

        for key in total_losses.keys():
            if key in losses:
                total_losses[key] += losses[key].item()

        num_batches += 1
        global_step += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses, global_step


def validate(model: LIFT, dataloader, loss_fn: LIFTLoss, device, epoch, config, logger: TensorBoardLogger):
    torch.cuda.empty_cache()
    model.eval()

    total_losses = {'total': 0.0, 'l1': 0.0}
    total_psnr = 0.0
    num_batches = 0

    # Capture samples for visualization
    sample_data = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
            frames = batch['frames'].to(device, non_blocking=True)
            ref_frames = batch['ref_frames'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            timestep = batch['timestep'].to(device, non_blocking=True)

            with torch.autocast('cuda', enabled=config.mixed_precision):
                outputs = model(frames, ref_frames, timestep[0].item())
                pred = outputs['prediction']
                losses = loss_fn(pred, gt)

            # Save first batch for visualization
            if i == 0:
                sample_data = {
                    'outputs': outputs,
                    'gt': gt,
                    'ref_frames': ref_frames
                }

            psnr = compute_psnr(pred.float(), gt.float())
            total_psnr += psnr.item()

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            num_batches += 1

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_psnr = total_psnr / num_batches

    logger.log_validation(
        epoch=epoch,
        avg_losses=avg_losses,
        avg_psnr=avg_psnr,
        outputs=sample_data.get('outputs'),
        gt=sample_data.get('gt'),
        ref_frames=sample_data.get('ref_frames')
    )

    return avg_losses, avg_psnr


def get_dataset_class(dataset_name):
    if dataset_name.lower() == 'vimeo':
        return Vimeo15Dataset
    elif dataset_name.lower() == 'x4k':
        return X4K1000FPSDataset
    elif dataset_name.lower() == 'ucf101':
        return UCF101Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Train LIFT model')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='vimeo', choices=['vimeo', 'x4k', 'ucf101'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_encoder', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--max_sequences', type=int, default=None)
    args = parser.parse_args()

    config = Config()
    config.data_root = args.data_root
    if args.batch_size: config.batch_size = args.batch_size
    if args.num_epochs: config.num_epochs = args.num_epochs
    if args.num_workers: config.num_workers = args.num_workers
    if args.num_frames: config.num_frames = args.num_frames

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nLoading {args.dataset.upper()} datasets from {config.data_root}...")
    DatasetClass = get_dataset_class(args.dataset)

    try:
        train_dataset = DatasetClass(
            data_root=config.data_root, mode='train', num_frames=config.num_frames,
            crop_size=config.crop_size, augment=True, input_scale=config.input_scale,
            max_sequences=args.max_sequences
        )

        val_dataset = DatasetClass(
            data_root=config.data_root, mode='val', num_frames=config.num_frames,
            crop_size=config.crop_size, augment=False, input_scale=config.input_scale,
            max_sequences=args.max_sequences
        )

        train_dataset.visualize_samples('vis_debug', num_samples=5)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=True, prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True, prefetch_factor=2
    )

    print("\nCreating LIFT model...")
    model = LIFT(config).to(device)

    scaler = torch.GradScaler('cuda', enabled=config.mixed_precision)
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    loss_fn = LIFTLoss(config).to(device)

    # Initialize Logger
    logger = TensorBoardLogger(config.log_dir, config, model)

    start_epoch = 0
    global_step = 0

    # Checkpoint Loading
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        model_state = model.state_dict()
        new_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                     if k in model_state and v.shape == model_state[k].shape}

        model.load_state_dict(new_state, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from epoch {start_epoch}")

    elif args.pretrained_encoder:
        print(f"Loading pretrained encoder: {args.pretrained_encoder}")
        checkpoint = torch.load(args.pretrained_encoder, map_location=device)
        enc_state = {k.replace('encoder.', ''): v for k, v in checkpoint.items()
                     if 'encoder.' in k or k in model.encoder.state_dict()}
        model.encoder.load_state_dict(enc_state, strict=False)

    print(f"\nStarting training for {config.num_epochs} epochs...")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        train_losses, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch, config, logger, global_step, scaler
        )

        logger.log_epoch_summary(epoch + 1, train_losses)
        logger.log_encoder_status(epoch + 1, epoch < getattr(config, 'freeze_encoder_epochs', 10))

        print(f"Epoch {epoch+1}: Train Loss: {train_losses['total']:.4f}")

        if (epoch + 1) % config.val_interval == 0:
            val_losses, val_psnr = validate(model, val_loader, loss_fn, device, epoch, config, logger)
            print(f"Epoch {epoch+1}: Val Loss: {val_losses['total']:.4f} | PSNR: {val_psnr:.2f} dB")

            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config.to_dict(),
                }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
                print(f"Saved best model (Loss: {best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config.to_dict(),
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    logger.close()
    print("Training complete!")

if __name__ == '__main__':
    main()