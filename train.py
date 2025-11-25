"""
Training script for LIFT model.
Optimized for Intel i7-14700K (20 Cores) + NVIDIA RTX 4070 Ti Super (16GB).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

from utils.visualization import flow_to_color

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Import all available datasets
from dataset import (
    Vimeo15Dataset,
    X4K1000FPSDataset,
    UCF101Dataset,
    collate_fn
)
from model import LIFT, LIFTLoss
from configs.default import Config

# --- HARDWARE OPTIMIZATION 1: CuDNN Benchmark ---
# Why: Your crop_size is fixed (e.g., 224x224).
# This allows CuDNN to benchmark convolution algorithms once and pick the fastest one for your 4070 Ti.
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def log_images_to_tensorboard(writer, outputs, gt, ref_frames, epoch, prefix='train'):
    """Log images, flows, occlusions to TensorBoard."""
    pred = outputs['prediction']

    writer.add_images(f'{prefix}/prediction', pred[:4].clamp(0, 1), epoch)
    writer.add_images(f'{prefix}/ground_truth', gt[:4].clamp(0, 1), epoch)
    writer.add_images(f'{prefix}/ref_frame_7', ref_frames[:4, 0].clamp(0, 1), epoch)
    writer.add_images(f'{prefix}/ref_frame_9', ref_frames[:4, 1].clamp(0, 1), epoch)

    if 'flows' in outputs:
        flow_31_vis = flow_to_color(outputs['flows']['flow_31'][:4])
        flow_32_vis = flow_to_color(outputs['flows']['flow_32'][:4])
        writer.add_images(f'{prefix}/flow_to_7', flow_31_vis, epoch)
        writer.add_images(f'{prefix}/flow_to_9', flow_32_vis, epoch)

    if 'occlusions' in outputs:
        occ_31 = outputs['occlusions']['occ_31'][:4].expand(-1, 3, -1, -1)
        occ_32 = outputs['occlusions']['occ_32'][:4].expand(-1, 3, -1, -1)
        writer.add_images(f'{prefix}/occlusion_7', occ_31, epoch)
        writer.add_images(f'{prefix}/occlusion_9', occ_32, epoch)

    if 'warped' in outputs:
        writer.add_images(f'{prefix}/warped_from_7', outputs['warped']['warped_31'][:4].clamp(0, 1), epoch)
        writer.add_images(f'{prefix}/warped_from_9', outputs['warped']['warped_32'][:4].clamp(0, 1), epoch)

    if 'attention_weights' in outputs:
        alphas = outputs['attention_weights']
        fig, ax = plt.subplots(figsize=(10, 3))
        x = [i for i in range(15) if i != 8]
        ax.bar(x, alphas[0].cpu().numpy())
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Temporal Attention Weights (α)')
        writer.add_figure(f'{prefix}/attention_weights', fig, epoch)
        plt.close(fig)

def get_optimizer(model, config):
    """Create AdamW optimizer."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    return optimizer


def get_lr_scheduler(optimizer, config, steps_per_epoch):
    """
    Create learning rate scheduler with warmup and cosine decay.
    """
    def lr_lambda(step):
        if step < config.lr_warmup_steps:
            # Linear warmup
            return step / config.lr_warmup_steps
        else:
            # Cosine decay
            total_steps = config.num_epochs * steps_per_epoch
            progress = (step - config.lr_warmup_steps) / (total_steps - config.lr_warmup_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            # Scale from lr to lr_min
            return cosine_decay * (1.0 - config.lr_min / config.learning_rate) + config.lr_min / config.learning_rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def compute_psnr(pred, target) -> torch.Tensor:
    """Compute PSNR between predicted and target images."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return -10 * torch.log10(mse)


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, epoch, config, writer, global_step, scaler):
    """Train for one epoch."""
    model.train()
    model.set_epoch(epoch)  # Handle encoder freezing

    total_losses = {
        'total': 0.0,
        'l1': 0.0,
        'lap': 0.0,
        'flow_smooth': 0.0,
        'occlusion': 0.0
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for step, batch in enumerate(pbar):
        # Move data to device (Non-blocking allows overlap with computation)
        frames = batch['frames'].to(device, non_blocking=True)
        ref_frames = batch['ref_frames'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)
        timestep = batch['timestep'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast('cuda', enabled=config.mixed_precision):
            # Forward pass
            outputs = model(frames, ref_frames, timestep[0].item())
            pred = outputs['prediction']

            # Compute loss
            losses = loss_fn(
                pred, gt,
                flow1=outputs['flows']['flow_31'],
                flow2=outputs['flows']['flow_32'],
                logit_occ1=outputs['occlusions']['logit_occ_31'],
                logit_occ2=outputs['occlusions']['logit_occ_32'],
                warped1=outputs['warped']['warped_31'],
                warped2=outputs['warped']['warped_32']
            )
            loss = losses['total']

        # Backward pass with gradient scaling (prevents underflow in FP16)
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        # Accumulate losses
        for key in total_losses.keys():
            if key in losses:
                total_losses[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

        # Log to tensorboard
        if global_step % config.log_interval == 0:
            writer.add_scalar('train/loss_total', loss.item(), global_step)
            writer.add_scalar('train/loss_l1', losses['l1'].item(), global_step)
            writer.add_scalar('train/loss_lap', losses['lap'].item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

            # Optional: Log VRAM usage
            if step % 100 == 0:
                vram_gb = torch.cuda.memory_allocated() / 1e9
                writer.add_scalar('system/vram_gb', vram_gb, global_step)

        global_step += 1

    # Compute average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses, global_step


def validate(model, dataloader, loss_fn, device, epoch, config, writer):
    """Validation loop."""
    model.eval()

    total_losses = {
        'total': 0.0,
        'l1': 0.0,
        'lap': 0.0,
    }
    total_psnr = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            frames = batch['frames'].to(device, non_blocking=True)
            ref_frames = batch['ref_frames'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            timestep = batch['timestep'].to(device, non_blocking=True)

            with torch.autocast('cuda', enabled=config.mixed_precision):
                outputs = model(frames, ref_frames, timestep[0].item())
                pred = outputs['prediction']
                losses = loss_fn(pred, gt)

            # Compute PSNR (ensure float32 for accuracy)
            psnr = compute_psnr(pred.float(), gt.float())

            # Accumulate
            for key in total_losses.keys():
                if key in losses:
                    total_losses[key] += losses[key].item()
            total_psnr += psnr.item()
            num_batches += 1

    # Average
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_psnr = total_psnr / num_batches

    # Log to tensorboard
    writer.add_scalar('val/loss_total', avg_losses['total'], epoch)
    writer.add_scalar('val/loss_l1', avg_losses['l1'], epoch)
    writer.add_scalar('val/psnr', avg_psnr, epoch)

    return avg_losses, avg_psnr


def get_dataset_class(dataset_name):
    """Factory function to get the correct dataset class."""
    if dataset_name.lower() == 'vimeo':
        return Vimeo15Dataset
    elif dataset_name.lower() == 'x4k':
        return X4K1000FPSDataset
    elif dataset_name.lower() == 'ucf101':
        return UCF101Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choices: vimeo, x4k, ucf101")


def main():
    import sys
    from configs.default import Config

    parser = argparse.ArgumentParser(description='Train LIFT model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--dataset', type=str, default='vimeo', choices=['vimeo', 'x4k', 'ucf101'],
                        help='Dataset type (vimeo, x4k, ucf101)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='Path to pretrained encoder weights')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of dataloader workers (overrides config)')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Override number of frames (default from config is 7, but LIFT needs 15)')
    parser.add_argument('--max_sequences', type=int, default=None,
                    help='Limit total number of sequences per dataset (train/val)')
    args = parser.parse_args()

    # Configuration
    config = Config()
    config.data_root = args.data_root
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # Handle num_frames override (Fixing the default.py vs LIFT requirement)
    if args.num_frames is not None:
        config.num_frames = args.num_frames
    elif config.num_frames != 15:
        print(f"WARNING: config.num_frames is {config.num_frames}. If training LIFT, you likely want --num_frames 15.")

    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Create datasets
    print(f"\nLoading {args.dataset.upper()} datasets from {config.data_root}...")

    DatasetClass = get_dataset_class(args.dataset)

    try:
        train_dataset = DatasetClass(
            data_root=config.data_root,
            mode='train',
            num_frames=config.num_frames,
            crop_size=config.crop_size,
            augment=True,
            input_scale=config.input_scale,
            max_sequences=args.max_sequences
        )

        val_dataset = DatasetClass(
            data_root=config.data_root,
            mode='val',
            num_frames=config.num_frames,
            crop_size=config.crop_size,
            augment=False,
            input_scale=config.input_scale,
            max_sequences=args.max_sequences
        )
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {config.data_root}. Check directory structure.")

    # --- HARDWARE OPTIMIZATION 4: DataLoader Settings ---
    # persistent_workers=True: Keeps worker processes alive between epochs.
    # This is crucial for short epochs or when worker startup time is high (typical with 15 frames/large libraries).
    # prefetch_factor=2: Buffers 2 batches per worker to ensure GPU always has data.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Create model
    print("\nCreating LIFT model...")
    model = LIFT(config).to(device)

    # Initialize GradScaler
    # Jeśli mixed_precision=False, scaler będzie działał w trybie "passthrough" (nic nie robi)
    scaler = torch.GradScaler('cuda', enabled=config.mixed_precision)

    # Load pretrained encoder
    if args.pretrained_encoder:
        print(f"Loading pretrained encoder from {args.pretrained_encoder}")
        checkpoint = torch.load(args.pretrained_encoder, map_location=device)

        encoder_state = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): k = k[7:]
            encoder_state[k] = v

        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in encoder_state.items()
            if 'encoder.' in k or k in model.encoder.state_dict()
        }
        try:
            model.encoder.load_state_dict(encoder_state, strict=False)
            print("Pretrained encoder loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load some encoder weights: {e}")

    # Print info
    params = model.count_parameters()
    print(f"\nModel parameters: Total: {params['total']:,}")

    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))

    # --- FIX: Move loss function to GPU ---
    # The previous error was because loss buffers (kernels) were on CPU.
    loss_fn = LIFTLoss(config).to(device)

    # Load checkpoint
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_state = model.state_dict()
        checkpoint_state = checkpoint['model_state_dict']

        # Tworzymy nowy słownik, pomijając klucze o niezgodnych wymiarach
        new_state_dict = {}
        for k, v in checkpoint_state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state[k].shape}")
            else:
                # Opcjonalnie: pomijamy klucze, których nie ma w nowym modelu
                pass

        # Ładujemy przefiltrowane wagi (strict=False pozwala na brakujące klucze np. pos_enc)
        model.load_state_dict(new_state_dict, strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming from epoch {start_epoch}")

    writer = SummaryWriter(config.log_dir)

    print("\n" + "="*60)
    print(f"Starting training on {config.num_epochs} epochs...")
    print(f"Batch Size: {config.batch_size} | Workers: {config.num_workers} | AMP: Enabled")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch, config, writer, global_step, scaler
        )
        print(f"Epoch {epoch+1}: Train Loss: {train_losses['total']:.4f} (L1: {train_losses['l1']:.4f})")

        # Validate
        if (epoch + 1) % config.val_interval == 0 or (epoch + 1) % 2 == 0:
            val_losses, val_psnr = validate(model, val_loader, loss_fn, device, epoch, config, writer)
            print(f"Epoch {epoch+1}: Val Loss: {val_losses['total']:.4f} | PSNR: {val_psnr:.2f} dB")

            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_psnr': val_psnr,
                    'config': config.to_dict(),
                }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config.to_dict(),
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    writer.close()
    print("Training complete!")

if __name__ == '__main__':
    main()