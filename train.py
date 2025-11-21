"""
Training script for LIFT model.

Complete implementation with all stages integrated.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np

from dataset import Vimeo64Dataset, collate_fn
from model import LIFT, LIFTLoss
from configs.default import Config


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


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, epoch, config, writer, global_step):
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
        # Move data to device
        frames = batch['frames'].to(device)          # [B, 64, 3, H, W]
        ref_frames = batch['ref_frames'].to(device)  # [B, 2, 3, H, W]
        gt = batch['gt'].to(device)                  # [B, 3, H, W]
        timestep = batch['timestep'].to(device)      # [B]

        # Forward pass
        outputs = model(frames, ref_frames, timestep[0].item())
        pred = outputs['prediction']

        # Compute loss
        losses = loss_fn(
            pred, gt,
            flow1=outputs['flows']['flow_31'],
            flow2=outputs['flows']['flow_32'],
            occ1=outputs['occlusions']['occ_31'],
            occ2=outputs['occlusions']['occ_32'],
            warped1=None,  # Can add warped frames for occlusion loss
            warped2=None
        )
        loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()
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

            # Log attention weights visualization
            attn = outputs['attention_weights'][0].cpu().numpy()
            writer.add_histogram('train/attention_weights', attn, global_step)

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
            frames = batch['frames'].to(device)
            ref_frames = batch['ref_frames'].to(device)
            gt = batch['gt'].to(device)
            timestep = batch['timestep'].to(device)

            # Forward pass
            outputs = model(frames, ref_frames, timestep[0].item())
            pred = outputs['prediction']

            # Compute loss
            losses = loss_fn(pred, gt)

            # Compute PSNR
            psnr = compute_psnr(pred, gt)

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


def main():
    parser = argparse.ArgumentParser(description='Train LIFT model')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                       help='Path to pretrained encoder weights')
    args = parser.parse_args()

    # Configuration
    config = Config()
    config.data_root = args.data_root
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = Vimeo64Dataset(
        data_root=config.data_root,
        mode='train',
        num_frames=config.num_frames,
        crop_size=config.crop_size,
        augment=True,
        input_scale=config.input_scale
    )

    val_dataset = Vimeo64Dataset(
        data_root=config.data_root,
        mode='val',
        num_frames=config.num_frames,
        crop_size=config.crop_size,  # Use same size for validation
        augment=False,
        input_scale=config.input_scale
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )

    # Create model
    print("\nCreating LIFT model...")
    model = LIFT(config).to(device)

    # Load pretrained encoder if provided
    if args.pretrained_encoder:
        print(f"Loading pretrained encoder from {args.pretrained_encoder}")
        checkpoint = torch.load(args.pretrained_encoder, map_location=device)
        model.encoder.load_state_dict(checkpoint, strict=False)

    # Print model info
    params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))

    # Create loss function
    loss_fn = LIFTLoss(config)

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"Resuming from epoch {start_epoch}")

    # TensorBoard writer
    writer = SummaryWriter(config.log_dir)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch, config, writer, global_step
        )
        print(f"Train - Loss: {train_losses['total']:.4f}, L1: {train_losses['l1']:.4f}, Lap: {train_losses['lap']:.4f}")

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_losses, val_psnr = validate(model, val_loader, loss_fn, device, epoch, config, writer)
            print(f"Val - Loss: {val_losses['total']:.4f}, PSNR: {val_psnr:.2f} dB")

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f'best_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_psnr': val_psnr,
                    'config': config.to_dict(),
                }, checkpoint_path)
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

    writer.close()
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()
