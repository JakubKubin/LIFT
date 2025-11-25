"""
Training script for LIFT model.
Optimized for Intel i7-14700K (20 Cores) + NVIDIA RTX 4070 Ti Super (16GB).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional

from utils.visualization import (
    flow_to_color,
    plot_attention_weights,
    plot_loss_components,
    plot_flow_histogram,
    visualize_occlusion_maps,
    compute_gradient_stats,
    create_error_map,
    create_comparison_grid
)

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

torch.backends.cudnn.benchmark = True

class TensorBoardLogger:
    """
    Comprehensive TensorBoard logging for LIFT training.

    Handles all logging operations with configurable intervals.
    """

    def __init__(self, log_dir: str, config, model: LIFT):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            config: Training configuration
            model: LIFT model instance
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)
        self.config = config

        self.scalar_interval = getattr(config, 'log_interval', 50)
        self.image_interval = getattr(config, 'image_log_interval', 500)
        self.histogram_interval = getattr(config, 'histogram_log_interval', 1000)
        self.gradient_interval = getattr(config, 'gradient_log_interval', 500)

        self._log_hyperparameters(config)
        self._log_model_architecture(model)

        print(f"TensorBoard logging to: {self.log_dir}")
        print(f"  Scalar interval: {self.scalar_interval}")
        print(f"  Image interval: {self.image_interval}")

    def _log_hyperparameters(self, config):
        """Log training hyperparameters."""
        hparams = {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_frames': config.num_frames,
            'crop_size': str(config.crop_size),  # Fix: Convert tuple to string
            'weight_decay': config.weight_decay,
            'encoder_freeze_epochs': getattr(config, 'freeze_encoder_epochs', 10),
            'transformer_layers': config.transformer_layers,
            'transformer_heads': config.transformer_heads,
            'transformer_dim': config.transformer_dim,
            'mixed_precision': config.mixed_precision,
            'loss_l1_weight': config.loss_l1_weight,
            'loss_lap_weight': config.loss_lap_weight,
            'loss_perceptual_weight': config.loss_perceptual_weight,
        }

        self.writer.add_hparams(hparams, {}, run_name='.')

        config_text = "\n".join([f"**{k}**: {v}" for k, v in hparams.items()])
        self.writer.add_text('config/hyperparameters', config_text, 0)

    def _log_model_architecture(self, model: LIFT):
        """Log model architecture summary."""
        params = model.count_parameters()

        arch_text = f"""
## LIFT Model Architecture

### Parameter Count
- **Total**: {params['total']:,}
- **Trainable**: {params['trainable']:,}
- **Frozen**: {params['frozen']:,}

### Modules
- Encoder: FrameEncoder (s1, s4, s8, s16 features)
- Transformer: TemporalAggregator ({self.config.transformer_layers} layers)
- Flow Estimator: 2-scale cascade (s8 -> s4)
- Synthesis: Occlusion-aware blending + Context injection
- Refinement: Full-resolution with s1 features
"""
        self.writer.add_text('model/architecture', arch_text, 0)

    def log_scalars(self, tag_prefix: str, scalars: dict, step: int):
        """
        Log multiple scalar values.

        Args:
            tag_prefix: Prefix for tags (e.g., 'train', 'val')
            scalars: Dictionary of scalar values
            step: Global step
        """
        for name, value in scalars.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f'{tag_prefix}/{name}', value, step)

    def log_training_step(
        self,
        step: int,
        losses: dict,
        lr: float,
        outputs: Optional[dict] = None,
        gt: Optional[torch.Tensor] = None,
        ref_frames: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        batch_time: Optional[float] = None
    ):
        """
        Log training step metrics.

        Args:
            step: Global step
            losses: Dictionary of loss values
            lr: Current learning rate
            outputs: Model outputs (optional, for image logging)
            gt: Ground truth frames (optional)
            ref_frames: Reference frames (optional)
            model: Model for gradient logging (optional)
            batch_time: Time for batch processing (optional)
        """
        if step % self.scalar_interval == 0:
            self.writer.add_scalar('train/loss_total', losses['total'].item(), step)
            self.writer.add_scalar('train/loss_l1', losses['l1'].item(), step)
            self.writer.add_scalar('train/loss_lap', losses['lap'].item(), step)
            self.writer.add_scalar('train/lr', lr, step)

            if 'lpips' in losses:
                self.writer.add_scalar('train/loss_lpips', losses['lpips'].item(), step)
            if 'flow_smooth' in losses and losses['flow_smooth'].item() > 0:
                self.writer.add_scalar('train/loss_flow_smooth', losses['flow_smooth'].item(), step)
            if 'occlusion' in losses and losses['occlusion'].item() > 0:
                self.writer.add_scalar('train/loss_occlusion', losses['occlusion'].item(), step)
            if 'char' in losses:
                self.writer.add_scalar('train/loss_charbonnier', losses['char'].item(), step)

            if batch_time is not None:
                self.writer.add_scalar('system/batch_time_ms', batch_time * 1000, step)
                self.writer.add_scalar('system/throughput_samples_sec',
                                      self.config.batch_size / batch_time, step)

            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / 1e9
                vram_reserved = torch.cuda.memory_reserved() / 1e9
                self.writer.add_scalar('system/vram_allocated_gb', vram_allocated, step)
                self.writer.add_scalar('system/vram_reserved_gb', vram_reserved, step)

        if step % self.image_interval == 0 and outputs is not None and gt is not None:
            self._log_training_images(step, outputs, gt, ref_frames)

        if step % self.gradient_interval == 0 and model is not None:
            self._log_gradients(step, model)

        if step % self.histogram_interval == 0 and outputs is not None:
            self._log_histograms(step, outputs)

    def _log_training_images(
        self,
        step: int,
        outputs: dict,
        gt: torch.Tensor,
        ref_frames: Optional[torch.Tensor] = None
    ):
        """Log training visualizations."""
        pred = outputs['prediction']
        n_samples = min(4, pred.shape[0])

        self.writer.add_images('train/prediction',
                              pred[:n_samples].clamp(0, 1), step)
        self.writer.add_images('train/ground_truth',
                              gt[:n_samples].clamp(0, 1), step)

        error_map = create_error_map(pred[:n_samples], gt[:n_samples])
        self.writer.add_images('train/error_map', error_map, step)

        if ref_frames is not None:
            self.writer.add_images('train/ref_frame_I7',
                                  ref_frames[:n_samples, 0].clamp(0, 1), step)
            self.writer.add_images('train/ref_frame_I9',
                                  ref_frames[:n_samples, 1].clamp(0, 1), step)

        if 'flows' in outputs:
            flow_7 = outputs['flows']['flow_7'][:n_samples]
            flow_9 = outputs['flows']['flow_9'][:n_samples]

            flow_7_vis = flow_to_color(flow_7)
            flow_9_vis = flow_to_color(flow_9)

            self.writer.add_images('train/flow_I7_to_I8', flow_7_vis, step)
            self.writer.add_images('train/flow_I9_to_I8', flow_9_vis, step)

            flow_mag_31 = torch.sqrt(flow_7[:, 0]**2 + flow_7[:, 1]**2)
            flow_mag_32 = torch.sqrt(flow_9[:, 0]**2 + flow_9[:, 1]**2)
            self.writer.add_images('train/flow_magnitude_I7',
                                  flow_mag_31.unsqueeze(1) / (flow_mag_31.max() + 1e-8), step)
            self.writer.add_images('train/flow_magnitude_I9',
                                  flow_mag_32.unsqueeze(1) / (flow_mag_32.max() + 1e-8), step)

        if 'occlusions' in outputs:
            occ_7 = outputs['occlusions']['occ_7'][:n_samples]
            occ_9 = outputs['occlusions']['occ_9'][:n_samples]

            self.writer.add_images('train/occlusion_I7', occ_7, step)
            self.writer.add_images('train/occlusion_I9', occ_9, step)

            occ_diff = occ_7 - occ_9
            occ_diff_vis = (occ_diff + 1) / 2
            self.writer.add_images('train/occlusion_difference', occ_diff_vis, step)

        if 'warped' in outputs:
            self.writer.add_images('train/warped_from_I7',
                                  outputs['warped']['warped_7'][:n_samples].clamp(0, 1), step)
            self.writer.add_images('train/warped_from_I9',
                                  outputs['warped']['warped_9'][:n_samples].clamp(0, 1), step)

        if 'coarse' in outputs:
            coarse_up = F.interpolate(outputs['coarse'][:n_samples],
                                     size=pred.shape[2:], mode='bilinear')
            self.writer.add_images('train/coarse_frame', coarse_up.clamp(0, 1), step)

        if 'attention_weights' in outputs:
            alphas = outputs['attention_weights']
            fig = plot_attention_weights(alphas, num_frames=15, gap_idx=8)
            self.writer.add_figure('train/attention_weights', fig, step)
            plt.close(fig)

            self.writer.add_scalar('train/attention_max', alphas.max().item(), step)
            self.writer.add_scalar('train/attention_entropy',
                                  -(alphas * torch.log(alphas + 1e-8)).sum(dim=-1).mean().item(), step)

    def _log_gradients(self, step: int, model: nn.Module):
        """Log gradient statistics."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        self.writer.add_scalar('gradients/total_norm', total_norm, step)

        module_norms = {}
        for name, module in model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                norm = 0.0
                count = 0
                for p in module.parameters(recurse=False):
                    if p.grad is not None:
                        norm += p.grad.data.norm(2).item() ** 2
                        count += 1
                if count > 0:
                    module_norms[name] = (norm ** 0.5) / count

        for name in ['encoder', 'transformer', 'flow_estimator', 'synthesis', 'refinement']:
            matching = {k: v for k, v in module_norms.items() if name in k}
            if matching:
                avg_norm = sum(matching.values()) / len(matching)
                self.writer.add_scalar(f'gradients/{name}_avg_norm', avg_norm, step)

    def _log_histograms(self, step: int, outputs: dict):
        """Log tensor histograms."""
        pred = outputs['prediction']
        self.writer.add_histogram('histograms/prediction_values', pred, step)

        if 'flows' in outputs:
            flow_7 = outputs['flows']['flow_7']
            flow_mag = torch.sqrt(flow_7[:, 0]**2 + flow_7[:, 1]**2)
            self.writer.add_histogram('histograms/flow_magnitude', flow_mag, step)

        if 'attention_weights' in outputs:
            self.writer.add_histogram('histograms/attention_weights',
                                     outputs['attention_weights'], step)

    def log_validation(
        self,
        epoch: int,
        avg_losses: dict,
        avg_psnr: float,
        avg_ssim: Optional[float] = None,
        outputs: Optional[dict] = None,
        gt: Optional[torch.Tensor] = None,
        ref_frames: Optional[torch.Tensor] = None
    ):
        """
        Log validation metrics and visualizations.

        Args:
            epoch: Current epoch
            avg_losses: Average losses over validation set
            avg_psnr: Average PSNR
            avg_ssim: Average SSIM (optional)
            outputs: Sample outputs for visualization (optional)
            gt: Sample ground truth (optional)
            ref_frames: Sample reference frames (optional)
        """
        self.writer.add_scalar('val/loss_total', avg_losses['total'], epoch)
        self.writer.add_scalar('val/loss_l1', avg_losses['l1'], epoch)
        self.writer.add_scalar('val/loss_lap', avg_losses.get('lap', 0), epoch)
        self.writer.add_scalar('val/psnr', avg_psnr, epoch)

        if avg_ssim is not None:
            self.writer.add_scalar('val/ssim', avg_ssim, epoch)

        if outputs is not None and gt is not None:
            self._log_validation_images(epoch, outputs, gt, ref_frames)

    def _log_validation_images(
        self,
        epoch: int,
        outputs: dict,
        gt: torch.Tensor,
        ref_frames: Optional[torch.Tensor] = None
    ):
        """Log validation visualizations."""
        pred = outputs['prediction']
        n_samples = min(4, pred.shape[0])

        self.writer.add_images('val/prediction',
                              pred[:n_samples].clamp(0, 1), epoch)
        self.writer.add_images('val/ground_truth',
                              gt[:n_samples].clamp(0, 1), epoch)

        error_map = create_error_map(pred[:n_samples], gt[:n_samples])
        self.writer.add_images('val/error_map', error_map, epoch)

        if ref_frames is not None:
            self.writer.add_images('val/ref_frame_I7',
                                  ref_frames[:n_samples, 0].clamp(0, 1), epoch)
            self.writer.add_images('val/ref_frame_I9',
                                  ref_frames[:n_samples, 1].clamp(0, 1), epoch)

        if 'flows' in outputs:
            flow_7_vis = flow_to_color(outputs['flows']['flow_7'][:n_samples])
            flow_9_vis = flow_to_color(outputs['flows']['flow_9'][:n_samples])
            self.writer.add_images('val/flow_I7_to_I8', flow_7_vis, epoch)
            self.writer.add_images('val/flow_I9_to_I8', flow_9_vis, epoch)

        if 'occlusions' in outputs:
            self.writer.add_images('val/occlusion_I7',
                                  outputs['occlusions']['occ_7'][:n_samples], epoch)
            self.writer.add_images('val/occlusion_I9',
                                  outputs['occlusions']['occ_9'][:n_samples], epoch)

            fig = visualize_occlusion_maps(
                outputs['occlusions']['occ_7'][:1],
                outputs['occlusions']['occ_9'][:1]
            )
            self.writer.add_figure('val/occlusion_analysis', fig, epoch)
            plt.close(fig)

        if 'attention_weights' in outputs:
            fig = plot_attention_weights(
                outputs['attention_weights'],
                title=f"Validation Attention Weights (Epoch {epoch})"
            )
            self.writer.add_figure('val/attention_weights', fig, epoch)
            plt.close(fig)

    def log_epoch_summary(
        self,
        epoch: int,
        train_losses: dict,
        val_losses: Optional[dict] = None,
        val_psnr: Optional[float] = None,
        epoch_time: Optional[float] = None
    ):
        """Log epoch summary metrics."""
        self.writer.add_scalar('epoch/train_loss', train_losses['total'], epoch)

        if val_losses is not None:
            self.writer.add_scalar('epoch/val_loss', val_losses['total'], epoch)

        if val_psnr is not None:
            self.writer.add_scalar('epoch/val_psnr', val_psnr, epoch)

        if epoch_time is not None:
            self.writer.add_scalar('epoch/time_seconds', epoch_time, epoch)

        loss_dict = {k: v for k, v in train_losses.items() if v > 0}
        if loss_dict:
            fig = plot_loss_components(loss_dict, title=f"Train Loss Breakdown (Epoch {epoch})")
            self.writer.add_figure('epoch/loss_components', fig, epoch)
            plt.close(fig)

    def log_encoder_status(self, epoch: int, is_frozen: bool):
        """Log encoder freeze/unfreeze status."""
        self.writer.add_scalar('training/encoder_frozen', int(is_frozen), epoch)

        status = "FROZEN" if is_frozen else "TRAINABLE"
        self.writer.add_text('training/encoder_status',
                            f"Epoch {epoch}: Encoder is **{status}**", epoch)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

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


def train_epoch(model: LIFT, dataloader, optimizer, scheduler, loss_fn: LIFTLoss, device, epoch, config, writer, global_step, scaler):
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
                flow1=outputs['flows']['flow_7'],
                flow2=outputs['flows']['flow_9'],
                logit_occ1=outputs['occlusions']['logit_occ_7'],
                logit_occ2=outputs['occlusions']['logit_occ_9'],
                warped1=outputs['warped']['warped_7'],
                warped2=outputs['warped']['warped_9']
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


def validate(model: LIFT, dataloader, loss_fn: LIFTLoss, device, epoch, config, writer):
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

    # Initialize logger
    logger = TensorBoardLogger(config.log_dir, config, model)

    # Load checkpoint
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # --- POPRAWKA: Filtrowanie niezgodnych kształtów (szczególnie pos_enc) ---
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
        # -------------------------------------------------------------------------

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "="*60)
    print(f"Starting training on {config.num_epochs} epochs...")
    print(f"Batch Size: {config.batch_size} | Workers: {config.num_workers} | AMP: Enabled")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_losses, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch, config, logger.writer, global_step, scaler
        )
        print(f"Epoch {epoch+1}: Train Loss: {train_losses['total']:.4f} (L1: {train_losses['l1']:.4f})")

        # Log epoch summary
        logger.log_epoch_summary(epoch + 1, train_losses)
        logger.log_encoder_status(epoch + 1, epoch < getattr(config, 'freeze_encoder_epochs', 10))

        # Validate
        if (epoch + 1) % config.val_interval == 0 or (epoch + 1) % 2 == 0:
            val_losses, val_psnr = validate(model, val_loader, loss_fn, device, epoch, config, logger.writer)
            print(f"Epoch {epoch+1}: Val Loss: {val_losses['total']:.4f} | PSNR: {val_psnr:.2f} dB")

            # Log validation summary
            logger.log_epoch_summary(epoch + 1, train_losses, val_losses, val_psnr)

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

    logger.close()
    print("Training complete!")

if __name__ == '__main__':
    main()