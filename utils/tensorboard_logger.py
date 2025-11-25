import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional, TYPE_CHECKING

# Import visualization utilities
from utils.visualization import flow_to_color, create_error_map

# Import model for type hinting
if TYPE_CHECKING:
    from model import LIFT

class TensorBoardLogger:
    """
    Simplified TensorBoard logging for LIFT training.
    """

    def __init__(self, log_dir: str, config, model: 'LIFT'):
        """
        Initialize TensorBoard logger.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)
        self.config = config

        # Intervals
        self.scalar_interval = getattr(config, 'log_interval', 100)
        self.image_interval = getattr(config, 'image_log_interval', 500)

        self._log_hyperparameters(config)
        self._log_model_architecture(model)

        print(f"TensorBoard logging to: {self.log_dir}")

    def _log_hyperparameters(self, config):
        """Log training hyperparameters."""
        hparams = {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_frames': config.num_frames,
            'crop_size': str(config.crop_size),
            'mixed_precision': config.mixed_precision,
        }
        self.writer.add_hparams(hparams, {}, run_name='.')

    def _log_model_architecture(self, model: 'LIFT'):
        """Log model architecture summary."""
        params = model.count_parameters()
        self.writer.add_text('model/architecture', 
                             f"Total Params: {params['total']:,} | Trainable: {params['trainable']:,}", 0)

    def log_training_step(
        self,
        step: int,
        losses: dict,
        lr: float,
        outputs: Optional[dict] = None,
        gt: Optional[torch.Tensor] = None,
        ref_frames: Optional[torch.Tensor] = None,
        batch_time: Optional[float] = None
    ):
        """Log training step metrics."""
        # Scalars
        if step % self.scalar_interval == 0:
            for name, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f'train/loss_{name}', value, step)
            
            self.writer.add_scalar('train/lr', lr, step)
            
            if batch_time:
                self.writer.add_scalar('system/batch_time_ms', batch_time * 1000, step)
            
            if torch.cuda.is_available():
                self.writer.add_scalar('system/vram_gb', 
                                     torch.cuda.memory_allocated() / 1e9, step)

        # Images
        if step % self.image_interval == 0 and outputs is not None and gt is not None:
            self._log_images('train', step, outputs, gt, ref_frames)

    def log_validation(
        self,
        epoch: int,
        avg_losses: dict,
        avg_psnr: float,
        outputs: Optional[dict] = None,
        gt: Optional[torch.Tensor] = None,
        ref_frames: Optional[torch.Tensor] = None
    ):
        """Log validation metrics."""
        self.writer.add_scalar('val/loss_total', avg_losses['total'], epoch)
        self.writer.add_scalar('val/psnr', avg_psnr, epoch)
        
        if outputs is not None and gt is not None:
            self._log_images('val', epoch, outputs, gt, ref_frames)

    def _log_images(
        self, 
        prefix: str, 
        step: int, 
        outputs: dict, 
        gt: torch.Tensor, 
        ref_frames: Optional[torch.Tensor] = None
    ):
        """Helper to log images."""
        pred = outputs['prediction']
        n = min(4, pred.shape[0])  # Log max 4 images

        self.writer.add_images(f'{prefix}/prediction', pred[:n].clamp(0, 1), step)
        self.writer.add_images(f'{prefix}/ground_truth', gt[:n].clamp(0, 1), step)
        self.writer.add_images(f'{prefix}/error_map', create_error_map(pred[:n], gt[:n]), step)

        if ref_frames is not None:
            self.writer.add_images(f'{prefix}/ref_frame_7', ref_frames[:n, 0].clamp(0, 1), step)
            self.writer.add_images(f'{prefix}/ref_frame_9', ref_frames[:n, 1].clamp(0, 1), step)

        if 'flows' in outputs:
            for key in ['flow_7', 'flow_9']:
                if key in outputs['flows']:
                    flow = outputs['flows'][key][:n]
                    self.writer.add_images(f'{prefix}/{key}', flow_to_color(flow), step)

        if 'occlusions' in outputs:
            for key in ['occ_7', 'occ_9']:
                if key in outputs['occlusions']:
                    self.writer.add_images(f'{prefix}/{key}', outputs['occlusions'][key][:n], step)

    def log_epoch_summary(self, epoch: int, train_losses: dict):
        """Log epoch-level summaries."""
        self.writer.add_scalar('epoch/train_loss', train_losses['total'], epoch)

    def log_encoder_status(self, epoch: int, is_frozen: bool):
        """Log encoder status."""
        self.writer.add_scalar('training/encoder_frozen', int(is_frozen), epoch)

    def close(self):
        self.writer.close()