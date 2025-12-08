import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional
import torchvision.utils as vutils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from utils.visualization import flow_to_color, create_error_map

from model import LIFT

class TensorBoardLogger:
    """
    Simplified TensorBoard logging for LIFT training.
    """

    def __init__(self, log_dir: str, config, model: 'LIFT'):
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
        if hasattr(model, 'count_parameters'):
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
        """
        Logs a side-by-side comparison grid: [Ref7 | GT | Pred | Ref9 | Error]
        """
        # Take the first sample from the batch [0]
        pred = outputs['prediction'][0].clamp(0, 1)
        target = gt[0].clamp(0, 1)

        # Error Map [1, H, W] -> repeat to [3, H, W] for visualization
        # We unsqueeze to [1, C, H, W] for utility, get [0] result back
        err = create_error_map(pred.unsqueeze(0), target.unsqueeze(0), amplify=5.0)[0]

        # Prepare list for stacking
        images_to_stack = []

        # 1. Ref 7
        if ref_frames is not None:
            ref7 = ref_frames[0, 0].clamp(0, 1)
            images_to_stack.append(ref7)

        # 2. GT
        images_to_stack.append(target)

        # 3. Pred
        images_to_stack.append(pred)

        # 4. Ref 9
        if ref_frames is not None:
            ref9 = ref_frames[0, 1].detach().clamp(0, 1)
            images_to_stack.append(ref9)

        # 5. Error
        images_to_stack.append(err)

        # Create Grid
        grid = vutils.make_grid(images_to_stack, nrow=len(images_to_stack), padding=2, normalize=False)
        self.writer.add_image(f'{prefix}/comparison_side_by_side', grid, step)

        # Log Occlusions
        if 'occlusions' in outputs:
            occ_list = []
            if 'occ_7' in outputs['occlusions']:
                # Expand 1-channel mask to 3-channel for visualization
                occ_list.append(outputs['occlusions']['occ_7'][0].detach().repeat(3, 1, 1))
            if 'occ_9' in outputs['occlusions']:
                occ_list.append(outputs['occlusions']['occ_9'][0].detach().repeat(3, 1, 1))

            if occ_list:
                occ_grid = vutils.make_grid(occ_list, nrow=len(occ_list), padding=2)
                self.writer.add_image(f'{prefix}/occlusions', occ_grid, step)

        # Log Flow
        if 'flows' in outputs:
            flow_list = []
            if 'flow_7' in outputs['flows']:
                # flow_to_color returns [B, 3, H, W], we take [0]
                flow_list.append(flow_to_color(outputs['flows']['flow_7'][:1])[0].detach())
            if 'flow_9' in outputs['flows']:
                flow_list.append(flow_to_color(outputs['flows']['flow_9'][:1])[0].detach())

            if flow_list:
                flow_grid = vutils.make_grid(flow_list, nrow=len(flow_list), padding=2)
                self.writer.add_image(f'{prefix}/optical_flow', flow_grid, step)

    def log_epoch_summary(self, epoch: int, train_losses: dict):
        """Log epoch-level summaries."""
        total_loss = train_losses['total']
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        self.writer.add_scalar('epoch/train_loss', total_loss, epoch)

    def log_encoder_status(self, epoch: int, is_frozen: bool):
        """Log encoder status."""
        self.writer.add_scalar('training/encoder_frozen', int(is_frozen), epoch)

    def close(self):
        self.writer.close()