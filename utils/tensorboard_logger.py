import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional
import torchvision.utils as vutils
import io
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from utils.visualization import flow_to_color, create_error_map

from model import LIFT, backward_warp

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
    
    def log_attention_weights(self, step, weights):
        """
        Loguje histogram wag atencji dla klatek 0-14 (lub 0-T-1).
        weights: [B, T]
        """
        if weights is None: return
        
        avg_weights = weights.mean(dim=0).detach().cpu().numpy() # [T]
        T = len(avg_weights)
        
        # Number of frames (0 to T-1)
        frames = range(T)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(frames, avg_weights, color='skyblue')
        
        # Etykiety
        ax.set_xlabel('Indeks Klatki (0 do 14)')
        ax.set_ylabel('Średnia Waga Atencji')
        ax.set_title(f'Temporal Attention Weights (Step {step})')
        ax.set_xticks(frames)
        
        mid_idx = T // 2
        ref_indices = [mid_idx - 1, mid_idx + 1]
        
        for idx in ref_indices:
            if 0 <= idx < T:
                 ax.bar(idx, avg_weights[idx], color='red', alpha=0.7)
                 ax.text(idx, avg_weights[idx], 'Ref', ha='center', va='bottom', color='black', fontweight='bold')
        
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image_pil = Image.open(buf)
        image_tensor = to_tensor(image_pil)
        plt.close(fig)
        buf.close()

        self.writer.add_image('debug/attention_weights', image_tensor, step)
    
    def log_warping_sanity_check(self, step: int, ref_frame: torch.Tensor, flow: torch.Tensor, target: torch.Tensor, tag: str = "check"):
        """
        Logs a visual sanity check for warping to TensorBoard.
        Visualizes: [Ref | Flow | Warped | Target | Error]
        """
        # Używamy pierwszego elementu z batcha [1, C, H, W]
        ref = ref_frame[0:1].detach()
        fl = flow[0:1].detach()
        tgt = target[0:1].detach()

        if fl.shape[-1] != ref.shape[-1]:
            scale_factor = ref.shape[-1] / fl.shape[-1]
            # Upsample flow (bilinear)
            fl = F.interpolate(fl, size=ref.shape[-2:], mode='bilinear', align_corners=False)
            # Scale flow vectors (important!): if the image is 4x larger, the displacement must also be 4x larger
            fl = fl * scale_factor

        with torch.no_grad():
            # 1. Perform Warping (using flow generated by the model)
            warped = backward_warp(ref, fl)
            
            # 2. Calculate reconstruction error (does the warped frame resemble the target?)
            diff = torch.abs(warped - tgt).mean()
            self.writer.add_scalar(f'sanity/{tag}_diff', diff.item(), step)

            # 3. Prepare visualization (CPU, Numpy)
            img_ref = ref[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            img_flow = flow_to_color(fl)[0].cpu().numpy().transpose(1, 2, 0)
            img_warped = warped[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            img_tgt = tgt[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            
            # Error Map
            img_err_tensor = create_error_map(warped, tgt, amplify=5.0)[0]
            img_err = img_err_tensor.cpu().permute(1, 2, 0).numpy()

            # Matplotlib figure
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            
            ax[0].imshow(img_ref)
            ax[0].set_title("Reference (Source)")
            
            ax[1].imshow(img_flow)
            ax[1].set_title(f"Flow (Upped)\nMax: {fl.abs().max():.1f}px")
            
            ax[2].imshow(img_warped)
            ax[2].set_title(f"Warped (Result)\nDiff: {diff:.4f}")
            
            ax[3].imshow(img_tgt)
            ax[3].set_title("Target (Goal)")
            
            ax[4].imshow(img_err)
            ax[4].set_title("Error x5")

            for a in ax: a.axis('off')
            plt.tight_layout()

            # Konwersja plot -> tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            image_pil = Image.open(buf)
            image_tensor = to_tensor(image_pil)
            plt.close(fig)
            buf.close()
            
            self.writer.add_image(f'sanity/{tag}_visual', image_tensor, step)

    def _log_images(
        self,
        prefix: str,
        step: int,
        outputs: dict,
        gt: torch.Tensor,
        ref_frames: Optional[torch.Tensor] = None
    ):
        """
        Logs a side-by-side comparison grid with labels: [Ref7 | GT | Pred | Ref9 | Error]
        Uses Matplotlib to add titles.
        """
        pred = outputs['prediction'][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        target = gt[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        # Error Map
        err_tensor = create_error_map(outputs['prediction'][:1], gt[:1], amplify=5.0)[0]
        err = err_tensor.detach().cpu().permute(1, 2, 0).numpy()

        # Tuple list (Image, Title)
        images_with_labels = []

        # 1. Ref 7 (Input)
        if ref_frames is not None:
            ref7 = ref_frames[0, 0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            images_with_labels.append((ref7, "Ref 7 (Input)"))

        # 2. GT
        images_with_labels.append((target, "Ground Truth"))

        # 3. Pred
        images_with_labels.append((pred, "LIFT Prediction"))

        # 4. Ref 9 (Input)
        if ref_frames is not None:
            ref9 = ref_frames[0, 1].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            images_with_labels.append((ref9, "Ref 9 (Input)"))

        # 5. Error
        images_with_labels.append((err, "Error Map (x5)"))

        # 2. Rysowanie wykresu Matplotlib
        num_imgs = len(images_with_labels)
        fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 4))
        
        if num_imgs == 1: axes = [axes] # Obsługa przypadku jednego obrazka

        for ax, (img, title) in zip(axes, images_with_labels):
            ax.imshow(img)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()

        # 3. Konwersja plot -> tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image_pil = Image.open(buf)
        image_tensor = to_tensor(image_pil)
        plt.close(fig)
        buf.close()

        self.writer.add_image(f'{prefix}/comparison', image_tensor, step)

        # Log Occlusions
        if 'occlusions' in outputs:
            occ_list = []
            if 'occ_7' in outputs['occlusions']:
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