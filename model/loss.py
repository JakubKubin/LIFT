"""
Loss functions for LIFT model training.

Includes:
- L1 Loss
- Laplacian Pyramid Loss (multi-scale L1)
- Perceptual Loss (VGG-based)
- Flow smoothness loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

def gaussian_kernel(size=5, sigma=1.0, channels=3):
    """
    Create Gaussian kernel for pyramid construction.

    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation
        channels: Number of channels

    Returns:
        Gaussian kernel tensor [channels, 1, size, size]
    """
    # Create 1D Gaussian
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    # Create 2D Gaussian
    g2d = g[:, None] * g[None, :]
    g2d = g2d / g2d.sum()

    # Expand for multiple channels
    kernel = g2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    return kernel


class GaussianBlur(nn.Module):
    """Gaussian blur operation for pyramid construction."""

    kernel: torch.Tensor

    def __init__(self, channels=3):
        super().__init__()
        kernel = gaussian_kernel(size=5, sigma=1.0, channels=channels)
        self.register_buffer('kernel', kernel)
        self.channels = channels

    def forward(self, x):
        # Pad to maintain size
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        # Apply convolution
        return F.conv2d(x, self.kernel, groups=self.channels)


class LaplacianPyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss.

    Computes L1 loss at multiple scales using Laplacian pyramids.
    This captures both high-frequency details and low-frequency structure.
    """

    def __init__(self, max_levels=5, channels=3):
        super().__init__()
        self.max_levels = max_levels
        self.blur = GaussianBlur(channels=channels)

    def downsample(self, x):
        """Downsample by factor of 2."""
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        """Upsample by factor of 2 using zero insertion + Gaussian blur."""
        B, C, H, W = x.shape

        # Create upsampled tensor with zeros
        x_up = torch.zeros(B, C, H * 2, W * 2, device=x.device, dtype=x.dtype)
        x_up[:, :, ::2, ::2] = x

        # Blur to interpolate
        x_up = self.blur(x_up) * 4  # Scale by 4 to compensate for zero insertion

        return x_up

    def build_pyramid(self, x):
        """
        Build Laplacian pyramid.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of Laplacian levels (high to low frequency)
        """
        pyramid = []
        current = x

        for level in range(self.max_levels):
            # Blur and downsample
            blurred = self.blur(current)
            downsampled = self.downsample(blurred)

            # Upsample back
            upsampled = self.upsample(downsampled)

            # Laplacian = original - upsampled
            # This captures the high-frequency details lost in downsampling
            if current.shape == upsampled.shape:
                 laplacian = current - upsampled
            else:
                 # Handle edge cases where upsampling might not match exactly due to odd dimensions
                 # Crop or pad if necessary, though usually power-of-2 dimensions avoid this
                 # For simplicity here, assume dimensions match or resize upsampled
                 upsampled = F.interpolate(upsampled, size=current.shape[2:], mode='bilinear', align_corners=False)
                 laplacian = current - upsampled

            pyramid.append(laplacian)

            # Move to next level
            current = downsampled

        return pyramid

    def forward(self, pred, target):
        """
        Compute Laplacian pyramid loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            Scalar loss value
        """
        pred_pyramid = self.build_pyramid(pred)
        target_pyramid = self.build_pyramid(target)

        loss = 0
        for pred_level, target_level in zip(pred_pyramid, target_pyramid):
            loss += F.l1_loss(pred_level, target_level)

        return loss


class FlowSmoothnessLoss(nn.Module):
    """
    Flow smoothness loss using first-order gradients.

    Encourages smooth flow fields while allowing sharp edges.
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow):
        """
        Compute smoothness loss on flow field.

        Args:
            flow: Flow tensor [B, 2, H, W]

        Returns:
            Scalar smoothness loss
        """
        # Horizontal gradients
        grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])

        # Vertical gradients
        grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])

        return grad_x.mean() + grad_y.mean()


class OcclusionLoss(nn.Module):
    """
    Loss for occlusion map prediction.

    Encourages occlusion maps to identify correctly which reference
    frame is visible at each location.
    """

    def __init__(self):
        super().__init__()

    def forward(self, occ1_logits, occ2_logits, warped1, warped2, gt):
        """
        Compute occlusion loss.

        Args:
            occ1_logits: Occlusion logits for frame 1 [B, 1, H_s4, W_s4]
            occ2_logits: Occlusion logits for frame 2 [B, 1, H_s4, W_s4]
            warped1: Warped frame 1 [B, 3, H_s4, W_s4]
            warped2: Warped frame 2 [B, 3, H_s4, W_s4]
            gt: Ground truth [B, 3, H, W] (Full resolution)

        Returns:
            Scalar occlusion loss
        """
        if gt.shape[-2:] != warped1.shape[-2:]:
            gt = F.interpolate(gt, size=warped1.shape[-2:], mode='bilinear', align_corners=False)

        # Compute photometric errors
        error1 = torch.abs(warped1 - gt).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        error2 = torch.abs(warped2 - gt).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Ideal occlusion maps (1 where error is low)
        ideal_occ1 = (error1 < error2).float()
        ideal_occ2 = (error2 < error1).float()

        # BCE loss between predicted and ideal occlusion
        loss = F.binary_cross_entropy_with_logits(occ1_logits, ideal_occ1) + \
               F.binary_cross_entropy_with_logits(occ2_logits, ideal_occ2)

        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 loss).

    More robust to outliers than L2 loss, smoother than L1 loss.
    """

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Compute Charbonnier loss.

        Args:
            pred: Predicted tensor
            target: Target tensor

        Returns:
            Scalar loss
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


class LIFTLoss(nn.Module):
    """
    Combined loss for LIFT model.

    Combines:
    - Laplacian pyramid loss (multi-scale L1)
    - Charbonnier loss (robust L1)
    - Flow smoothness loss
    - Occlusion loss (optional)
    """

    def __init__(self, config):
        super().__init__()

        self.lap_loss = LaplacianPyramidLoss(max_levels=5, channels=3)
        self.char_loss = CharbonnierLoss()
        self.flow_smooth_loss = FlowSmoothnessLoss()
        self.occ_loss = OcclusionLoss()
        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

        # Loss weights from config
        self.w_l1 = config.loss_l1_weight
        self.w_lap = config.loss_lap_weight
        self.w_flow = config.loss_flow_weight
        self.w_occ = config.loss_occlusion_weight
        self.w_perc = config.loss_perceptual_weight

        self.mixed_precision = config.mixed_precision

    def forward(self, pred, target, flow1=None, flow2=None,
                logit_occ1=None, logit_occ2=None,
                warped1=None, warped2=None):
        """
        Compute total loss.

        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            flow1: Flow from ref frame 1 (optional)
            flow2: Flow from ref frame 2 (optional)
            logit_occ1: Occlusion map 1 (optional)
            logit_occ2: Occlusion map 2 (optional)
            warped1: Warped frame 1 (optional)
            warped2: Warped frame 2 (optional)

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}

        # Reconstruction losses
        losses['l1'] = F.l1_loss(pred, target)
        losses['lap'] = self.lap_loss(pred, target)
        losses['char'] = self.char_loss(pred, target)
        if self.mixed_precision:
            with torch.autocast('cuda', enabled=False):
                pred_f32 = pred.float()
                target_f32 = target.float()
                losses['lpips'] = self.lpips_loss(pred_f32 * 2 - 1, target_f32 * 2 - 1).mean()
        else:
            # LPIPS expects [-1, 1] range
            losses['lpips'] = self.lpips_loss(pred * 2 - 1, target * 2 - 1).mean()

        # Flow smoothness
        if flow1 is not None and flow2 is not None:
            losses['flow_smooth'] = (self.flow_smooth_loss(flow1) + self.flow_smooth_loss(flow2)) / 2.0
        else:
            losses['flow_smooth'] = torch.tensor(0.0, device=pred.device)

        # Occlusion loss
        if (logit_occ1 is not None and logit_occ2 is not None and warped1 is not None and warped2 is not None):
            losses['occlusion'] = self.occ_loss(logit_occ1, logit_occ2, warped1, warped2, target)
        else:
            losses['occlusion'] = torch.tensor(0.0, device=pred.device)

        # Total weighted loss
        losses['total'] = (
            self.w_l1 * losses['l1'] +
            self.w_lap * losses['lap'] +
            self.w_flow * losses['flow_smooth'] +
            self.w_occ * losses['occlusion'] +
            self.w_perc * losses['lpips']
        )

        return losses


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test tensors
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    flow = torch.randn(2, 2, 256, 256).to(device)

    # Test Laplacian loss
    lap_loss = LaplacianPyramidLoss()
    lap_loss.to(device)
    loss = lap_loss(pred, target)
    print(f"Laplacian loss: {loss.item():.4f}")

    # Test flow smoothness
    smooth_loss = FlowSmoothnessLoss()
    loss = smooth_loss(flow)
    print(f"Flow smoothness loss: {loss.item():.4f}")

    # Test Charbonnier loss
    char_loss = CharbonnierLoss()
    loss = char_loss(pred, target)
    print(f"Charbonnier loss: {loss.item():.4f}")