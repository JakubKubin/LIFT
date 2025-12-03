import torch
import torch.nn.functional as F
import numpy as np
from math import exp
import lpips

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    """
    Oblicza SSIM (Structural Similarity Index).
    img1, img2: [B, 3, H, W], zakres [0, 1]
    """
    padd = window_size // 2

    if window is None:
        real_size = min(window_size, img1.size(2), img1.size(3))
        window = create_window(real_size, img1.size(1)).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=padd, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=padd, groups=img1.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=img1.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Evaluator:
    """
    Klasa pomocnicza do obliczania metryk jakości wideo.
    """
    def __init__(self, device):
        self.device = device
        # Inicjalizacja LPIPS (VGG) - tryb ewaluacji, bez gradientów
        try:
            self.lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: 'lpips' library not found. LPIPS metric will be 0.")
            self.lpips_model = None

        self.ssim_window = None

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Oblicza PSNR, SSIM i LPIPS.

        Args:
            pred: Predykcja [B, 3, H, W] w zakresie [0, 1]
            target: Ground Truth [B, 3, H, W] w zakresie [0, 1]
        """
        pred = pred.float().clamp(0, 1)
        target = target.float().clamp(0, 1)

        # 1. PSNR
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = -10 * torch.log10(mse).item()

        # 2. SSIM
        if self.ssim_window is None or self.ssim_window.device != pred.device:
            real_size = min(11, pred.size(2), pred.size(3))
            self.ssim_window = create_window(real_size, pred.size(1)).to(self.device).type_as(pred)

        ssim_val = ssim(pred, target, window=self.ssim_window).item()

        # 3. LPIPS
        lpips_val = 0.0
        if self.lpips_model is not None:
            # LPIPS wymaga wejścia [-1, 1]
            with torch.no_grad():
                lpips_val = self.lpips_model(pred * 2 - 1, target * 2 - 1).mean().item()

        return {
            'PSNR': psnr,
            'SSIM': ssim_val,
            'LPIPS': lpips_val
        }