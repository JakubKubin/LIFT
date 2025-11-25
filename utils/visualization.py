"""Visualization utilities for LIFT TensorBoard logging."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict

def flow_to_color(flow: torch.Tensor, max_flow: Optional[float] = None) -> torch.Tensor:
    """Convert optical flow [B,2,H,W] to RGB color wheel [B,3,H,W]."""
    B, _, H, W = flow.shape
    device = flow.device

    u, v = flow[:, 0], flow[:, 1]
    mag = torch.sqrt(u**2 + v**2)
    if max_flow is None:
        max_flow_val = mag.max().item() + 1e-8
    else:
        max_flow_val = max_flow

    # HSV to RGB conversion
    h = (torch.atan2(v, u) + np.pi) / (2 * np.pi)  # Hue from angle
    s = torch.clamp(mag / max_flow_val, 0, 1)      # Saturation from magnitude
    v_val = torch.ones_like(h)                     # Value = 1

    # Simplified HSV->RGB
    h6 = h * 6
    i = h6.long() % 6
    f = h6 - i.float()
    p = v_val * (1 - s)
    q = v_val * (1 - f * s)
    t = v_val * (1 - (1 - f) * s)

    # Unrolled sector assignment to avoid Pylance ambiguity with loop/where
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    # Sector 0
    mask = (i == 0)
    r = torch.where(mask, v_val, r)
    g = torch.where(mask, t, g)
    b = torch.where(mask, p, b)

    # Sector 1
    mask = (i == 1)
    r = torch.where(mask, q, r)
    g = torch.where(mask, v_val, g)
    b = torch.where(mask, p, b)

    # Sector 2
    mask = (i == 2)
    r = torch.where(mask, p, r)
    g = torch.where(mask, v_val, g)
    b = torch.where(mask, t, b)

    # Sector 3
    mask = (i == 3)
    r = torch.where(mask, p, r)
    g = torch.where(mask, q, g)
    b = torch.where(mask, v_val, b)

    # Sector 4
    mask = (i == 4)
    r = torch.where(mask, t, r)
    g = torch.where(mask, p, g)
    b = torch.where(mask, v_val, b)

    # Sector 5
    mask = (i == 5)
    r = torch.where(mask, v_val, r)
    g = torch.where(mask, p, g)
    b = torch.where(mask, q, b)

    rgb = torch.stack([r, g, b], dim=1)
    return rgb.clamp(0, 1)


def create_error_map(pred: torch.Tensor, target: torch.Tensor, amplify: float = 5.0) -> torch.Tensor:
    """Create amplified error visualization [B,3,H,W]."""
    error = torch.abs(pred - target).mean(dim=1, keepdim=True) * amplify
    error = error.clamp(0, 1)
    return torch.cat([error, torch.zeros_like(error), 1 - error], dim=1)


def plot_attention_weights(weights: torch.Tensor, num_frames: int = 15, gap_idx: int = 8) -> Figure:
    """Create bar plot of temporal attention weights."""
    if weights.dim() > 1:
        weights = weights[0]
    w = weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 3))
    x_labels = [i for i in range(num_frames) if i != gap_idx]
    colors = ['#2ecc71' if i in [gap_idx-1, gap_idx+1] else '#3498db' for i in x_labels]

    ax.bar(range(len(w)), w, color=colors)
    ax.axhline(1/len(w), color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels([str(x) for x in x_labels])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Weight')
    plt.tight_layout()
    return fig


def plot_loss_components(losses: dict) -> Figure:
    """Create bar chart of loss components."""
    filtered = {k: v for k, v in losses.items() if k != 'total' and v > 0}

    fig, ax = plt.subplots(figsize=(8, 4))
    if filtered:
        ax.barh(list(filtered.keys()), list(filtered.values()), color='#3498db')
        ax.set_xlabel('Loss Value')
    plt.tight_layout()
    return fig


def visualize_occlusion_maps(occ1: torch.Tensor, occ2: torch.Tensor) -> Figure:
    """Visualize occlusion maps side by side."""
    o1 = occ1[0, 0].detach().cpu().numpy()
    o2 = occ2[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(o1, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Occ I_7->I_8')
    axes[0].axis('off')

    axes[1].imshow(o2, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Occ I_9->I_8')
    axes[1].axis('off')

    axes[2].imshow(o1 - o2, cmap='RdBu', vmin=-1, vmax=1)
    axes[2].set_title('Difference')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def compute_gradient_stats(model: torch.nn.Module) -> dict:
    """Compute gradient statistics per module."""
    stats = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            module = name.rsplit('.', 1)[0] if '.' in name else name
            if module not in stats:
                stats[module] = []
            stats[module].append(p.grad.norm().item())

    return {k: {'mean': float(np.mean(v)), 'max': float(np.max(v))} for k, v in stats.items()}


def plot_flow_histogram(flow: torch.Tensor, title: str = "Flow Magnitude") -> Figure:
    """Create histogram of flow magnitudes."""
    mag = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
    mag_np = mag.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(mag_np, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Magnitude')
    plt.tight_layout()
    return fig


def create_comparison_grid(
    images: Dict[str, torch.Tensor],
    nrow: int = 4,
    padding: int = 2
) -> torch.Tensor:
    """
    Create a comparison grid from multiple image tensors.

    Args:
        images: Dictionary of named images, each [B, C, H, W]
        nrow: Number of images per row
        padding: Padding between images

    Returns:
        Grid tensor [3, grid_H, grid_W]
    """
    from torchvision.utils import make_grid

    grids = []
    for name, img in images.items():
        if img is None:
            continue
        if img.dim() == 3:
            img = img.unsqueeze(0)

        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        elif img.shape[1] == 2:
            img = flow_to_color(img)

        img = img.clamp(0, 1)
        grid = make_grid(img[:nrow], nrow=nrow, normalize=False, padding=padding)
        grids.append(grid)

    if len(grids) == 0:
        return torch.zeros(3, 64, 64)

    final_grid = torch.cat(grids, dim=1)
    return final_grid