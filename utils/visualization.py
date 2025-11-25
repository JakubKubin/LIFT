"""
Visualization utilities for LIFT model TensorBoard logging.

Provides:
- Optical flow colorization
- Attention weight visualization
- Side-by-side image comparisons
- Gradient flow visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Optional, List, Tuple


def flow_to_color(flow: torch.Tensor, max_flow: Optional[float] = None) -> torch.Tensor:
    """
    Convert optical flow to RGB color wheel visualization.

    Args:
        flow: Optical flow tensor [B, 2, H, W]
        max_flow: Maximum flow magnitude for normalization (auto if None)

    Returns:
        RGB visualization [B, 3, H, W] in range [0, 1]
    """
    B, _, H, W = flow.shape
    device = flow.device

    u = flow[:, 0]
    v = flow[:, 1]

    mag = torch.sqrt(u**2 + v**2)
    if max_flow is None:
        max_flow = mag.max().item() + 1e-8

    ang = torch.atan2(v, u)

    # Use torch.pi to ensure tensor operations
    pi = torch.tensor(np.pi, device=device)
    hsv_h = (ang + pi) / (2 * pi)
    hsv_s = torch.clamp(mag / max_flow, 0, 1)
    hsv_v = torch.ones_like(hsv_h)

    hi = (hsv_h * 6).long() % 6
    f = hsv_h * 6 - hi.float()
    p = hsv_v * (1 - hsv_s)
    q = hsv_v * (1 - f * hsv_s)
    t = hsv_v * (1 - (1 - f) * hsv_s)

    rgb = torch.zeros(B, 3, H, W, device=device)

    for i in range(6):
        mask = (hi == i).unsqueeze(1).expand(-1, 3, -1, -1)
        if i == 0:
            rgb_i = torch.stack([hsv_v, t, p], dim=1)
        elif i == 1:
            rgb_i = torch.stack([q, hsv_v, p], dim=1)
        elif i == 2:
            rgb_i = torch.stack([p, hsv_v, t], dim=1)
        elif i == 3:
            rgb_i = torch.stack([p, q, hsv_v], dim=1)
        elif i == 4:
            rgb_i = torch.stack([t, p, hsv_v], dim=1)
        else:
            rgb_i = torch.stack([hsv_v, p, q], dim=1)
        rgb = torch.where(mask, rgb_i, rgb)

    return rgb


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


def plot_attention_weights(
    weights: torch.Tensor,
    num_frames: int = 15,
    gap_idx: int = 8,
    title: str = "Temporal Attention Weights"
) -> Figure:
    """
    Create bar plot of temporal attention weights.

    Args:
        weights: Attention weights [T] or [B, T] (uses first sample)
        num_frames: Total number of frame positions
        gap_idx: Index of missing frame (GT)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if weights.dim() > 1:
        weights = weights[0]

    # Use a separate variable for numpy array to avoid type conflicts
    weights_np = weights.detach().cpu().numpy()

    x_labels = [i for i in range(num_frames) if i != gap_idx]

    fig, ax = plt.subplots(figsize=(12, 4))

    colors = ['#2ecc71' if i in [gap_idx-1, gap_idx+1] else '#3498db' 
              for i in x_labels]

    bars = ax.bar(range(len(weights_np)), weights_np, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(weights_np)))
    ax.set_xticklabels([str(i) for i in x_labels])
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Attention Weight (α)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.axhline(y=1.0/len(weights_np), color='red', linestyle='--', 
               label=f'Uniform: {1.0/len(weights_np):.3f}', alpha=0.7)

    max_idx = np.argmax(weights_np)
    # Cast to python primitives for matplotlib compatibility
    ax.annotate(f'{weights_np[max_idx]:.3f}', 
                xy=(float(max_idx), float(weights_np[max_idx])),
                xytext=(float(max_idx), float(weights_np[max_idx]) + 0.02),
                ha='center', fontsize=10, fontweight='bold')

    ax.legend(loc='upper right')
    # Cast ylim to float
    ax.set_ylim(0, float(max(weights_np.max() * 1.2, 0.2)))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_components(
    losses: Dict[str, float],
    title: str = "Loss Components"
) -> Figure:
    """
    Create pie chart of loss components.

    Args:
        losses: Dictionary of loss names and values
        title: Plot title

    Returns:
        Matplotlib figure
    """
    filtered = {k: v for k, v in losses.items() 
                if k != 'total' and v > 0}

    if not filtered:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No loss components', ha='center', va='center')
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(filtered.keys())
    values = list(filtered.values())

    # Use plt.get_cmap to avoid static analysis errors with plt.cm.Set3
    cmap = plt.get_cmap('Set3')
    colors = cmap(np.linspace(0, 1, len(names)))

    wedges, texts, autotexts = ax1.pie(
        values, labels=names, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    ax1.set_title(f'{title} (Proportions)', fontsize=12, fontweight='bold')

    bars = ax2.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Loss Value', fontsize=11)
    ax2.set_title(f'{title} (Values)', fontsize=12, fontweight='bold')

    for bar, val in zip(bars, values):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    ax2.set_xlim(0, max(values) * 1.3)

    plt.tight_layout()
    return fig


def plot_flow_histogram(
    flow: torch.Tensor,
    title: str = "Flow Magnitude Distribution"
) -> Figure:
    """
    Create histogram of flow magnitudes.

    Args:
        flow: Optical flow tensor [B, 2, H, W]
        title: Plot title

    Returns:
        Matplotlib figure
    """
    mag = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
    mag_np = mag.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(mag_np, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    # Cast to python float
    ax.axvline(float(mag_np.mean()), color='red', linestyle='--', 
               label=f'Mean: {mag_np.mean():.2f}')
    ax.axvline(float(np.median(mag_np)), color='green', linestyle='--',
               label=f'Median: {np.median(mag_np):.2f}')

    ax.set_xlabel('Flow Magnitude (pixels)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress"
) -> Figure:
    """
    Plot training curves from history.

    Args:
        history: Dictionary with lists of values per metric
        title: Plot title

    Returns:
        Matplotlib figure
    """
    n_metrics = len(history)
    if n_metrics == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (name, values) in enumerate(history.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.plot(values, linewidth=1.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Step')
        ax.grid(alpha=0.3)

    for idx in range(len(history), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_occlusion_maps(
    occ1: torch.Tensor,
    occ2: torch.Tensor,
    title: str = "Occlusion Maps"
) -> Figure:
    """
    Visualize occlusion maps with colorbar.

    Args:
        occ1: Occlusion map 1 [B, 1, H, W]
        occ2: Occlusion map 2 [B, 1, H, W]
        title: Plot title

    Returns:
        Matplotlib figure
    """
    occ1_np = occ1[0, 0].detach().cpu().numpy()
    occ2_np = occ2[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(occ1_np, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Occlusion I₇→I₈', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(occ2_np, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Occlusion I₉→I₈', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    combined = occ1_np - occ2_np
    im3 = axes[2].imshow(combined, cmap='RdBu', vmin=-1, vmax=1)
    axes[2].set_title('Difference (O₇ - O₉)', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def compute_gradient_stats(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Compute gradient statistics for each module.

    Args:
        model: PyTorch model with gradients

    Returns:
        Dictionary with gradient stats per module
    """
    stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            module_name = name.rsplit('.', 1)[0] if '.' in name else name

            if module_name not in stats:
                stats[module_name] = {
                    'mean': [],
                    'std': [],
                    'max': [],
                    'norm': []
                }

            stats[module_name]['mean'].append(grad.mean().item())
            stats[module_name]['std'].append(grad.std().item())
            stats[module_name]['max'].append(grad.abs().max().item())
            stats[module_name]['norm'].append(grad.norm().item())

    aggregated = {}
    for module, values in stats.items():
        aggregated[module] = {
            'mean': np.mean(values['mean']),
            'std': np.mean(values['std']),
            'max': np.max(values['max']),
            'norm': np.mean(values['norm'])
        }

    return aggregated


def create_error_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    amplify: float = 5.0
) -> torch.Tensor:
    """
    Create amplified error visualization.

    Args:
        pred: Predicted image [B, 3, H, W]
        target: Target image [B, 3, H, W]
        amplify: Error amplification factor

    Returns:
        Error map [B, 3, H, W]
    """
    error = torch.abs(pred - target)
    error_gray = error.mean(dim=1, keepdim=True)
    error_amplified = (error_gray * amplify).clamp(0, 1)
    error_colored = torch.cat([
        error_amplified,
        torch.zeros_like(error_amplified),
        1 - error_amplified
    ], dim=1)

    return error_colored


if __name__ == '__main__':
    print("Testing visualization utilities...")

    flow = torch.randn(2, 2, 64, 64) * 10
    flow_vis = flow_to_color(flow)
    print(f"Flow visualization shape: {flow_vis.shape}")

    weights = torch.softmax(torch.randn(14), dim=0)
    fig = plot_attention_weights(weights)
    plt.savefig('/tmp/test_attention.png')
    plt.close(fig)
    print("Attention plot saved")

    losses = {'l1': 0.05, 'lap': 0.02, 'lpips': 0.01, 'flow_smooth': 0.005}
    fig = plot_loss_components(losses)
    plt.savefig('/tmp/test_losses.png')
    plt.close(fig)
    print("Loss plot saved")

    print("All visualization tests passed!")