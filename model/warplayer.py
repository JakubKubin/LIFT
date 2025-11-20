"""
Warping utilities for optical flow-based image warping.

This module provides efficient backward warping using bilinear sampling.
"""

import torch
import torch.nn.functional as F

# Cache for grid tensors to avoid recreation
_grid_cache = {}


def get_grid(flow, device):
    """
    Create sampling grid for backward warping.

    Args:
        flow: Flow tensor [B, 2, H, W]
        device: Device to create grid on

    Returns:
        Grid tensor [B, H, W, 2] normalized to [-1, 1]
    """
    key = (str(device), str(flow.shape))

    if key not in _grid_cache:
        B, _, H, W = flow.shape

        # Create normalized coordinate grid
        # X coordinates: -1 (left) to 1 (right)
        x_coords = torch.linspace(-1.0, 1.0, W, device=device)
        x_coords = x_coords.view(1, 1, 1, W).expand(B, -1, H, -1)

        # Y coordinates: -1 (top) to 1 (bottom)
        y_coords = torch.linspace(-1.0, 1.0, H, device=device)
        y_coords = y_coords.view(1, 1, H, 1).expand(B, -1, -1, W)

        # Concatenate to form grid [B, 2, H, W]
        grid = torch.cat([x_coords, y_coords], dim=1)
        _grid_cache[key] = grid

    return _grid_cache[key]


def backward_warp(image, flow):
    """
    Backward warp image using optical flow.

    This performs backward warping: for each pixel in the output,
    we sample from the input image at the location specified by the flow.

    Args:
        image: Input image tensor [B, C, H, W]
        flow: Optical flow tensor [B, 2, H, W]
              flow[:, 0] is horizontal displacement (x)
              flow[:, 1] is vertical displacement (y)

    Returns:
        Warped image tensor [B, C, H, W]

    Memory optimization:
    - Grid is cached to avoid recreation
    - Uses in-place operations where possible
    """
    B, C, H, W = image.shape
    device = image.device

    # Get base grid
    base_grid = get_grid(flow, device)

    # Normalize flow to [-1, 1] range for grid_sample
    # Flow is in pixel coordinates, need to convert to normalized coordinates
    flow_norm = torch.empty_like(flow)
    flow_norm[:, 0] = flow[:, 0] / ((W - 1.0) / 2.0)  # Normalize x
    flow_norm[:, 1] = flow[:, 1] / ((H - 1.0) / 2.0)  # Normalize y

    # Add flow to base grid to get sampling locations
    # [B, 2, H, W] + [B, 2, H, W] -> [B, 2, H, W]
    sampling_grid = base_grid + flow_norm

    # Permute to [B, H, W, 2] for grid_sample
    sampling_grid = sampling_grid.permute(0, 2, 3, 1)

    # Perform bilinear sampling
    # mode='bilinear': bilinear interpolation
    # padding_mode='border': use border values for out-of-bounds locations
    # align_corners=True: align corner pixels
    warped = F.grid_sample(
        image,
        sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return warped


def clear_cache():
    """Clear the grid cache. Useful when changing resolution."""
    global _grid_cache
    _grid_cache.clear()


if __name__ == '__main__':
    # Test warping
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test image
    img = torch.rand(2, 3, 256, 256).to(device)

    # Create random flow
    flow = torch.randn(2, 2, 256, 256).to(device) * 10  # Random displacement

    # Warp
    warped = backward_warp(img, flow)

    print(f"Input shape: {img.shape}")
    print(f"Flow shape: {flow.shape}")
    print(f"Output shape: {warped.shape}")
    print(f"Output range: [{warped.min():.3f}, {warped.max():.3f}]")
