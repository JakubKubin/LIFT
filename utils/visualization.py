import torch
import numpy as np

def flow_to_color(flow, max_flow=None):
    """Convert optical flow to RGB color wheel visualization."""
    B, _, H, W = flow.shape
    u, v = flow[:, 0], flow[:, 1]

    if max_flow is None:
        max_flow = torch.max(torch.sqrt(u**2 + v**2))

    mag = torch.sqrt(u**2 + v**2)
    ang = torch.atan2(v, u)

    hsv_h = (ang + np.pi) / (2 * np.pi)
    hsv_s = torch.clamp(mag / (max_flow + 1e-8), 0, 1)
    hsv_v = torch.ones_like(hsv_h)

    hi = (hsv_h * 6).long() % 6
    f = hsv_h * 6 - hi.float()
    p = hsv_v * (1 - hsv_s)
    q = hsv_v * (1 - f * hsv_s)
    t = hsv_v * (1 - (1 - f) * hsv_s)

    rgb = torch.zeros(B, 3, H, W, device=flow.device)

    for i in range(6):
        mask = (hi == i).unsqueeze(1)
        if i == 0:   rgb_i = torch.stack([hsv_v, t, p], dim=1)
        elif i == 1: rgb_i = torch.stack([q, hsv_v, p], dim=1)
        elif i == 2: rgb_i = torch.stack([p, hsv_v, t], dim=1)
        elif i == 3: rgb_i = torch.stack([p, q, hsv_v], dim=1)
        elif i == 4: rgb_i = torch.stack([t, p, hsv_v], dim=1)
        else:        rgb_i = torch.stack([hsv_v, p, q], dim=1)
        rgb = torch.where(mask, rgb_i, rgb)

    return rgb

def make_grid_with_labels(images_dict, nrow=4):
    """Create grid from dict of named images."""
    from torchvision.utils import make_grid
    grids = []
    for name, img in images_dict.items():
        if img.dim() == 3:
            img = img.unsqueeze(0)
        grids.append(make_grid(img[:nrow], nrow=nrow, normalize=True, padding=2))
    return grids, list(images_dict.keys())