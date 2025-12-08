"""
Stage 5: Full Resolution Refinement

Refines coarse interpolated frame to full resolution with high-quality details.

Memory optimization:
- Channel reduction (128 -> 32) for reference features
- Lightweight ResBlock architecture
- Efficient upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm for refinement.

    Architecture:
    - Conv -> GroupNorm -> ReLU
    - Conv -> GroupNorm
    - Residual connection
    """

    def __init__(self, channels, num_groups=8):
        super().__init__()

        # Determine number of groups for GroupNorm
        groups = min(num_groups, channels) if channels > 0 else 1

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class FullResolutionRefinement(nn.Module):
    """
    Complete full-resolution refinement module.

    Pipeline:
    1. Upsample coarse frame to full resolution
    2. Extract s1 features (already full resolution)
    3. Concatenate all inputs
    4. Process through lightweight refinement network
    5. Add residual correction to upsampled coarse frame
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input:
        # - Upsampled Coarse Frame (3 channels)
        # - Feature Ref 1 (s1, N channels)
        # - Feature Ref 2 (s1, N channels)

        s1_channels = config.encoder_channels['s1']
        input_channels = 3 + s1_channels * 2

        hidden_channels = config.refine_channels[0]

        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_channels, num_groups=8),
            ResBlock(hidden_channels, num_groups=8)
        )

        # Middle layer
        self.conv_mid = nn.Sequential(
            nn.Conv2d(hidden_channels, config.refine_channels[1], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.refine_channels[1]),
            nn.ReLU(inplace=True)
        )

        # Output layer (residual)
        self.conv_out = nn.Conv2d(config.refine_channels[1], 3, kernel_size=3, padding=1, bias=True)

    def forward(self, coarse_frame, ref_feats_s1):
        """
        Refine coarse frame using full resolution s1 features.

        Args:
            coarse_frame: [B, 3, H/4, W/4]
            ref_feats_s1: [B, 2, C_s1, H, W] (Already Full Res)
        """
        B, _, H_s4, W_s4 = coarse_frame.shape

        # Step 5.1: Upsample coarse frame to full resolution
        H_full = H_s4 * 4
        W_full = W_s4 * 4

        upsampled_coarse = F.interpolate(
            coarse_frame,
            size=(H_full, W_full),
            mode='bilinear',
            align_corners=False
        )

        # Step 5.2: Extract s1 features directly (no upsampling needed)
        feat_7_s1 = ref_feats_s1[:, 0]  # [B, C_s1, H, W]
        feat_9_s1 = ref_feats_s1[:, 1]  # [B, C_s1, H, W]

        # Step 5.3: Concatenate all inputs
        refine_input = torch.cat([upsampled_coarse, feat_7_s1, feat_9_s1], dim=1)

        # Step 5.4: Process through refinement network
        x = self.conv_init(refine_input)
        x = self.res_blocks(x)
        x = self.conv_mid(x)
        residual = self.conv_out(x)

        # Step 5.5: Add residual
        final_frame = upsampled_coarse + residual
        final_frame = torch.clamp(final_frame, 0.0, 1.0)

        return {
            'final_frame': final_frame,
            'upsampled_coarse': upsampled_coarse,
            'residual': residual
        }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from configs.default import Config

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test inputs
    B, H, W = 2, 256, 256
    # Coarse frame is at 1/4 resolution (s4)
    coarse_frame = torch.rand(B, 3, H // 4, W // 4).to(device)

    c_s1 = config.encoder_channels['s1']
    ref_feats_s1 = torch.rand(B, 2, c_s1, H, W).to(device)

    # Create refinement module
    refinement = FullResolutionRefinement(config).to(device)

    # Forward pass
    with torch.no_grad():
        output = refinement(coarse_frame, ref_feats_s1)

    print("Full Resolution Refinement outputs:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
            if 'frame' in key:
                print(f"    Range: [{value.min():.4f}, {value.max():.4f}]")

    # Verify output shape
    assert output['final_frame'].shape == (B, 3, H, W), \
        f"Expected shape ({B}, 3, {H}, {W}), got {output['final_frame'].shape}"

    # Verify output is in valid range
    assert output['final_frame'].min() >= 0.0 and output['final_frame'].max() <= 1.0, \
        "Final frame out of valid range!"

    print("\nFinal frame statistics:")
    print(f"  Mean: {output['final_frame'].mean():.4f}")
    print(f"  Std: {output['final_frame'].std():.4f}")

    # Compare with coarse frame
    print("\nImprovement analysis:")
    print(f"  Coarse mean: {output['upsampled_coarse'].mean():.4f}")
    print(f"  Final mean: {output['final_frame'].mean():.4f}")
    print(f"  Residual mean: {output['residual'].mean():.4f}")
    print(f"  Residual std: {output['residual'].std():.4f}")
