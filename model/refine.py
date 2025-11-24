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

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)

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
    2. Reduce channels of reference features (128 -> 32)
    3. Upsample reduced features to full resolution
    4. Concatenate all inputs (3 + 32 + 32 = 67 channels)
    5. Process through lightweight refinement network
    6. Output residual correction
    7. Add to upsampled coarse for final output
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Channel reduction for memory efficiency
        # Input: 128 channels -> Output: 32 channels
        self.reduce_feat_7 = nn.Conv2d(
            config.encoder_channels['s4'],
            config.refine_reduce_channels,
            kernel_size=1,
            bias=False
        )
        self.reduce_feat_9 = nn.Conv2d(
            config.encoder_channels['s4'],
            config.refine_reduce_channels,
            kernel_size=1,
            bias=False
        )

        # Refinement network
        # Input: 3 (RGB) + 32 (feat_7) + 32 (feat_9) = 67 channels
        input_channels = 3 + config.refine_reduce_channels * 2
        hidden_channels = config.refine_channels[0]  # 15

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

        # Output layer (no activation - this is a residual)
        self.conv_out = nn.Conv2d(config.refine_channels[1], 3, kernel_size=3, padding=1, bias=True)

    def forward(self, coarse_frame, ref_feats_s4):
        """
        Refine coarse frame to full resolution.

        Args:
            coarse_frame: Coarse interpolated frame [B, 3, H/4, W/4]
            ref_feats_s4: Reference frame features at s4 [B, 2, 128, H/4, W/4]

        Returns:
            Dictionary with:
                - 'final_frame': Final high-quality interpolated frame [B, 3, H, W]
                - 'upsampled_coarse': Upsampled coarse frame [B, 3, H, W]
                - 'residual': Residual correction [B, 3, H, W]
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

        # Step 5.2: Reduce channels of reference features
        # Extract individual reference frame features
        feat_7_s4 = ref_feats_s4[:, 0]  # [B, 128, H/4, W/4]
        feat_9_s4 = ref_feats_s4[:, 1]  # [B, 128, H/4, W/4]

        # Apply channel reduction (128 -> 32)
        feat_7_reduced = self.reduce_feat_7(feat_7_s4)  # [B, 32, H/4, W/4]
        feat_9_reduced = self.reduce_feat_9(feat_9_s4)  # [B, 32, H/4, W/4]

        # Step 5.3: Upsample reduced features to full resolution
        feat_7_full = F.interpolate(
            feat_7_reduced,
            size=(H_full, W_full),
            mode='bilinear',
            align_corners=False
        )
        feat_9_full = F.interpolate(
            feat_9_reduced,
            size=(H_full, W_full),
            mode='bilinear',
            align_corners=False
        )

        # Step 5.4: Concatenate all inputs
        refine_input = torch.cat([upsampled_coarse, feat_7_full, feat_9_full], dim=1)
        # [B, 67, H, W]

        # Step 5.5: Process through refinement network
        x = self.conv_init(refine_input)
        x = self.res_blocks(x)
        x = self.conv_mid(x)
        residual = self.conv_out(x)

        # Step 5.6: Add residual to upsampled coarse
        final_frame = upsampled_coarse + residual

        # Clamp to valid range [0, 1]
        final_frame = torch.clamp(final_frame, 0.0, 1.0)

        return {
            'final_frame': final_frame,
            'upsampled_coarse': upsampled_coarse,
            'residual': residual
        }


if __name__ == '__main__':
    # Test full resolution refinement
    import sys
    sys.path.append('..')
    from configs.default import Config

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test inputs
    B, H, W = 2, 256, 256
    # Coarse frame is at 1/4 resolution (s4)
    coarse_frame = torch.rand(B, 3, H // 4, W // 4).to(device)
    # Reference features are also at 1/4 resolution (s4)
    # Using config to get correct channel count
    c_s4 = config.encoder_channels['s4']
    ref_feats_s4 = torch.rand(B, 2, c_s4, H // 4, W // 4).to(device)

    # Create refinement module
    refinement = FullResolutionRefinement(config).to(device)

    # Forward pass
    with torch.no_grad():
        output = refinement(coarse_frame, ref_feats_s4)

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

    # Check memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")