"""
Stage 4: Coarse Frame Synthesis

Synthesizes coarse interpolated frame at s4 resolution using:
1. Backward warping of reference frames with predicted flows
2. Occlusion-aware blending
3. Context injection from 15-frame temporal aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import backward_warp


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Basic convolutional block with ReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        nn.ReLU(inplace=True)
    )


class ContextInjectionNet(nn.Module):
    """
    Lightweight network for injecting 15-frame temporal context.

    Takes blended coarse frame and temporal context, outputs residual correction.
    Architecture: 2 convolutional layers as specified.
    """

    def __init__(self, context_dim=256, hidden_dim=15):
        super().__init__()

        # Input: 3 (RGB) + context_dim channels
        self.conv1 = conv_block(3 + context_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, blended_frame, context):
        """
        Inject context into blended frame.

        Args:
            blended_frame: Blended coarse frame [B, 3, H, W]
            context: Temporal context [B, C, H, W]

        Returns:
            Residual correction [B, 3, H, W]
        """
        # Concatenate blended frame and context
        x = torch.cat([blended_frame, context], dim=1)

        # Process through network
        x = self.conv1(x)
        residual = self.conv2(x)

        return residual


class CoarseSynthesis(nn.Module):
    """
    Complete coarse frame synthesis module.

    Pipeline:
    1. Downsample reference frames to s4 resolution
    2. Backward warp using predicted flows
    3. Blend warped frames using occlusion maps
    4. Inject temporal context
    5. Output coarse interpolated frame at s4
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.context_net = ContextInjectionNet(
            context_dim=config.transformer_dim,
            hidden_dim=config.context_net_channels
        )

    def forward(self, ref_frames, flow_output, context):
        """
        Synthesize coarse interpolated frame.

        Args:
            ref_frames: Reference frames [B, 2, 3, H, W]
            flow_output: Dictionary from FlowEstimator with:
                - 'flow_31': Flow from frame 31 [B, 2, H/4, W/4]
                - 'flow_32': Flow from frame 32 [B, 2, H/4, W/4]
                - 'occ_31': Occlusion map for frame 31 [B, 1, H/4, W/4]
                - 'occ_32': Occlusion map for frame 32 [B, 1, H/4, W/4]
            context: Temporal context from transformer [B, 256, H/16, W/16]

        Returns:
            Dictionary with:
                - 'coarse_frame': Coarse interpolated frame [B, 3, H/4, W/4]
                - 'blended_frame': Frame before context injection [B, 3, H/4, W/4]
                - 'warped_31': Warped frame 31 [B, 3, H/4, W/4]
                - 'warped_32': Warped frame 32 [B, 3, H/4, W/4]
        """
        # Extract reference frames
        I_31 = ref_frames[:, 0]  # [B, 3, H, W]
        I_32 = ref_frames[:, 1]  # [B, 3, H, W]

        # Extract flows and occlusion maps
        flow_31 = flow_output['flow_31']  # [B, 2, H/4, W/4]
        flow_32 = flow_output['flow_32']  # [B, 2, H/4, W/4]
        occ_31 = flow_output['occ_31']    # [B, 1, H/4, W/4]
        occ_32 = flow_output['occ_32']    # [B, 1, H/4, W/4]

        # Step 4.1: Downsample reference frames to s4
        H_s4, W_s4 = flow_31.shape[2], flow_31.shape[3]
        I_31_s4 = F.interpolate(I_31, size=(H_s4, W_s4), mode='bilinear', align_corners=False)
        I_32_s4 = F.interpolate(I_32, size=(H_s4, W_s4), mode='bilinear', align_corners=False)

        # Step 4.2: Backward warp using predicted flows
        warped_31 = backward_warp(I_31_s4, flow_31)
        warped_32 = backward_warp(I_32_s4, flow_32)

        # Step 4.3: Occlusion-aware blending
        # Numerator: weighted sum of warped frames
        numerator = occ_31 * warped_31 + occ_32 * warped_32

        # Denominator: sum of weights (with epsilon for numerical stability)
        epsilon = 1e-6
        denominator = occ_31 + occ_32 + epsilon

        # Blended frame
        blended_frame = numerator / denominator

        # Step 4.4: Context injection
        # Upsample context to s4 resolution
        context_s4 = F.interpolate(context, size=(H_s4, W_s4), mode='bilinear', align_corners=False)

        # Get residual correction from context network
        residual = self.context_net(blended_frame, context_s4)

        # Step 4.5: Add residual to get final coarse frame
        coarse_frame = blended_frame + residual

        # Clamp to valid range [0, 1]
        coarse_frame = torch.clamp(coarse_frame, 0.0, 1.0)

        return {
            'coarse_frame': coarse_frame,
            'blended_frame': blended_frame,
            'warped_31': warped_31,
            'warped_32': warped_32,
            'residual': residual
        }


if __name__ == '__main__':
    # Test coarse synthesis
    import sys
    sys.path.append('..')
    from configs.default import Config

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test inputs
    B, H, W = 2, 256, 256
    ref_frames = torch.rand(B, 2, 3, H, W).to(device)

    # Simulate flow estimator output
    # Sizes based on s4 resolution (H/4, W/4)
    flow_output = {
        'flow_31': torch.randn(B, 2, H // 4, W // 4).to(device) * 5,
        'flow_32': torch.randn(B, 2, H // 4, W // 4).to(device) * 5,
        'occ_31': torch.rand(B, 1, H // 4, W // 4).to(device),
        'occ_32': torch.rand(B, 1, H // 4, W // 4).to(device),
    }

    # Context is at s16 (H/16, W/16)
    context = torch.rand(B, 256, H // 16, W // 16).to(device)

    # Create synthesis module
    synthesis = CoarseSynthesis(config).to(device)

    # Forward pass
    with torch.no_grad():
        output = synthesis(ref_frames, flow_output, context)

    print("Coarse Synthesis outputs:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
            if 'frame' in key:
                print(f"    Range: [{value.min():.4f}, {value.max():.4f}]")

    # Verify output is in valid range
    assert output['coarse_frame'].min() >= 0.0 and output['coarse_frame'].max() <= 1.0, \
        "Coarse frame out of valid range!"

    print("\nCoarse frame statistics:")
    print(f"  Mean: {output['coarse_frame'].mean():.4f}")
    print(f"  Std: {output['coarse_frame'].std():.4f}")

    # Check memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")