"""
Stage 3: Multi-scale Flow Estimation

Two-scale cascade (s8 -> s4) for bi-directional optical flow and occlusion prediction.
Adapted from RIFE's IFNet with modifications for LIFT:
- Accepts context from 15-frame transformer
- Predicts separate flows for each reference frame
- Keeps occlusion maps in logit space until final output
- Two-scale cascade instead of three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import backward_warp

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Basic convolutional block with PReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        nn.PReLU(out_channels)
    )


class ResidualBlock(nn.Module):
    """Residual block for flow estimation."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = self.prelu(out)
        return out


class FlowEstimationBlock(nn.Module):
    """
    Single-scale flow estimation block.

    Predicts:
    - flow_7: Flow from I_7 to I_t [B, 2, H, W]
    - flow_9: Flow from I_9 to I_t [B, 2, H, W]
    - logit_occ_7: Occlusion logits for frame 7 [B, 1, H, W]
    - logit_occ_9: Occlusion logits for frame 9 [B, 1, H, W]

    Total output: 6 channels (2+2+1+1)
    """

    def __init__(self, in_channels, hidden_channels=15, num_res_blocks=8):
        super().__init__()

        # Initial downsampling and feature extraction
        self.conv_init = nn.Sequential(
            conv_block(in_channels, hidden_channels // 2, 3, 2, 1),
            conv_block(hidden_channels // 2, hidden_channels, 3, 2, 1)
        )

        # Residual blocks for feature processing
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(hidden_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Output layer: upsample and predict flows + occlusion logits
        self.output_conv = nn.ConvTranspose2d(
            hidden_channels, 6,
            kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, prev_flow=None, prev_occ_logits=None, scale=1):
        """
        Forward pass for flow estimation.

        Args:
            x: Input features [B, C, H, W]
            prev_flow: Previous flow estimate (if refining) [B, 4, H, W]
            prev_occ_logits: Previous occlusion logits [B, 2, H, W]
            scale: Scale factor for current resolution

        Returns:
            flow: Predicted flows [B, 4, H, W] (flow_7 + flow_9)
            occ_logits: Occlusion logits [B, 2, H, W]
        """
        # Downsample input if scale != 1
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)

        # If refining, concatenate previous estimates
        if prev_flow is not None:
            # Downsample and scale previous flow
            prev_flow = F.interpolate(
                prev_flow, scale_factor=1.0 / scale,
                mode='bilinear', align_corners=False
            ) / scale
            x = torch.cat([x, prev_flow], dim=1)

        if prev_occ_logits is not None:
            # Downsample previous occlusion logits
            prev_occ_logits = F.interpolate(
                prev_occ_logits, scale_factor=1.0 / scale,
                mode='bilinear', align_corners=False
            )
            x = torch.cat([x, prev_occ_logits], dim=1)

        # Extract features
        x = self.conv_init(x)
        x = self.res_blocks(x) + x  # Residual connection

        # Predict output
        out = self.output_conv(x)

        # Upsample to target scale
        out = F.interpolate(out, scale_factor=scale * 2, mode='bilinear', align_corners=False)

        # Split into flows and occlusion logits
        flow = out[:, :4] * scale * 2  # Scale flow values appropriately
        occ_logits = out[:, 4:6]  # Keep as logits

        return flow, occ_logits


class FlowEstimator(nn.Module):
    """
    Complete flow estimation module with two-scale cascade.

    Architecture:
    1. Coarse estimation at scale s8 (1/8 resolution)
    2. Refinement at scale s4 (1/4 resolution)

    Inputs:
    - Reference frames I_7 and I_9
    - Their features from encoder
    - Context from 15-frame transformer
    - Timestep t

    Outputs:
    - Bi-directional flows: flow_7, flow_9
    - Occlusion maps: O_7, O_9 (after sigmoid)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # s8: Coarse input channels
        c_s8_input = config.encoder_channels['s8'] * 2 + config.transformer_dim + 1

        self.block_s8 = FlowEstimationBlock(
            in_channels=c_s8_input,
            hidden_channels=config.flow_channels[8],
            num_res_blocks=8
        )

        # s4: Refinement input channels (Zwiększone dla Warped Features)
        c_s4_feats = config.encoder_channels['s4']

        # Obliczamy kanały wejściowe:
        # Oryginalne (2*C) + Warpowane (2*C) + Kontekst + Flow + Occ + Czas
        c_s4_input = (
            c_s4_feats * 2 +       # feat_7, feat_9
            c_s4_feats * 2 +       # warped_feat_7, warped_feat_9 (DODANE)
            config.transformer_dim +  # context
            4 + 2 + 1              # prev_flow, prev_occ, timestep
        )

        self.block_s4 = FlowEstimationBlock(
            in_channels=c_s4_input,
            hidden_channels=config.flow_channels[4],
            num_res_blocks=6
        )

    def forward(self, ref_frames, ref_feats_s8, ref_feats_s4, context, timestep):
        """
        Estimate optical flows and occlusion maps.

        Args:
            ref_frames: Reference frames [B, 2, 3, H, W] (frames 31 and 32)
            ref_feats_s8: Features at s8 [B, 2, 192, H/8, W/8]
            ref_feats_s4: Features at s4 [B, 2, 128, H/4, W/4]
            context: Temporal context from transformer [B, 256, H/16, W/16]
            timestep: Interpolation timestep [B] or scalar

        Returns:
            Dictionary with:
                - 'flow_7': Flow from frame 7 to target [B, 2, H/4, W/4]
                - 'flow_9': Flow from frame 9 to target [B, 2, H/4, W/4]
                - 'occ_7': Occlusion map for frame 7 [B, 1, H/4, W/4]
                - 'occ_9': Occlusion map for frame 9 [B, 1, H/4, W/4]
                - 'logit_occ_7': Occlusion logits for frame 7 (before sigmoid)
                - 'logit_occ_9': Occlusion logits for frame 9 (before sigmoid)
        """
        B = ref_frames.shape[0]
        device = ref_frames.device

        feat_7_s8, feat_9_s8 = ref_feats_s8[:, 0], ref_feats_s8[:, 1]
        feat_7_s4, feat_9_s4 = ref_feats_s4[:, 0], ref_feats_s4[:, 1]

        # Prepare timestep as spatial tensor
        if isinstance(timestep, (float, int)):
            timestep = torch.full((B, 1, 1, 1), timestep, device=device)
        elif timestep.dim() == 0:
            timestep = timestep.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif timestep.dim() == 1:
             timestep = timestep.view(B, 1, 1, 1)

        # Stage 3.1: Coarse estimation at s8
        # Downsample context to s8
        H_s8, W_s8 = feat_7_s8.shape[2:]
        context_s8 = F.interpolate(context, size=(H_s8, W_s8), mode='bilinear', align_corners=False)
        timestep_s8 = timestep.expand(-1, -1, H_s8, W_s8)

        # Concatenate all inputs for s8
        input_s8 = torch.cat([feat_7_s8, feat_9_s8, context_s8, timestep_s8], dim=1)
        flow_s8, occ_logits_s8 = self.block_s8(input_s8, scale=1)

        # Stage 3.2: Refinement at s4
        # Upsample flows from s8 to s4
        H_s4, W_s4 = feat_7_s4.shape[2:]

        # Upsample coarse flow
        flow_up = F.interpolate(flow_s8, size=(H_s4, W_s4), mode='bilinear', align_corners=False) * 2.0
        occ_up = F.interpolate(occ_logits_s8, size=(H_s4, W_s4), mode='bilinear', align_corners=False)

        # flow_up[:, :2] to ruch w stronę klatki 7. Pobieramy cechy z 7 i przesuwamy je "do nas" (do czasu t)
        warped_feat_7 = backward_warp(feat_7_s4, flow_up[:, :2])

        # flow_up[:, 2:] to ruch w stronę klatki 9.
        warped_feat_9 = backward_warp(feat_9_s4, flow_up[:, 2:])

        # Reszta bez zmian...
        context_s4 = F.interpolate(context, size=(H_s4, W_s4), mode='bilinear', align_corners=False)
        timestep_s4 = timestep.expand(-1, -1, H_s4, W_s4)

        input_s4 = torch.cat([
            feat_7_s4, feat_9_s4,         # Oryginały
            warped_feat_7, warped_feat_9, # Warpowane
            context_s4,
            flow_up, occ_up,
            timestep_s4
        ], dim=1)

        flow_res, occ_res = self.block_s4(input_s4, scale=1)

        # Sumujemy
        flow_final = flow_up + flow_res
        occ_logits_final = occ_up + occ_res

        return {
            'flow_7': flow_final[:, :2],
            'flow_9': flow_final[:, 2:],
            'occ_7': torch.sigmoid(occ_logits_final[:, 0:1]),
            'occ_9': torch.sigmoid(occ_logits_final[:, 1:2]),
            'logit_occ_7': occ_logits_final[:, 0:1],
            'logit_occ_9': occ_logits_final[:, 1:2]
        }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs.default import Config

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test inputs
    B, H, W = 2, 256, 256
    ref_frames = torch.rand(B, 2, 3, H, W).to(device)
    ref_feats_s8 = torch.rand(B, 2, 192, H // 8, W // 8).to(device)
    ref_feats_s4 = torch.rand(B, 2, 128, H // 4, W // 4).to(device)
    context = torch.rand(B, 256, H // 16, W // 16).to(device)
    timestep = torch.tensor(0.5).to(device)

    # Create flow estimator
    flow_estimator = FlowEstimator(config).to(device)

    # Forward pass
    with torch.no_grad():
        output = flow_estimator(ref_frames, ref_feats_s8, ref_feats_s4, context, timestep)

    print("Flow Estimator outputs:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Verify occlusion maps are in [0, 1]
    print(f"\nOcclusion map ranges:")
    print(f"  occ_7: [{output['occ_7'].min():.4f}, {output['occ_7'].max():.4f}]")
    print(f"  occ_9: [{output['occ_9'].min():.4f}, {output['occ_9'].max():.4f}]")