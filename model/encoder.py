"""
Stage 1: Feature Extraction Encoder

Extracts multi-scale features from each of the 15 input frames.
Uses a shared encoder based on RIFE architecture with positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """
    Basic convolutional block with PReLU activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Stride for convolution
        padding: Padding for convolution

    Returns:
        Sequential module
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        nn.PReLU(out_channels)
    )


class ResidualBlock(nn.Module):
    """
    Residual block for feature extraction.

    Memory efficient implementation with in-place operations where possible.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # Residual connection
        out = self.prelu(out)
        return out


class FeatureEncoder(nn.Module):
    """
    Multi-scale feature encoder for single frame.

    Extracts features at four scales: s1 (1/1), s4 (1/4), s8 (1/8), s16 (1/16).
    Based on RIFE architecture but adapted for LIFT.
    """

    def __init__(self, config):
        super().__init__()

        # Channel dimensions for each scale
        c_s1 = config.encoder_channels['s1']    # 32
        c_s4 = config.encoder_channels['s4']    # 128
        c_s8 = config.encoder_channels['s8']    # 192
        c_s16 = config.encoder_channels['s16']  # 256

        # Full resolution input conv
        # Input: H, W -> Output: H, W (Stride 1)
        self.conv_s1 = nn.Sequential(
            conv_block(3, 32, 3, 1, 1),
            conv_block(32, c_s1, 3, 1, 1)
        )

        # s1 -> s2 -> s4
        # Input: H, W -> Output: H/4, W/4
        self.conv_s1_to_s4 = nn.Sequential(
            conv_block(c_s1, 64, 3, 2, 1),      # Downsample to H/2
            conv_block(64, 64, 3, 2, 1),        # Downsample to H/4
            ResidualBlock(64),
            ResidualBlock(64),
            conv_block(64, c_s4, 3, 1, 1)
        )

        # Scale s8 (1/8 resolution)
        self.conv_s8 = nn.Sequential(
            conv_block(c_s4, c_s4, 3, 2, 1),    # Downsample to H/8
            ResidualBlock(c_s4),
            ResidualBlock(c_s4),
            conv_block(c_s4, c_s8, 3, 1, 1)
        )

        # Scale s16 (1/16 resolution)
        self.conv_s16 = nn.Sequential(
            conv_block(c_s8, c_s8, 3, 2, 1),    # Downsample to H/16
            ResidualBlock(c_s8),
            ResidualBlock(c_s8),
            conv_block(c_s8, c_s16, 3, 1, 1)
        )

    def forward(self, x):
        """
        Extract multi-scale features.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Dictionary with keys 's1', 's4', 's8', 's16' containing features
        """
        # Initial features (now H/2, W/2)
        feat_s1 = self.conv_s1(x)           # [B, 32, H, W]

        # Downsample to s4
        feat_s4 = self.conv_s1_to_s4(feat_s1) # [B, 128, H/4, W/4]

        # Downsample to s8
        feat_s8 = self.conv_s8(feat_s4)     # [B, 192, H/8, W/8]

        # Downsample to s16
        feat_s16 = self.conv_s16(feat_s8)   # [B, 256, H/16, W/16]

        return {
            's1': feat_s1,
            's4': feat_s4,
            's8': feat_s8,
            's16': feat_s16
        }


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal positions.

    Adds temporal information to features so the model knows which frame
    each feature map comes from in the 15-frame sequence.
    """

    pe: torch.Tensor

    def __init__(self, num_frames=15, d_model=256):
        super().__init__()

        # Create positional encoding for all frames
        # Shape: [num_frames, d_model]
        pe = torch.zeros(num_frames, d_model)
        position = torch.arange(0, num_frames, dtype=torch.float32).unsqueeze(1)

        # Create frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x, frame_idx):
        """
        Add positional encoding to features.

        Args:
            x: Features [B, C, H, W]
            frame_idx: Frame index in [0, num_frames-1]

        Returns:
            Features with added positional encoding [B, C, H, W]
        """
        # Get positional encoding for this frame [C]
        pe = self.pe[frame_idx]  # [C]

        # Reshape to [1, C, 1, 1] and add to features
        pe = pe.view(1, -1, 1, 1)

        return x + pe


class FrameEncoder(nn.Module):
    """
    Complete encoding module for all 15 frames.

    Processes frames efficiently using shared encoder and adds positional encoding.

    Memory optimization:
    - Processes frames one at a time (or in small batches)
    - Only keeps s16 features for all frames
    - Only keeps s1, s4, s8 features for reference frames (7, 9)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_frames = config.model_frames # 14 frames to process
        self.total_frames = config.num_frames # 15 total temporal positions

        # Shared encoder for all frames
        self.encoder = FeatureEncoder(config)

        # Positional encodings for each scale
        self.pos_enc_s1 = PositionalEncoding(
            self.total_frames, config.encoder_channels['s1']
        )
        self.pos_enc_s4 = PositionalEncoding(
            self.total_frames, config.encoder_channels['s4']
        )
        self.pos_enc_s8 = PositionalEncoding(
            self.total_frames, config.encoder_channels['s8']
        )
        self.pos_enc_s16 = PositionalEncoding(
            self.total_frames, config.encoder_channels['s16']
        )

        self.ref_indices = config.ref_indices

    def forward(self, frames):
        """
        Encode all 15 frames with positional information.

        Args:
            frames: Input frames [B, 15, 3, H, W]

        Returns:
            Dictionary with:
                - 'feats_s16': Features at s16 for all frames [B, 15, C, H/16, W/16]
                - 'ref_feats_s1': Features at s1 for ref frames [B, 2, C, H, W]
                - 'ref_feats_s4': Features at s4 for ref frames [B, 2, C, H/4, W/4]
                - 'ref_feats_s8': Features at s8 for ref frames [B, 2, C, H/8, W/8]

        Memory optimization:
        - Process frames sequentially to avoid memory spikes
        - Only accumulate necessary features
        """
        B, T, C, H, W = frames.shape

        # Storage for features
        feats_s16_list = []
        ref_feats_s1 = []
        ref_feats_s4 = []
        ref_feats_s8 = []

        gap_idx = self.total_frames // 2
        has_gap = (T == self.num_frames)

        for t in range(T):
            frame_t = frames[:, t]
            feats = self.encoder(frame_t)

            # Calculate REAL temporal index
            if has_gap:
                real_t = t if t < gap_idx else t + 1
            else:
                real_t = t

            # s16 for Transformer (ALL frames)
            feat_s16 = self.pos_enc_s16(feats['s16'], real_t)
            feats_s16_list.append(feat_s16)

            # s1, s4, s8 only for Reference Frames (Memory Optimization)
            if t in self.ref_indices:
                feat_s1 = self.pos_enc_s1(feats['s1'], real_t)
                feat_s4 = self.pos_enc_s4(feats['s4'], real_t)
                feat_s8 = self.pos_enc_s8(feats['s8'], real_t)

                ref_feats_s1.append(feat_s1)
                ref_feats_s4.append(feat_s4)
                ref_feats_s8.append(feat_s8)

        # Fallback
        if len(ref_feats_s4) == 0:
             for t in [0, 1]:
                 if t < T:
                    feats = self.encoder(frames[:, t])
                    ref_feats_s1.append(feats['s1'])
                    ref_feats_s4.append(feats['s4'])
                    ref_feats_s8.append(feats['s8'])

        feats_s16 = torch.stack(feats_s16_list, dim=1)
        ref_feats_s1 = torch.stack(ref_feats_s1, dim=1) # [B, 2, 32, H, W]
        ref_feats_s4 = torch.stack(ref_feats_s4, dim=1)
        ref_feats_s8 = torch.stack(ref_feats_s8, dim=1)

        return {
            'feats_s16': feats_s16,
            'ref_feats_s1': ref_feats_s1,
            'ref_feats_s4': ref_feats_s4,
            'ref_feats_s8': ref_feats_s8
        }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs.default import Config

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test input
    num_test = 14
    frames = torch.rand(2, num_test, 3, 256, 256).to(device)
    encoder = FrameEncoder(config).to(device)

    # Forward pass
    with torch.no_grad():
        output = encoder(frames)

    print("Encoder output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")