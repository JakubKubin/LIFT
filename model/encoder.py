"""
Stage 1: Feature Extraction Encoder

Extracts multi-scale features from each of the 64 input frames.
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
    
    Extracts features at three scales: s4 (1/4), s8 (1/8), s16 (1/16).
    Based on RIFE architecture but adapted for LIFT.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Channel dimensions for each scale
        c_s4 = config.encoder_channels['s4']    # 128
        c_s8 = config.encoder_channels['s8']    # 192
        c_s16 = config.encoder_channels['s16']  # 256
        
        # Initial convolution: 3 -> 32 -> 64
        self.conv_init = nn.Sequential(
            conv_block(3, 32, 3, 1, 1),
            conv_block(32, 64, 3, 1, 1)
        )
        
        # Scale s4 (1/4 resolution): stride 2 downsampling
        self.conv_s4 = nn.Sequential(
            conv_block(64, 64, 3, 2, 1),  # Downsample
            ResidualBlock(64),
            ResidualBlock(64),
            conv_block(64, c_s4, 3, 1, 1)
        )
        
        # Scale s8 (1/8 resolution): another stride 2
        self.conv_s8 = nn.Sequential(
            conv_block(c_s4, c_s4, 3, 2, 1),  # Downsample
            ResidualBlock(c_s4),
            ResidualBlock(c_s4),
            conv_block(c_s4, c_s8, 3, 1, 1)
        )
        
        # Scale s16 (1/16 resolution): final stride 2
        self.conv_s16 = nn.Sequential(
            conv_block(c_s8, c_s8, 3, 2, 1),  # Downsample
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
            Dictionary with keys 's4', 's8', 's16' containing features
        """
        # Initial features
        x = self.conv_init(x)
        
        # Extract at each scale
        feat_s4 = self.conv_s4(x)     # [B, 128, H/4, W/4]
        feat_s8 = self.conv_s8(feat_s4)  # [B, 192, H/8, W/8]
        feat_s16 = self.conv_s16(feat_s8) # [B, 256, H/16, W/16]
        
        return {
            's4': feat_s4,
            's8': feat_s8,
            's16': feat_s16
        }


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal positions.
    
    Adds temporal information to features so the model knows which frame
    each feature map comes from in the 64-frame sequence.
    """
    
    def __init__(self, num_frames=64, d_model=256):
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
    Complete encoding module for all 64 frames.
    
    Processes frames efficiently using shared encoder and adds positional encoding.
    
    Memory optimization:
    - Processes frames one at a time (or in small batches)
    - Only keeps s16 features for all frames
    - Only keeps s4, s8 features for reference frames (31, 32)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_frames = config.num_frames
        
        # Shared encoder for all frames
        self.encoder = FeatureEncoder(config)
        
        # Positional encodings for each scale
        self.pos_enc_s4 = PositionalEncoding(
            self.num_frames, 
            config.encoder_channels['s4']
        )
        self.pos_enc_s8 = PositionalEncoding(
            self.num_frames,
            config.encoder_channels['s8']
        )
        self.pos_enc_s16 = PositionalEncoding(
            self.num_frames,
            config.encoder_channels['s16']
        )
        
        # Reference frame indices (31, 32)
        self.ref_indices = [31, 32]
    
    def forward(self, frames):
        """
        Encode all 64 frames with positional information.
        
        Args:
            frames: Input frames [B, 64, 3, H, W]
            
        Returns:
            Dictionary with:
                - 'feats_s16': Features at s16 for all frames [B, 64, C, H/16, W/16]
                - 'ref_feats_s4': Features at s4 for ref frames [B, 2, C, H/4, W/4]
                - 'ref_feats_s8': Features at s8 for ref frames [B, 2, C, H/8, W/8]
        
        Memory optimization:
        - Process frames sequentially to avoid memory spikes
        - Only accumulate necessary features
        """
        B, T, C, H, W = frames.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"
        
        # Storage for features
        device = frames.device
        feats_s16_list = []
        ref_feats_s4 = []
        ref_feats_s8 = []
        
        # Process each frame sequentially for memory efficiency
        for t in range(T):
            # Extract single frame [B, 3, H, W]
            frame_t = frames[:, t]
            
            # Extract features at all scales
            feats = self.encoder(frame_t)
            
            # Add positional encoding to s16 features
            feat_s16 = self.pos_enc_s16(feats['s16'], t)
            feats_s16_list.append(feat_s16)
            
            # Store s4 and s8 features only for reference frames
            if t in self.ref_indices:
                feat_s4 = self.pos_enc_s4(feats['s4'], t)
                feat_s8 = self.pos_enc_s8(feats['s8'], t)
                ref_feats_s4.append(feat_s4)
                ref_feats_s8.append(feat_s8)
        
        # Stack features along temporal dimension
        feats_s16 = torch.stack(feats_s16_list, dim=1)  # [B, 64, C, H/16, W/16]
        ref_feats_s4 = torch.stack(ref_feats_s4, dim=1)  # [B, 2, C, H/4, W/4]
        ref_feats_s8 = torch.stack(ref_feats_s8, dim=1)  # [B, 2, C, H/8, W/8]
        
        return {
            'feats_s16': feats_s16,
            'ref_feats_s4': ref_feats_s4,
            'ref_feats_s8': ref_feats_s8
        }


if __name__ == '__main__':
    # Test encoder
    from configs.default import Config
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input
    frames = torch.rand(2, 64, 3, 256, 256).to(device)
    
    # Create encoder
    encoder = FrameEncoder(config).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = encoder(frames)
    
    print("Encoder output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
    
    # Check memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
