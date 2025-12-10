"""
Stage 2: Temporal Transformer

Aggregates temporal information from 15 frames using windowed attention.
Uses spatial patching and temporal windowing for memory efficiency.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.default import Config

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient spatial processing.

    Reduces parameters and computation compared to standard convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FullTemporalAttention(nn.Module):
    """
    Pełna atencja czasowa (Full Self-Attention).
    Zgodnie ze specyfikacją, dla T=14 koszt obliczeniowy jest niewielki,
    więc nie potrzebujemy okienkowania.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.dropout_p = dropout

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, L, D]
               B = batch size
               T = liczba klatek (14)
               L = liczba tokenów przestrzennych (np. 64 dla patch 2x2)
               D = wymiar osadzenia (256)
        """
        B, T, L, D = x.shape

        # Reshape, aby atencja działała po wymiarze czasu (T) dla każdego patcha (L) niezależnie
        # [B, T, L, D] -> [B, L, T, D] -> [B*L, T, D]
        x = x.permute(0, 2, 1, 3).reshape(B * L, T, D)

        # QKV projection
        qkv = self.qkv(x)  # [B*L, T, 3*D]
        qkv = qkv.reshape(B * L, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B*L, heads, T, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention: Attention(Q, K, V)
        # [B*L, heads, T, head_dim] @ [B*L, heads, head_dim, T] -> [B*L, heads, T, T]
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        # Reshape back
        out = out.transpose(1, 2).reshape(B * L, T, D) # [B*L, T, D]

        # [B*L, T, D] -> [B, L, T, D] -> [B, T, L, D]
        out = out.view(B, L, T, D).permute(0, 2, 1, 3)

        out = self.proj(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """
    Complete transformer block with temporal attention and spatial processing.

    Architecture:
    1. Windowed temporal attention along time dimension
    2. Spatial depthwise separable convolution
    3. Feed-forward network (FFN)

    Uses pre-normalization (LayerNorm before attention/FFN) for stability.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.dim = dim

        # Temporal attention
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = FullTemporalAttention(dim, num_heads, dropout)

        # Spatial processing
        self.norm2 = nn.GroupNorm(8, dim)  # GroupNorm for spatial features
        self.spatial_conv = DepthwiseSeparableConv(dim, dim, kernel_size=3, padding=1)

        # Feed-forward network
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, spatial_h, spatial_w):
        """
        Process tokens through transformer block.

        Args:
            x: Input tensor [B, T, L, D]
            spatial_h: Spatial height of feature map
            spatial_w: Spatial width of feature map

        Returns:
            Output tensor [B, T, L, D]
        """
        B, T, L, D = x.shape

        # 1. Temporal attention
        residual = x
        x = self.norm1(x)
        x = self.temporal_attn(x)
        x = x + residual

        # 2. Spatial processing
        x_spatial = x.view(B * T, spatial_h, spatial_w, D).permute(0, 3, 1, 2)
        residual = x_spatial
        x_spatial = self.norm2(x_spatial)
        x_spatial = self.spatial_conv(x_spatial)
        x_spatial = x_spatial + residual
        x = x_spatial.permute(0, 2, 3, 1).reshape(B, T, L, D)

        # 3. FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + residual

        return x


class TemporalAggregator(nn.Module):
    """
    Aggregates 15-frame temporal context using transformer.

    Process:
    1. Convert spatial features to tokens (spatial patching)
    2. Process through transformer layers
    3. Aggregate temporal information using learned attention weights
    """

    def __init__(self, config):
        super().__init__()

        self.num_layers = config.transformer_layers
        self.dim = config.transformer_dim
        self.num_heads = config.transformer_heads
        self.patch_size = config.spatial_patch_size

        # encoder_channels['s16'] = 256
        # patch_size = 2
        # patch_dim = 256 * 2 * 2 = 1024

        # Calculate expected input dimension from config
        c_in = config.encoder_channels['s16']
        patch_dim = c_in * self.patch_size * self.patch_size

        print(f"TemporalAggregator Config: layers={self.num_layers}, dim={self.dim}, heads={self.num_heads}, dropout={config.transformer_dropout}")

        # Define projection layer in __init__ to ensure it's registered
        if patch_dim != self.dim:
            self.patch_proj = nn.Linear(patch_dim, self.dim)
        else:
            self.patch_proj = nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.dim,
                self.num_heads,
                config.transformer_dropout
            )
            for _ in range(self.num_layers)
        ])

        # Frame importance network
        # Learns which frames are most important for interpolation
        self.frame_importance = nn.Sequential(
            nn.Linear(self.dim, self.dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // 4, 1)
        )

    def spatial_to_tokens(self, x, patch_size):
        """
        Convert spatial features to tokens using patching.

        Args:
            x: Features [B, T, C, H, W]
            patch_size: Size of spatial patches

        Returns:
            Tokens [B, T, L, D] where L = (H/patch_size) * (W/patch_size)
        """
        B, T, C, H, W = x.shape

        # Unfold to patches
        # [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.view(B * T, C, H, W)

        # Use unfold for patching
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        # [B*T, C*patch_size*patch_size, num_patches]

        num_patches = patches.shape[2]
        patch_dim = C * patch_size * patch_size

        # Reshape to [B, T, num_patches, patch_dim]
        tokens = patches.transpose(1, 2).reshape(B, T, num_patches, patch_dim)

        # Project to model dimension using the layer defined in __init__
        tokens = self.patch_proj(tokens)

        return tokens, H // patch_size, W // patch_size

    def tokens_to_spatial(self, tokens, spatial_h, spatial_w, patch_size):
        """
        Convert tokens back to spatial features.

        Args:
            tokens: Tokens [B, T, L, D]
            spatial_h: Height of spatial grid (in patches)
            spatial_w: Width of spatial grid (in patches)
            patch_size: Size of patches

        Returns:
            Features [B, T, D, H, W]
        """
        B, T, L, D = tokens.shape

        # Reshape to spatial grid
        x = tokens.view(B, T, spatial_h, spatial_w, D)
        x = x.permute(0, 1, 4, 2, 3)  # [B, T, D, H, W]

        # Upsample if patch_size > 1
        if patch_size > 1:
            H_out = spatial_h * patch_size
            W_out = spatial_w * patch_size
            x = F.interpolate(
                x.view(B * T, D, spatial_h, spatial_w),
                size=(H_out, W_out),
                mode='bilinear',
                align_corners=False
            )
            x = x.view(B, T, D, H_out, W_out)

        return x

    def forward(self, feats_s16):
        """
        Aggregate temporal context from 15 frames.

        Args:
            feats_s16: Features at s16 scale [B, 15, 256, H/16, W/16]

        Returns:
            Dictionary with:
                - 'context': Aggregated temporal context [B, 256, H/16, W/16]
                - 'attention_weights': Frame importance weights [B, 15]
        """
        B, T, C, H, W = feats_s16.shape

        # Convert to tokens
        tokens, spatial_h, spatial_w = self.spatial_to_tokens(
            feats_s16, self.patch_size
        )  # [B, T, L, D]

        # Process through transformer layers
        for layer in self.layers:
            tokens = layer(tokens, spatial_h, spatial_w)

        # Compute frame importance weights
        # Global average pool over spatial tokens for each frame
        frame_features = tokens.mean(dim=2)  # [B, T, D]

        # Predict importance scores
        importance_logits = self.frame_importance(frame_features)  # [B, T, 1]
        importance_weights = torch.softmax(importance_logits.squeeze(-1), dim=1)  # [B, T]

        # Weighted aggregation
        # [B, T, L, D] * [B, T, 1, 1] -> [B, T, L, D] -> [B, L, D]
        weighted_tokens = tokens * importance_weights.unsqueeze(-1).unsqueeze(-1)
        aggregated_tokens = weighted_tokens.sum(dim=1)  # [B, L, D]

        # Convert back to spatial features
        aggregated_tokens = aggregated_tokens.unsqueeze(1)  # [B, 1, L, D]
        context = self.tokens_to_spatial(
            aggregated_tokens, spatial_h, spatial_w, self.patch_size
        )  # [B, 1, D, H, W]
        context = context.squeeze(1)  # [B, D, H, W]

        return {
            'context': context,
            'attention_weights': importance_weights
        }


if __name__ == '__main__':
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test
    num_test_frames = config.model_frames
    feats_s16 = torch.rand(2, num_test_frames, 256, 16, 16).to(device)

    # Create aggregator
    aggregator = TemporalAggregator(config).to(device)

    # Forward pass
    with torch.no_grad():
        output = aggregator(feats_s16)

    print("Temporal Aggregator output shapes:")
    print(f"  Context: {output['context'].shape}")
    print(f"  Attention weights: {output['attention_weights'].shape}")
    print(f"\nAttention weights for first sample:")
    print(f"  Min: {output['attention_weights'][0].min():.4f}")
    print(f"  Max: {output['attention_weights'][0].max():.4f}")
    print(f"  Sum: {output['attention_weights'][0].sum():.4f}")
