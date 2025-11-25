"""
LIFT: Long-range Interpolation with Far Temporal context

Main model that integrates all 5 stages:
1. Feature Extraction (encoder.py)
2. Temporal Aggregation (transformer.py)
3. Flow Estimation (ifnet.py)
4. Coarse Synthesis (synthesis.py)
5. Full Resolution Refinement (refine.py)
"""

import torch
import torch.nn as nn
from .encoder import FrameEncoder
from .transformer import TemporalAggregator
from .ifnet import FlowEstimator
from .synthesis import CoarseSynthesis
from .refine import FullResolutionRefinement


class LIFT(nn.Module):
    """
    Complete LIFT model for video frame interpolation.

    Interpolates frame at t=0.5 between reference frames 31 and 32
    using context from all 15 frames in the sequence.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Stage 1: Feature Extraction
        self.encoder = FrameEncoder(config)

        # Stage 2: Temporal Aggregation
        self.transformer = TemporalAggregator(config)

        # Stage 3: Flow Estimation
        self.flow_estimator = FlowEstimator(config)

        # Stage 4: Coarse Synthesis
        self.synthesis = CoarseSynthesis(config)

        # Stage 5: Full Resolution Refinement
        self.refinement = FullResolutionRefinement(config)

        # Training parameters
        self.freeze_encoder_epochs = config.freeze_encoder_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """
        Set current epoch for training.

        Used to control encoder freezing during initial training.
        """
        self.current_epoch = epoch

        # Freeze/unfreeze encoder based on epoch
        if epoch < self.freeze_encoder_epochs:
            self._freeze_encoder()
        else:
            self._unfreeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder weights for stable training of new modules."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self):
        """Unfreeze encoder for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, frames, ref_frames=None, timestep=0.5, return_intermediate=False):
        """
        Forward pass through all stages.

        Args:
            frames: Input frames [B, 15, 3, H, W]
            ref_frames: Reference frames [B, 2, 3, H, W] (optional, extracted from frames if None)
            timestep: Interpolation timestep (default 0.5)
            return_intermediate: Whether to return intermediate outputs for visualization

        Returns:
            Dictionary with:
                - 'prediction': Final interpolated frame [B, 3, H, W]
                - 'coarse': Coarse frame before refinement [B, 3, H/4, W/4]
                - 'flows': Flow predictions
                - 'occlusions': Occlusion maps
                - 'attention_weights': Temporal attention weights [B, 15]

            If return_intermediate=True, also includes:
                - All intermediate outputs from each stage
        """
        B, T, C, H, W = frames.shape

        # Extract reference frames if not provided
        if ref_frames is None:
            ref_frames = frames[:, self.encoder.ref_indices]  # [B, 2, 3, H, W]

        # Stage 1: Extract multi-scale features from all 15 frames
        encoder_output = self.encoder(frames)
        feats_s16 = encoder_output['feats_s16']
        ref_feats_s4 = encoder_output['ref_feats_s4']
        ref_feats_s8 = encoder_output['ref_feats_s8']
        ref_feats_s1 = encoder_output['ref_feats_s1'] # Pobieramy s1

        # Stage 2 (Transformer) i Stage 3 (Flow) i Stage 4 (Synthesis) bez zmian...
        transformer_output = self.transformer(feats_s16)
        context = transformer_output['context']
        attention_weights = transformer_output['attention_weights']

        flow_output = self.flow_estimator(
            ref_frames,
            ref_feats_s8,
            ref_feats_s4,
            context,
            timestep
        )

        synthesis_output = self.synthesis(ref_frames, flow_output, context)
        coarse_frame = synthesis_output['coarse_frame']

        # Stage 5: Refine to full resolution using S1 features
        # Zmieniono argument z ref_feats_s4 na ref_feats_s1
        refinement_output = self.refinement(coarse_frame, ref_feats_s1)
        final_frame = refinement_output['final_frame']

        # Prepare output (bez zmian)
        output = {
            'prediction': final_frame,
            'coarse': coarse_frame,
            'flows': {
                'flow_31': flow_output['flow_31'],
                'flow_32': flow_output['flow_32'],
            },
            'occlusions': {
                'occ_31': flow_output['occ_31'],
                'occ_32': flow_output['occ_32'],
                'logit_occ_31': flow_output['logit_occ_31'],
                'logit_occ_32': flow_output['logit_occ_32'],
            },
            'warped': {
                'warped_31': synthesis_output['warped_31'],
                'warped_32': synthesis_output['warped_32'],
            },
            'attention_weights': attention_weights,
        }

        # Add intermediate outputs if requested
        if return_intermediate:
            output['intermediate'] = {
                'encoder': encoder_output,
                'transformer': transformer_output,
                'flow': flow_output,
                'synthesis': synthesis_output,
                'refinement': refinement_output,
            }

        return output

    def inference(self, frames, timestep=0.5):
        """
        Inference mode - simplified interface.

        Args:
            frames: Input frames [B, 15, 3, H, W]
            timestep: Interpolation timestep (default 0.5)

        Returns:
            Interpolated frame [B, 3, H, W]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(frames, timestep=timestep, return_intermediate=False)
        return output['prediction']

    def count_parameters(self):
        """Count number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def create_lift_model(config, pretrained_encoder=None):
    """
    Factory function to create LIFT model.

    Args:
        config: Configuration object
        pretrained_encoder: Path to pretrained encoder weights (optional)

    Returns:
        LIFT model instance
    """
    model = LIFT(config)

    # Load pretrained encoder if provided
    if pretrained_encoder is not None:
        print(f"Loading pretrained encoder from {pretrained_encoder}")
        checkpoint = torch.load(pretrained_encoder, map_location='cpu')

        # Load only encoder weights
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('encoder.')
        }
        model.encoder.load_state_dict(encoder_state, strict=False)
        print("Pretrained encoder loaded successfully")

    return model


if __name__ == '__main__':
    # Test complete LIFT model
    import sys
    sys.path.append('..')
    from configs.default import Config

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("Testing Complete LIFT Model")
    print("="*60)

    # Create model
    model = LIFT(config).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # Create test input DYNAMICALLY based on configuration
    B, H, W = 2, 256, 256

    # config.model_frames calculates the correct input size automatically
    # (e.g., returns 6 if num_frames=7, or 15 if num_frames=15)
    num_test_frames = config.model_frames

    print(f"\nGenerating random input with {num_test_frames} frames (based on config.num_frames={config.num_frames})...")
    frames = torch.rand(B, num_test_frames, 3, H, W).to(device)

    print(f"Input shape: {frames.shape}")

    # Test forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        # Note: Reference frame logic is handled internally by the model based on config
        output = model(frames, timestep=0.5, return_intermediate=True)

    print("\nOutput shapes:")
    print(f"  Prediction: {output['prediction'].shape}")
    print(f"  Coarse: {output['coarse'].shape}")
    print(f"  Flow 31: {output['flows']['flow_31'].shape}")
    print(f"  Flow 32: {output['flows']['flow_32'].shape}")
    print(f"  Occlusion 31: {output['occlusions']['occ_31'].shape}")
    print(f"  Occlusion 32: {output['occlusions']['occ_32'].shape}")
    print(f"  Attention weights: {output['attention_weights'].shape}")

    # Verify outputs
    print("\nOutput verification:")
    pred = output['prediction']
    print(f"  Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  Prediction mean: {pred.mean():.4f}")
    assert pred.shape == (B, 3, H, W), "Incorrect prediction shape!"
    assert pred.min() >= 0.0 and pred.max() <= 1.0, "Prediction out of valid range!"
    print("  ✓ All checks passed!")

    # Test attention weights
    print("\nAttention weight analysis:")
    attn = output['attention_weights'][0]  # First sample
    print(f"  Sum: {attn.sum():.4f} (should be 1.0)")
    print(f"  Most attended frame: {attn.argmax().item()}")
    print(f"  Top 5 frames: {attn.topk(5).indices.tolist()}")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Test inference mode
    print("\nTesting inference mode...")
    pred_inference = model.inference(frames, timestep=0.5)
    print(f"  Inference output shape: {pred_inference.shape}")

    # Test encoder freezing
    print("\nTesting encoder freezing:")
    model.set_epoch(0)
    print(f"  Epoch 0 - Encoder frozen: {not next(model.encoder.parameters()).requires_grad}")

    model.set_epoch(config.freeze_encoder_epochs)
    print(f"  Epoch {config.freeze_encoder_epochs} - Encoder unfrozen: {next(model.encoder.parameters()).requires_grad}")

    print("\n" + "="*60)
    print("LIFT Model Test Complete!")
    print("="*60)