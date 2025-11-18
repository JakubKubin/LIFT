"""
Default configuration for LIFT model training and inference.
"""

class Config:
    """Configuration parameters for LIFT model."""
    
    # Data parameters
    data_root = 'data/vimeo90k'
    num_frames = 64
    crop_size = (224, 224)
    target_resolution = (256, 448)  # Target resolution for experiments
    
    # Model architecture parameters
    # Stage 1: Encoder
    encoder_scales = ['s4', 's8', 's16']  # Output scales
    encoder_channels = {
        's4': 128,
        's8': 192,
        's16': 256
    }
    freeze_encoder_epochs = 10  # Freeze encoder for first N epochs
    
    # Stage 2: Transformer
    transformer_layers = 4
    transformer_dim = 256
    transformer_heads = 8
    temporal_window_size = 8  # Frames per window
    spatial_patch_size = 2
    transformer_dropout = 0.1
    
    # Stage 3: Flow estimation
    flow_scales = [8, 4]  # Coarse to fine
    flow_channels = {
        8: 256,
        4: 128
    }
    
    # Stage 4: Synthesis
    context_net_channels = 64
    
    # Stage 5: Refinement
    refine_channels = [64, 32]
    refine_reduce_channels = 32  # Reduce from 128 to this
    
    # Training parameters
    batch_size = 4
    num_epochs = 300
    learning_rate = 3e-4
    weight_decay = 1e-3
    lr_warmup_steps = 2000
    lr_min = 3e-6
    
    # Loss weights
    loss_l1_weight = 1.0
    loss_lap_weight = 1.0
    loss_flow_weight = 0.01
    loss_occlusion_weight = 0.1
    
    # Optimization
    gradient_clip = 1.0
    mixed_precision = True  # Use automatic mixed precision
    
    # DataLoader parameters
    num_workers = 4
    pin_memory = True
    prefetch_factor = 2
    
    # Logging and checkpointing
    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    log_interval = 100  # Log every N steps
    val_interval = 1000  # Validate every N steps
    save_interval = 5000  # Save checkpoint every N steps
    num_val_samples = 100  # Number of samples for validation
    
    # Distributed training
    world_size = 1
    local_rank = 0
    distributed = False
    
    # Inference parameters
    tta = False  # Test-time augmentation
    output_format = 'png'  # Output format for frames
    
    # Device
    device = 'cuda'
    
    # Reproducibility
    seed = 42
    deterministic = False  # Set to True for reproducible results (slower)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def __repr__(self):
        """String representation of config."""
        lines = ['Config:']
        for key, value in sorted(self.to_dict().items()):
            lines.append(f'  {key}: {value}')
        return '\n'.join(lines)


# Create default config instance
default_config = Config()
