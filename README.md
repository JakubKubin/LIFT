# LIFT: Long-range Interpolation with Far Temporal context

Implementation of LIFT video frame interpolation model with 64-frame temporal context.

## Project Structure

```
LIFT/
├── model/               # Model components
│   ├── encoder.py      # Stage 1: Feature extraction from 64 frames
│   ├── transformer.py  # Stage 2: Temporal aggregation with windowed attention
│   ├── ifnet.py       # Stage 3: Multi-scale flow estimation
│   ├── synthesis.py   # Stage 4: Coarse frame synthesis  
│   ├── refine.py      # Stage 5: Full-resolution refinement
│   ├── warplayer.py   # Warping utilities
│   ├── loss.py        # Loss functions
│   └── lift.py        # Main LIFT model
│
├── dataset/            # Data pipeline
│   ├── vimeo_64.py    # 64-frame sequence dataset
│   └── __init__.py
│
├── utils/              # Utilities
│   ├── flow_viz.py    # Flow visualization
│   └── __init__.py
│
├── configs/            # Configuration files
│   └── default.py     # Default hyperparameters
│
├── train.py           # Training script
├── inference.py       # Inference script
└── README.md          # This file
```

## Data Pipeline Design

### Memory Optimization Strategy

The key challenge is handling 64 frames efficiently. Our approach:

1. **Lazy Loading**: Frames are loaded on-demand, not stored in memory
2. **Sequential Reading**: Read frames one by one to avoid memory spikes
3. **Efficient Augmentation**: Apply transforms during loading, not after
4. **No Full-Sequence Storage**: Process frames in batches at different scales

### Dataset Classes

#### Vimeo64Dataset
- Loads 64 consecutive frames from image sequences
- Supports directory structures:
  - Vimeo-style: `sequences/category/sequence_id/im*.png`
  - Flat: `sequence_id/im*.png`
- Memory-efficient sequential loading
- Data augmentation: random crop, flip, rotation, temporal reversal

#### VideoSequenceDataset  
- Loads frames directly from video files
- More efficient for long sequences (uses video codecs)
- Supports random frame extraction for augmentation

### Data Format

Each dataset item returns a dictionary:
```python
{
    'frames': Tensor[64, 3, H, W],      # All 64 frames
    'ref_frames': Tensor[2, 3, H, W],   # Reference frames (31, 32)
    'gt': Tensor[3, H, W],              # Ground truth interpolation
    'timestep': float                    # Interpolation time (0.5)
}
```

### Memory Usage Estimates

For batch_size=4, resolution=256x256, fp32:
- Input frames (64 frames): 4 * 64 * 3 * 256 * 256 * 4 = 201 MB
- Features at s16: 4 * 64 * 256 * 16 * 16 * 4 = 67 MB
- Features at s8: 4 * 2 * 192 * 32 * 32 * 4 = 6 MB
- Features at s4: 4 * 2 * 128 * 64 * 64 * 4 = 16 MB

Total peak memory during forward pass: ~300-400 MB per batch (excluding gradients)

### DataLoader Configuration

Recommended settings for 16GB GPU:
```python
train_loader = DataLoader(
    dataset,
    batch_size=4,           # Adjust based on GPU memory
    num_workers=4,          # Parallel loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True,         # Consistent batch sizes
    shuffle=True
)
```

## Model Architecture (LIFT)

### Stage 1: Feature Extraction
- Shared encoder for all 64 frames
- Multi-scale features: s4, s8, s16
- Sinusoidal positional encoding
- Memory optimization: only keep s16 for all frames, s4/s8 only for ref frames

### Stage 2: Temporal Transformer
- 4-layer transformer with windowed attention
- Window size: 8 frames (8x8=64 frame coverage)
- Spatial patching: 2x2 patches at s16
- Adaptive frame weighting for temporal aggregation

### Stage 3: Flow Estimation
- Two-scale cascade: s8 -> s4
- Bi-directional flows: I_31 -> I_t and I_32 -> I_t  
- Occlusion maps predicted in logit space
- Context injection from 64-frame aggregation

### Stage 4: Coarse Synthesis
- Backward warping with occlusion-aware blending
- Context injection network
- Residual refinement

### Stage 5: Full Resolution Refinement
- Lightweight refinement network
- Channel reduction for memory efficiency (128 -> 32)
- ResBlock-based architecture with GroupNorm

## Training Strategy

1. **Phase 1 (Epochs 0-10)**: 
   - Freeze encoder weights (use pretrained RIFE)
   - Train transformer and flow modules
   - Resolution: 224x224

2. **Phase 2 (Epochs 10+)**:
   - Unfreeze encoder for fine-tuning
   - Train end-to-end
   - Resolution: 256x256 or 256x448

## Usage

### Training
```bash
python train.py --config configs/default.py --batch_size 4
```

### Inference
```bash
python inference.py --input video.mp4 --output output/ --checkpoint weights.pth
```

## Dependencies

- PyTorch >= 1.10
- OpenCV (cv2)
- NumPy
- TensorBoard (for logging)

## Notes

- This implementation focuses on memory efficiency for research purposes
- 64-frame context provides significant quality improvements over 2-frame methods
- Suitable for both academic research and practical applications
