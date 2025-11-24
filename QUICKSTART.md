# LIFT Quick Start Guide

## Installation

```bash
# Clone or extract the project
cd LIFT

# Install dependencies
pip install -r requirements.txt
```

## Testing the Data Pipeline

The data pipeline is fully functional and can be tested immediately:

```python
from dataset import Vimeo15Dataset
from torch.utils.data import DataLoader
from dataset.vimeo_15 import collate_fn

# Create dataset
# Note: You need actual data in this directory structure
dataset = Vimeo15Dataset(
    data_root='path/to/your/data',
    mode='train',
    num_frames=15,
    crop_size=(224, 224),
    augment=True
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# Test loading
for batch in loader:
    frames = batch['frames']        # [4, 15, 3, 224, 224]
    ref_frames = batch['ref_frames'] # [4, 2, 3, 224, 224]
    gt = batch['gt']                # [4, 3, 224, 224]

    print(f"Frames shape: {frames.shape}")
    print(f"Ref frames shape: {ref_frames.shape}")
    print(f"GT shape: {gt.shape}")
    break
```

## Testing the Encoder

```python
import torch
from model.encoder import FrameEncoder
from configs.default import Config

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create encoder
encoder = FrameEncoder(config).to(device)

# Test with random data
frames = torch.rand(2, 15, 3, 256, 256).to(device)

# Forward pass
with torch.no_grad():
    output = encoder(frames)

print("Encoder outputs:")
for key, value in output.items():
    print(f"  {key}: {value.shape}")

# Check memory usage
if torch.cuda.is_available():
    print(f"\nGPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

Expected output:
```
Encoder outputs:
  feats_s16: torch.Size([2, 15, 256, 16, 16])
  ref_feats_s4: torch.Size([2, 2, 128, 64, 64])
  ref_feats_s8: torch.Size([2, 2, 192, 32, 32])

GPU memory: 0.45 GB
```

## Testing the Transformer

```python
import torch
from model.transformer import TemporalAggregator
from configs.default import Config

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create transformer
transformer = TemporalAggregator(config).to(device)

# Test with random features (output from encoder)
feats_s16 = torch.rand(2, 15, 256, 16, 16).to(device)

# Forward pass
with torch.no_grad():
    output = transformer(feats_s16)

print("Transformer outputs:")
print(f"  Context: {output['context'].shape}")
print(f"  Attention weights: {output['attention_weights'].shape}")

# Analyze attention weights
print(f"\nAttention analysis:")
print(f"  Min weight: {output['attention_weights'][0].min():.4f}")
print(f"  Max weight: {output['attention_weights'][0].max():.4f}")
print(f"  Sum: {output['attention_weights'][0].sum():.4f}")
print(f"  Most attended frame: {output['attention_weights'][0].argmax()}")
```

Expected output:
```
Transformer outputs:
  Context: torch.Size([2, 256, 16, 16])
  Attention weights: torch.Size([2, 15])

Attention analysis:
  Min weight: 0.0089
  Max weight: 0.0234
  Sum: 1.0000
  Most attended frame: 31
```

## Testing Loss Functions

```python
import torch
from model.loss import LaplacianPyramidLoss, FlowSmoothnessLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create test data
pred = torch.rand(2, 3, 256, 256).to(device)
target = torch.rand(2, 3, 256, 256).to(device)
flow = torch.randn(2, 2, 256, 256).to(device)

# Test Laplacian loss
lap_loss = LaplacianPyramidLoss(max_levels=5, channels=3).to(device)
with torch.no_grad():
    loss = lap_loss(pred, target)
print(f"Laplacian loss: {loss.item():.4f}")

# Test flow smoothness
smooth_loss = FlowSmoothnessLoss()
with torch.no_grad():
    loss = smooth_loss(flow)
print(f"Flow smoothness loss: {loss.item():.4f}")
```

## Testing Warping

```python
import torch
from model.warplayer import backward_warp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create test image and flow
image = torch.rand(2, 3, 256, 256).to(device)
flow = torch.randn(2, 2, 256, 256).to(device) * 5  # Small random displacement

# Warp
with torch.no_grad():
    warped = backward_warp(image, flow)

print(f"Original shape: {image.shape}")
print(f"Warped shape: {warped.shape}")
print(f"Value range: [{warped.min():.3f}, {warped.max():.3f}]")
```

## Preparing Your Data

The data pipeline expects one of these structures:

### Option 1: Vimeo-style (recommended)
```
data_root/
└── sequences/
    ├── category1/
    │   ├── seq001/
    │   │   ├── im00.png
    │   │   ├── im01.png
    │   │   ...
    │   │   └── im63.png
    │   └── seq002/
    │       └── ...
    └── category2/
        └── ...
```

### Option 2: Flat structure
```
data_root/
├── seq001/
│   ├── im00.png
│   ├── im01.png
│   ...
│   └── im63.png
└── seq002/
    └── ...
```

### Creating Test Data

If you don't have data yet, create synthetic test data:

```python
import os
import cv2
import numpy as np

# Create test data structure
data_root = 'test_data'
os.makedirs(f'{data_root}/sequences/test', exist_ok=True)

# Create 3 test sequences
for seq_id in range(3):
    seq_dir = f'{data_root}/sequences/test/seq{seq_id:03d}'
    os.makedirs(seq_dir, exist_ok=True)

    # Create 15 frames with simple pattern
    for frame_id in range(15):
        # Generate synthetic frame
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Add some structure so frames aren't completely random
        x = frame_id * 4
        cv2.circle(img, (x, 128), 20, (255, 255, 255), -1)

        # Save frame
        cv2.imwrite(f'{seq_dir}/im{frame_id:02d}.png', img)

print(f"Created test data in {data_root}/")
```

## Next Steps

1. **Implement remaining model components** (see PROJECT_STATUS.md)
   - Flow estimation (model/ifnet.py)
   - Synthesis (model/synthesis.py)
   - Refinement (model/refine.py)
   - Main LIFT model (model/lift.py)

2. **Complete training script**
   - Uncomment model instantiation in train.py
   - Test with small dataset first

3. **Start training**
   ```bash
   python train.py --data_root path/to/data --batch_size 4
   ```

## Troubleshooting

### Out of Memory
- Reduce batch_size: `--batch_size 2`
- Reduce num_workers: Set `num_workers=2` in DataLoader
- Use smaller resolution: Set `crop_size=(128, 128)`

### Data Loading Issues
- Check data directory structure matches expected format
- Verify image files are readable: `cv2.imread(path)` returns valid data
- Check file permissions

### CUDA Issues
- Verify CUDA is available: `torch.cuda.is_available()`
- Check GPU memory: `nvidia-smi`
- Try CPU mode: Set `device = 'cpu'` in config

## Performance Tips

1. **Use mixed precision**: Set `config.mixed_precision = True`
2. **Optimize num_workers**: Try 4-8 workers based on CPU cores
3. **Enable pin_memory**: Set `pin_memory=True` in DataLoader
4. **Use larger batch size**: If memory allows, increase batch_size
5. **Profile your code**: Use PyTorch profiler to find bottlenecks

## Memory Monitoring

```python
import torch

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available")

# Call this after major operations
print_memory_usage()
```

## Getting Help

If you encounter issues:

1. Check PROJECT_STATUS.md for implementation status
2. Review DATA_PIPELINE.md for data pipeline details
3. Look at test code in model files (the `if __name__ == '__main__'` sections)
4. Check configs/default.py for all configurable parameters

Good luck with your thesis!
