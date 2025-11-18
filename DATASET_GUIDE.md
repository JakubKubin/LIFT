# X4K1000FPS and UCF-101 Dataset Guide

## Overview

This guide explains how to use the X4K1000FPS and UCF-101 datasets for training LIFT.

## Dataset Structures

### X4K1000FPS
```
/data/X4K1000FPS/
├── 001/
│   ├── scene1.mp4
│   ├── scene2.mp4
│   └── ...
├── 002/
│   ├── scene1.mp4
│   └── ...
├── 003/
└── ...
```

**Characteristics**:
- High frame rate videos (1000 FPS)
- Multiple scenes per category directory
- Format: MP4
- Ideal for frame interpolation (real intermediate frames available)

### UCF-101
```
/data/UCF-101/
├── ApplyEyeMakeup/
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   ├── v_ApplyEyeMakeup_g01_c02.avi
│   └── ...
├── ApplyLipstick/
│   └── ...
├── Archery/
│   └── ...
└── ... (101 action categories)
```

**Characteristics**:
- Action recognition dataset
- 101 action categories
- Format: AVI
- Standard frame rate (~25-30 FPS)

## Quick Start

### X4K1000FPS Dataset

```python
from dataset import X4K1000FPSDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = X4K1000FPSDataset(
    data_root='/data/X4K1000FPS',
    mode='train',
    num_frames=64,
    crop_size=(224, 224),
    augment=True
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Train
for batch in train_loader:
    frames = batch['frames']        # [B, 64, 3, H, W]
    ref_frames = batch['ref_frames'] # [B, 2, 3, H, W]
    gt = batch['gt']                # [B, 3, H, W]
    # ... training code
```

### UCF-101 Dataset

```python
from dataset import UCF101Dataset
from torch.utils.data import DataLoader

# Create dataset with custom splits
train_dataset = UCF101Dataset(
    data_root='/data/UCF-101',
    mode='train',
    num_frames=64,
    crop_size=(224, 224),
    augment=True,
    train_split=0.7,
    val_split=0.15
)

# Or use official splits
from dataset import create_ucf101_with_official_splits

train_dataset, test_dataset = create_ucf101_with_official_splits(
    data_root='/data/UCF-101',
    splits_root='/data/UCF-101/ucfTrainTestlist'
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Dataset Configuration

### X4K1000FPSDataset Parameters

```python
X4K1000FPSDataset(
    data_root='/data/X4K1000FPS',  # Root directory
    mode='train',                   # 'train', 'val', or 'test'
    num_frames=64,                  # Number of frames to extract
    crop_size=(224, 224),          # Random crop size (H, W)
    augment=True,                  # Data augmentation
    cache_frames=False,            # Cache frames in memory (use carefully!)
    train_split=0.8,               # 80% for training
    val_split=0.1                  # 10% for validation, 10% for test
)
```

### UCF101Dataset Parameters

```python
UCF101Dataset(
    data_root='/data/UCF-101',     # Root directory
    mode='train',                   # 'train', 'val', or 'test'
    num_frames=64,                  # Number of frames to extract
    crop_size=(224, 224),          # Random crop size (H, W)
    augment=True,                  # Data augmentation
    cache_frames=False,            # Cache frames in memory
    train_split=0.7,               # 70% for training
    val_split=0.15,                # 15% for validation
    use_official_splits=False,     # Use official UCF-101 splits
    split_file=None                # Path to trainlist*.txt or testlist*.txt
)
```

## Data Augmentation

Both datasets support the following augmentations (when `augment=True`):

1. **Random Crop**: Crop to specified size
2. **Random Horizontal Flip**: 50% probability
3. **Random Vertical Flip**: 50% probability
4. **Random Rotation**: 90°, 180°, 270° (25% each, 25% no rotation)
5. **Random Temporal Flip**: Reverse frame sequence (50% probability)

**Important**: All augmentations are applied consistently across all 64 frames in a sequence.

## Memory Management

### Frame Caching

By default, `cache_frames=False` to avoid memory issues. Each frame sequence takes:
- 64 frames × 256×256×3 × 1 byte = 12.5 MB per sequence
- With 10,000 sequences = 125 GB of RAM!

Only enable caching if:
- You have abundant RAM (128+ GB)
- Dataset is small (<1000 sequences)
- Training on the same data repeatedly

### Efficient Loading

The datasets use lazy loading:
```python
# Frames are loaded on-demand from video
# Only the requested 64 frames are extracted
# No entire video loaded into memory
```

## Training Script Example

```python
from dataset import X4K1000FPSDataset, collate_fn
from model import LIFT, LIFTLoss
from torch.utils.data import DataLoader

# Create datasets
train_dataset = X4K1000FPSDataset(
    data_root='/data/X4K1000FPS',
    mode='train',
    num_frames=64,
    crop_size=(224, 224),
    augment=True
)

val_dataset = X4K1000FPSDataset(
    data_root='/data/X4K1000FPS',
    mode='val',
    num_frames=64,
    crop_size=(224, 224),
    augment=False
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# Create model and train
model = LIFT(config).cuda()
loss_fn = LIFTLoss(config)

for epoch in range(num_epochs):
    for batch in train_loader:
        frames = batch['frames'].cuda()
        ref_frames = batch['ref_frames'].cuda()
        gt = batch['gt'].cuda()
        
        output = model(frames, ref_frames, timestep=0.5)
        loss = loss_fn(output['prediction'], gt)
        
        # ... backward pass
```

## Downloading Datasets

### X4K1000FPS

1. Download from official source
2. Extract to `/data/X4K1000FPS/`
3. Verify structure:
   ```bash
   ls /data/X4K1000FPS/
   # Should show: 001  002  003  ...
   
   ls /data/X4K1000FPS/001/
   # Should show: *.mp4 files
   ```

### UCF-101

1. Download UCF-101 dataset
   - Videos: https://www.crcv.ucf.edu/data/UCF101.php
   - Optional: Train/Test splits

2. Extract to `/data/UCF-101/`
   ```bash
   # Extract videos
   unrar x UCF101.rar /data/UCF-101/
   
   # Optional: Extract official splits
   unrar x UCF101TrainTestSplits-RecognitionTask.zip /data/UCF-101/
   ```

3. Verify structure:
   ```bash
   ls /data/UCF-101/
   # Should show: ApplyEyeMakeup  ApplyLipstick  Archery  ...
   
   ls /data/UCF-101/ApplyEyeMakeup/
   # Should show: *.avi files
   ```

## Testing Your Setup

### Test X4K1000FPS
```bash
cd LIFT
python -m dataset.x4k1000fps
```

Expected output:
```
Testing X4K1000FPS Dataset...
X4K1000FPS train: 12345 video sequences
Dataset loaded successfully!
Total sequences: 12345

Testing sample loading...
Sample shapes:
  frames: torch.Size([64, 3, 224, 224])
  ref_frames: torch.Size([2, 3, 224, 224])
  gt: torch.Size([3, 224, 224])
  timestep: 0.5

Dataset test passed!
```

### Test UCF-101
```bash
cd LIFT
python -m dataset.ucf101
```

## Performance Tips

### Dataloader Settings

For best performance:

```python
# RTX 4080 (16GB)
DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,      # Match CPU cores
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch 2 batches per worker
)

# RTX 4090 (24GB)
DataLoader(
    dataset,
    batch_size=6,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=2
)
```

### Video Decoding Optimization

The datasets use OpenCV for video decoding. For faster loading:

1. **Use SSD**: Place datasets on SSD, not HDD
2. **More Workers**: Increase `num_workers` to parallelize video decoding
3. **Smaller Crops**: Use smaller `crop_size` during initial training

### Monitoring Performance

```python
import time

start = time.time()
for i, batch in enumerate(train_loader):
    load_time = time.time() - start
    print(f"Batch {i}: {load_time:.3f}s")
    
    # Process batch
    # ...
    
    start = time.time()
    
    if i > 10:
        break
```

Typical loading times (with 4 workers):
- X4K1000FPS: 0.5-1.0s per batch
- UCF-101: 0.3-0.7s per batch

## Common Issues

### Issue: "Cannot open video"
**Solution**: Verify video file is not corrupted
```bash
ffmpeg -v error -i video.mp4 -f null -
```

### Issue: "No valid video sequences found"
**Solution**: Check directory structure matches expected format

### Issue: Slow loading
**Solution**: 
- Increase `num_workers`
- Move data to SSD
- Reduce `crop_size`

### Issue: Out of memory
**Solution**:
- Disable `cache_frames`
- Reduce `batch_size`
- Reduce `num_workers`
- Use smaller `crop_size`

## Using with Training Script

Update `train.py` to use these datasets:

```python
# For X4K1000FPS
from dataset import X4K1000FPSDataset

train_dataset = X4K1000FPSDataset(
    data_root='/data/X4K1000FPS',
    mode='train'
)

# For UCF-101
from dataset import UCF101Dataset

train_dataset = UCF101Dataset(
    data_root='/data/UCF-101',
    mode='train'
)

# Rest of training code remains the same
```

## Summary

**X4K1000FPS**: Best for frame interpolation (high FPS, real intermediate frames)
**UCF-101**: Good for diverse motion patterns (action videos)

Both datasets:
- Support 64-frame sequences for LIFT
- Automatic train/val/test splits
- Memory-efficient lazy loading
- Comprehensive data augmentation
- Easy integration with training script

Start with X4K1000FPS if available, as it provides real ground truth intermediate frames from high-FPS videos.
