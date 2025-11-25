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

## Quick Start

### X4K1000FPS Dataset

```python
from dataset import X4K1000FPSDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = X4K1000FPSDataset(
    data_root='/data/X4K1000FPS',
    mode='train',
    num_frames=15,
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

```

### UCF-101 Dataset

```python
from dataset import UCF101Dataset
from torch.utils.data import DataLoader

# Create dataset with custom splits
train_dataset = UCF101Dataset(
    data_root='/data/UCF-101',
    mode='train',
    num_frames=15,
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
    num_frames=15,                  # Number of frames to extract
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
    num_frames=15,                  # Number of frames to extract
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

**Important**: All augmentations are applied consistently across all 15 frames in a sequence.

## Memory Management

### Frame Caching

By default, `cache_frames=False` to avoid memory issues. Each frame sequence takes:
- 15 frames × 256×256×3 × 1 byte = 1 MB per sequence
- With 10,000 sequences = 1 GB of RAM!

## Downloading Datasets

### X4K1000FPS

1. Download from [official source](https://www.dropbox.com/scl/fo/88aarlg0v72dm8kvvwppe/AHxNqDye4_VMfqACzZNy5rU?rlkey=a2hgw60sv5prq3uaep2metxcn&e=2&dl=0)
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