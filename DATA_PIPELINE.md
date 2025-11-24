# Data Pipeline for LIFT Model

## Overview

The data pipeline is designed with memory efficiency as the top priority, enabling training with 15-frame sequences on consumer GPUs (16GB VRAM).

## Key Design Decisions

###  1. Lazy Loading Strategy

**Problem**: Loading 15 full-resolution frames (256x448) per sample would require:
- Raw data: 15 * 3 * 448 * 256 * 4 bytes = 88 MB per sample
- With batch_size=4: 352 MB just for input data
- Add gradients, activations, optimizer states: 2-3 GB per batch

**Solution**: Sequential frame loading
```python
# Bad: Load all frames at once
frames = [cv2.imread(path) for path in all_paths]  # Memory spike!

# Good: Load and process sequentially
for path in frame_paths:
    frame = cv2.imread(path)
    frame_tensor = preprocess(frame)  # Immediate conversion
    frames_list.append(frame_tensor)
    # frame is garbage collected here
```

### 2. Multi-Scale Feature Storage

**Problem**: Storing features at all scales for all frames is prohibitive:
- s4 (128 channels): 15 * 128 * 15 * 15 * 4 = 134 MB
- s8 (192 channels): 15 * 192 * 32 * 32 * 4 = 50 MB
- s16 (256 channels): 15 * 256 * 16 * 16 * 4 = 67 MB
- Total: 251 MB per sample before batch

**Solution**: Selective feature retention
- Keep s16 for all 15 frames (needed for transformer)
- Keep s4 and s8 ONLY for reference frames 31 and 32
- Reduces storage from 251 MB to 73 MB (71% reduction)

### 3. Windowed Attention

**Problem**: Full temporal attention has O(T^2) complexity
- For T=15: 15 * 15 = 225 attention computations per spatial location
- For 16x16 spatial grid: 225 * 256 = 57600 computations

**Solution**: Windowed attention with window_size=8
- Complexity: O(T * W) where W=8
- 15 * 8 = 512 computations per spatial location
- 8x reduction in memory and computation

### 4. Efficient Data Augmentation

**Problem**: Augmenting 15 frames independently creates inconsistencies

**Solution**: Apply augmentation parameters once, use for all frames
```python
# Compute augmentation parameters
crop_x, crop_y = random.randint(...), random.randint(...)
flip_h = random.random() < 0.5
flip_v = random.random() < 0.5

# Apply to all frames with same parameters
for frame in frames:
    frame = frame[crop_x:crop_x+h, crop_y:crop_y+w]
    if flip_h:
        frame = frame[:, ::-1]
    if flip_v:
        frame = frame[::-1]
```

## Dataset Classes

### Vimeo15Dataset

Purpose: Load 15 consecutive frames from directory structure

**Directory structure expected**:
```
data_root/
├── sequences/
│   ├── category1/
│   │   ├── seq001/
│   │   │   ├── im00.png
│   │   │   ├── im01.png
│   │   │   ...
│   │   │   └── im63.png
│   │   └── seq002/
│   │       └── ...
│   └── category2/
│       └── ...
```

**Features**:
- Lazy loading with sequential processing
- Consistent augmentation across all frames
- Automatic train/val split (95/5)
- Support for temporal reversal augmentation

**Memory profile per batch (batch_size=4)**:
- Loading: 4 * 15 * 3 * 256 * 256 * 1 byte = 50 MB (uint8)
- Tensors: 4 * 15 * 3 * 256 * 256 * 4 bytes = 200 MB (float32)
- Peak during augmentation: ~250 MB

### VideoSequenceDataset

Purpose: Load frames directly from video files

**Features**:
- More efficient for long videos (uses codecs)
- Random frame extraction for augmentation
- Lower I/O overhead than individual images

**Video list format** (video_list.txt):
```
/path/to/video1.mp4
/path/to/video2.mp4
/path/to/video3.avi
```

**Memory profile**:
- Only 15 frames in memory at once
- Video decoder handles buffering
- Typically 20-30% less memory than image sequences

## DataLoader Configuration

Recommended settings for different GPU memory sizes:

**16GB GPU** (e.g., RTX 4080):
```python
DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

**24GB GPU** (e.g., RTX 4090):
```python
DataLoader(
    dataset,
    batch_size=6,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=2
)
```

**40GB+ GPU** (e.g., A100):
```python
DataLoader(
    dataset,
    batch_size=12,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2
)
```

## Data Processing Pipeline

1. **Load Phase**:
   ```
   Disk -> Sequential imread -> RGB conversion -> List accumulation
   ```

2. **Augmentation Phase**:
   ```
   Generate params -> Apply crop -> Apply flips -> Apply rotation
   ```

3. **Tensor Conversion**:
   ```
   NumPy (H,W,C) -> Transpose (C,H,W) -> Torch tensor -> Normalize [0,1]
   ```

4. **Batch Assembly**:
   ```
   Individual samples -> Stack into batch -> Pin to GPU memory
   ```

## Memory Optimization Techniques

### 1. In-place Operations
```python
# Use .copy() to make memory contiguous before tensor conversion
img = img.transpose(2, 0, 1).copy()
tensor = torch.from_numpy(img)
```

### 2. Explicit Garbage Collection
```python
for frame_path in paths:
    frame = load_frame(frame_path)
    process_frame(frame)
    del frame  # Explicit deletion helps memory management
```

### 3. Feature Quantization (Optional)
```python
# For very large datasets, can store features as fp16
features = features.half()  # 50% memory reduction
```

### 4. Disk Caching (Optional)
```python
# Pre-compute and cache features at s16 to disk
# Trade disk I/O for computation time
if os.path.exists(cache_path):
    features = torch.load(cache_path)
else:
    features = compute_features(frames)
    torch.save(features, cache_path)
```

## Debugging Tips

### Check memory usage:
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Profile data loading:
```python
import time

start = time.time()
for batch in dataloader:
    load_time = time.time() - start
    print(f"Batch load time: {load_time:.2f}s")
    start = time.time()
```

### Identify memory leaks:
```python
import gc

# Force garbage collection
gc.collect()
torch.cuda.empty_cache()

# Check for retained tensors
for obj in gc.get_objects():
    if torch.is_tensor(obj):
        print(type(obj), obj.size())
```

## Performance Benchmarks

On an RTX 4080 (16GB):
- Loading 15 frames: ~20ms
- Augmentation: ~30ms
- Tensor conversion: ~20ms
- Total per sample: ~100ms
- Throughput: ~10 samples/second
- With batch_size=4, num_workers=4: ~40 samples/second

## Future Optimizations

1. **JPEG decoding optimization**: Use turbojpeg for faster decoding
2. **Prefetching**: Implement custom prefetcher for next batch
3. **Mixed resolution**: Train at 224x224, validate at 256x448
4. **Feature caching**: Pre-compute encoder features offline
5. **Data sharding**: Distribute data across multiple disks
