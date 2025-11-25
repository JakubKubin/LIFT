import torch
import torchvision
import os
import numpy as np
from dataset import UCF101Dataset

def check_ucf101_visualization():
    # 1. Configuration
    data_root = r'.\data\UCF-101'  # Adjust path if different
    output_path = 'ucf101_sample.png'

    print(f"Initializing UCF-101 dataset from {data_root}...")

    # 2. Initialize Dataset
    # We use mode='val' to avoid random augmentations (cropping/flipping) for visualization
    # so we see the raw center-cropped data.
    try:
        dataset = UCF101Dataset(
            data_root=data_root,
            mode='val',
            num_frames=15,
            crop_size=(224, 224),
            augment=False,           # Disable augmentation for clear visualization
            max_sequences=10         # Load only a few sequences for speed
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Check your path.")
        return

    print("Dataset initialized successfully.")

    # 3. Get one sample
    # Index 0 might be random depending on how list is shuffled, but it's consistent for this run
    sample = dataset[0]

    frames = sample['frames']      # [15, 3, H, W]
    ref_frames = sample['ref_frames'] # [2, 3, H, W]
    gt = sample['gt']              # [3, H, W] (This is the middle frame usually)

    print(f"\nSample loaded:")
    print(f"  Frames Shape: {frames.shape}")
    print(f"  Data Range: [{frames.min():.4f}, {frames.max():.4f}]")
    print(f"  Data Mean:  {frames.mean():.4f}")

    # Check for NaNs/Infs immediately (good sanity check for your training crash)
    if torch.isnan(frames).any() or torch.isinf(frames).any():
        print("!!! WARNING: Data contains NaNs or Infs !!!")
    else:
        print("  Data looks valid (no NaNs/Infs).")

    # 4. Create a Grid Visualization
    print(f"\nSaving visualization to {output_path}...")

    # Un-normalize if you had specific normalization (LIFT assumes [0,1], so we are good)
    # We will arrange them in a grid: 5 columns x 3 rows
    grid_img = torchvision.utils.make_grid(frames, nrow=5, padding=2, normalize=False)

    # Save image
    torchvision.utils.save_image(grid_img, output_path)

    print(f"Done! Open '{output_path}' to see the sequence.")

if __name__ == "__main__":
    check_ucf101_visualization()