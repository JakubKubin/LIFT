"""
UCF-101 Dataset for LIFT training.

Dataset structure:
/data/UCF-101/
├── ApplyEyeMakeup/
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   ├── v_ApplyEyeMakeup_g01_c02.avi
│   └── ...
├── ApplyLipstick/
│   └── ...
├── Archery/
│   └── ...
└── ... (101 action categories total)
"""

import os
import random
from pathlib import Path
from typing import Tuple
from .base_video import BaseVideoDataset

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.default import Config

config = Config()

class UCF101Dataset(BaseVideoDataset):
    """
    UCF-101 action recognition dataset adapted for frame interpolation.
    """
    def __init__(self,
                 data_root: str = '/data/UCF-101',
                 mode: str = 'train',
                 num_frames: int = 15,
                 crop_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 cache_frames: bool = False,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 split_file: str | None = None,
                 input_scale: float = 1.0,
                 max_sequences: int | None = None,
                 stride: int | None = 1):

        super().__init__(data_root, mode, num_frames, crop_size, augment,
                         cache_frames, input_scale, stride, max_sequences)

        self.split_file = split_file

        self._auto_build_dataset(['.avi'], train_split, val_split)

if __name__ == '__main__':
    # Test UCF-101 dataset
    import sys
    sys.path.append('..')

    print("Testing UCF-101 Dataset...")

    # Check if dataset exists
    data_root = '/data/UCF-101'
    if not os.path.exists(data_root):
        print(f"Dataset not found at {data_root}")
        print("Please ensure UCF-101 is downloaded and extracted to /data/")
        sys.exit(1)

    # Test with custom splits AND LIMIT
    print("\n1. Testing with custom splits (limited to 100)...")
    try:
        dataset = UCF101Dataset(
            data_root=data_root,
            mode='train',
            num_frames=15,
            crop_size=(224, 224),
            augment=True,
            cache_frames=False,
            max_sequences=100,
            stride=1
        )

        print(f"Dataset loaded successfully!")
        print(f"Total training sequences: {len(dataset)}")

        # Test loading one sample
        print("\nTesting sample loading...")
        sample = dataset[0]

        print("Sample shapes:")
        print(f"  frames: {sample['frames'].shape}")
        print(f"  ref_frames: {sample['ref_frames'].shape}")
        print(f"  gt: {sample['gt'].shape}")
        print(f"  timestep: {sample['timestep']}")

        print("\nValue ranges:")
        print(f"  frames: [{sample['frames'].min():.3f}, {sample['frames'].max():.3f}]")
        print(f"  ref_frames: [{sample['ref_frames'].min():.3f}, {sample['ref_frames'].max():.3f}]")
        print(f"  gt: [{sample['gt'].min():.3f}, {sample['gt'].max():.3f}]")

        print("\nCustom splits test passed!")

    except Exception as e:
        print(f"Error with custom splits: {e}")
        import traceback
        traceback.print_exc()