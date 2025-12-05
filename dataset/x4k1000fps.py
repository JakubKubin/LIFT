"""
X4K1000FPS Dataset for LIFT training.

Dataset structure:
/data/X4K1000FPS/
├── 001/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── 002/
│   └── ...
└── ...
"""

import os
import random
from pathlib import Path
from typing import Tuple
from .base_video import BaseVideoDataset
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.default import Config

config = Config()

class X4K1000FPSDataset(BaseVideoDataset):
    """
    X4K1000FPS video dataset for frame interpolation.

    Extracts 15 consecutive frames from high-frame-rate videos.
    """

    def __init__(self,
                 data_root: str | None = str(Path(__file__).parent.parent.resolve() / 'data' / 'X4K1000FPS'),
                 mode: str = 'train',
                 num_frames: int = 15,
                 crop_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 cache_frames: bool = False,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 input_scale: float = 1.0,
                 max_sequences: int | None = None,
                 stride: int = 1):

        if data_root is None:
            data_root = str(Path(__file__).parent.parent.resolve() / 'data' / 'X4K1000FPS')

        super().__init__(data_root, mode, num_frames, crop_size, augment,
                                cache_frames, input_scale, stride, max_sequences)
        self._auto_build_dataset(['.mp4'], train_split, val_split)

if __name__ == '__main__':
    # Test X4K1000FPS dataset
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    print("Testing X4K1000FPS Dataset...")

    # Check if dataset exists
    data_root = '/data/X4K1000FPS'
    if not os.path.exists(data_root):
        print(f"Dataset not found at {data_root}")
        print("Please ensure X4K1000FPS is downloaded and extracted to /data/")
        sys.exit(1)

    # Create dataset
    try:
        dataset = X4K1000FPSDataset(
            data_root=data_root,
            mode='train',
            num_frames=15,
            crop_size=(224, 224),
            augment=True,
            cache_frames=False,
            max_sequences=50
        )

        print(f"Dataset loaded successfully!")
        print(f"Total sequences: {len(dataset)}")

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

        print("\nDataset test passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()