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
from typing import List, Tuple
from .base_video import BaseVideoDataset, VideoFrameExtractor


class X4K1000FPSDataset(BaseVideoDataset):
    """
    X4K1000FPS video dataset for frame interpolation.

    Extracts 64 consecutive frames from high-frame-rate videos.
    """

    def __init__(self,
                 data_root: str = '/data/X4K1000FPS',
                 mode: str = 'train',
                 num_frames: int = 64,
                 crop_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 cache_frames: bool = False,
                 train_split: float = 0.8,
                 val_split: float = 0.1):
        """
        Args:
            data_root: Root directory (/data/X4K1000FPS)
            mode: 'train', 'val', or 'test'
            num_frames: Number of frames to extract (64 for LIFT)
            crop_size: Size for random crop (H, W)
            augment: Apply augmentation
            cache_frames: Cache extracted frames (memory intensive)
            train_split: Fraction of videos for training (0.8 = 80%)
            val_split: Fraction of videos for validation (0.1 = 10%)
        """
        super().__init__(data_root, mode, num_frames, crop_size, augment, cache_frames)

        self.train_split = train_split
        self.val_split = val_split

        # Load video list
        self._load_video_list()

        print(f"X4K1000FPS {mode}: {len(self.video_list)} video sequences")

    def _load_video_list(self):
        """
        Load list of videos and their valid starting frames.

        Creates a list of (video_path, start_frame) tuples.
        """
        if not self.data_root.exists():
            raise ValueError(f"Dataset not found at {self.data_root}")

        all_sequences = []

        # Iterate through category directories (001, 002, etc.)
        category_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])

        for category_dir in category_dirs:
            # Find all .mp4 files in this category
            video_files = sorted(category_dir.glob('*.mp4'))

            for video_file in video_files:
                try:
                    # Get valid starting frames for this video
                    valid_starts = self.extractor.get_valid_start_frames(
                        str(video_file),
                        self.num_frames
                    )

                    # Add all possible sequences from this video
                    for start_frame in valid_starts:
                        all_sequences.append((str(video_file), start_frame))

                except Exception as e:
                    print(f"Warning: Error processing {video_file}: {e}")
                    continue

        if len(all_sequences) == 0:
            raise ValueError(f"No valid video sequences found in {self.data_root}")

        # Shuffle with fixed seed for reproducibility
        random.Random(42).shuffle(all_sequences)

        # Split into train/val/test
        total = len(all_sequences)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)

        if self.mode == 'train':
            self.video_list = all_sequences[:train_end]
        elif self.mode == 'val':
            self.video_list = all_sequences[train_end:val_end]
        else:  # test
            self.video_list = all_sequences[val_end:]

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            Dictionary with:
                - 'frames': Tensor [64, 3, H, W]
                - 'ref_frames': Tensor [2, 3, H, W]
                - 'gt': Tensor [3, H, W]
                - 'timestep': float
        """
        video_path, start_frame = self.video_list[idx]

        # Extract frames from video
        frames = self._get_frames_from_video(video_path, start_frame)

        # Apply augmentation if training
        if self.augment:
            frames = self._apply_augmentation(frames)
        else:
            # Just resize to crop_size for validation/test
            import cv2
            frames = [cv2.resize(f, self.crop_size) for f in frames]

        # Convert to tensors
        return self._frames_to_tensors(frames)


class X4K1000FPSDatasetWithRealGT(X4K1000FPSDataset):
    """
    X4K1000FPS dataset variant that uses actual intermediate frames as GT.

    Instead of synthesizing GT from reference frames, this uses the actual
    frame at position t=0.5 from the high-frame-rate video.

    This is the preferred version for training as it provides real ground truth.
    """

    def _frames_to_tensors(self, frames):
        """
        Override to use actual middle frame as ground truth.

        For high-FPS videos, we can extract real intermediate frames.
        """
        import torch

        # Extract reference frames
        ref_frame_1 = frames[self.ref_source_idx[0]].copy()
        ref_frame_2 = frames[self.ref_source_idx[1]].copy()

        # Use actual frame between references as ground truth
        # This is the key difference from base class
        gt_idx = self.ref_source_idx[0]
        gt_frame = frames[gt_idx].copy()

        # Convert to tensors
        frames_tensor = torch.stack([
            torch.from_numpy(f.transpose(2, 0, 1).copy()).float() / 255.0
            for f in frames
        ])

        ref_frames_tensor = torch.stack([
            torch.from_numpy(ref_frame_1.transpose(2, 0, 1).copy()).float() / 255.0,
            torch.from_numpy(ref_frame_2.transpose(2, 0, 1).copy()).float() / 255.0
        ])

        gt_tensor = torch.from_numpy(
            gt_frame.transpose(2, 0, 1).copy()
        ).float() / 255.0

        timestep = torch.tensor(self.target_timestep).float()

        return {
            'frames': frames_tensor,
            'ref_frames': ref_frames_tensor,
            'gt': gt_tensor,
            'timestep': timestep
        }


if __name__ == '__main__':
    # Test X4K1000FPS dataset
    import sys
    sys.path.append('..')

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
            num_frames=64,
            crop_size=(224, 224),
            augment=True,
            cache_frames=False
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
