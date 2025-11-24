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
from typing import List, Tuple
from .base_video import BaseVideoDataset

import sys
# Ensure parent directory is in path to find configs
sys.path.append(str(Path(__file__).parent.parent))

from configs.default import Config

config = Config()

class UCF101Dataset(BaseVideoDataset):
    """
    UCF-101 action recognition dataset adapted for frame interpolation.

    Extracts 15 consecutive frames from action videos for LIFT training.
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
                 use_official_splits: bool = False,
                 split_file: str | None = None,
                 input_scale: float = 1.0,
                 max_sequences: int | None = None): # Add max_sequences
        """
        Args:
            data_root: Root directory (/data/UCF-101)
            mode: 'train', 'val', or 'test'
            num_frames: Number of frames to extract (15 for LIFT)
            crop_size: Size for random crop (H, W)
            augment: Apply augmentation
            cache_frames: Cache extracted frames (memory intensive)
            train_split: Fraction for training if not using official splits
            val_split: Fraction for validation if not using official splits
            use_official_splits: Use official UCF-101 train/test splits
            split_file: Path to official split file (trainlist01.txt, etc.)
            max_sequences: Maximum number of sequences to load (for debugging or limiting dataset size)
        """
        self.use_official_splits = use_official_splits
        self.split_file = split_file
        self.train_split = train_split
        self.val_split = val_split
        self.max_sequences = max_sequences # Store max_sequences

        super().__init__(data_root, mode, num_frames, crop_size, augment, cache_frames, input_scale)

        # Load video list
        self._load_video_list()

        print(f"UCF-101 {mode}: {len(self.video_list)} video sequences")

    def _load_official_split(self) -> List[str]:
        """Load official UCF-101 train/test split."""
        if self.split_file is None or not os.path.exists(self.split_file):
            raise ValueError(f"Split file not found: {self.split_file}")

        video_names = []
        with open(self.split_file, 'r') as f:
            for line in f:
                video_path = line.strip().split()[0]
                video_names.append(video_path)

        return video_names

    def _load_video_list(self):
        """
        Load list of videos and their valid starting frames.
        Optimized to stop loading once max_sequences is reached.
        """
        if not self.data_root.exists():
            raise ValueError(f"Dataset not found at {self.data_root}")

        # 1. Collect all video paths first (cheap operation)
        all_video_paths = []
        action_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])

        for action_dir in action_dirs:
            # We just collect paths here, don't open videos yet
            video_files = sorted(action_dir.glob('*.avi'))
            all_video_paths.extend(video_files)

        if not all_video_paths:
             raise ValueError(f"No valid video sequences found in {self.data_root}")

        # 2. Filter/Split Videos
        videos_to_process = []

        if self.use_official_splits:
            # Load official split list
            official_videos = set(self._load_official_split())

            # Filter all_video_paths to keep only those in official list
            filtered_videos = []
            for v_path in all_video_paths:
                # v_path is absolute path. We need "Action/Video.avi"
                rel_path = f"{v_path.parent.name}/{v_path.name}"
                if rel_path in official_videos:
                    filtered_videos.append(v_path)

            # Shuffle filtered videos to ensure random subset if max_sequences is used
            random.Random(config.seed).shuffle(filtered_videos)
            videos_to_process = filtered_videos

        else:
            # Custom Split: Shuffle and Split VIDEOS
            # This prevents data leakage (sequences from same video in train and val)
            random.Random(config.seed).shuffle(all_video_paths)

            total_videos = len(all_video_paths)
            train_end = int(total_videos * self.train_split)
            val_end = train_end + int(total_videos * self.val_split)

            if self.mode == 'train':
                videos_to_process = all_video_paths[:train_end]
            elif self.mode == 'val':
                videos_to_process = all_video_paths[train_end:val_end]
            else: # test
                videos_to_process = all_video_paths[val_end:]

        # 3. Extract Sequences from selected videos until limit
        self.video_list = []

        for video_path in videos_to_process:
            # Check limit before processing the video
            if self.max_sequences is not None and len(self.video_list) >= self.max_sequences:
                break

            try:
                # Get valid starting frames (This opens the video file - expensive)
                valid_starts = self.extractor.get_valid_start_frames(
                    str(video_path),
                    self.num_frames
                )

                # Add all valid sequences from this video
                for start_frame in valid_starts:
                     self.video_list.append((str(video_path), start_frame))

                     # Check limit after each addition
                     if self.max_sequences is not None and len(self.video_list) >= self.max_sequences:
                        break

            except Exception as e:
                # print(f"Warning: Error processing {video_path}: {e}")
                continue

    def __getitem__(self, idx):
        """
        Get a training sample with robust error handling.
        If a video fails to load, pick another random one.
        """
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts:
            try:
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

            except Exception as e:
                # print(f"Error loading video index {idx} ({video_path}): {e}. Retrying with new sample...")
                # Pick a new random index if the current one is corrupt
                idx = random.randint(0, len(self.video_list) - 1)
                attempts += 1

        raise RuntimeError(f"Failed to load any video after {max_attempts} attempts")


def create_ucf101_with_official_splits(data_root: str = '/data/UCF-101',
                                       splits_root: str = '/data/UCF-101/ucfTrainTestlist',
                                       max_sequences: int | None = None): # Add max_sequences
    """Helper function to create UCF-101 datasets with official train/test splits."""
    train_split_file = os.path.join(splits_root, 'trainlist01.txt')
    test_split_file = os.path.join(splits_root, 'testlist01.txt')

    train_dataset = UCF101Dataset(
        data_root=data_root,
        mode='train',
        use_official_splits=True,
        split_file=train_split_file,
        max_sequences=max_sequences # Pass max_sequences
    )

    test_dataset = UCF101Dataset(
        data_root=data_root,
        mode='test',
        use_official_splits=True,
        split_file=test_split_file,
        augment=False,
        max_sequences=max_sequences # Pass max_sequences (optional, maybe smaller for val)
    )

    return train_dataset, test_dataset


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
            use_official_splits=False,
            max_sequences=100 # Limit to 100
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

    # Test with official splits (if available)
    splits_root = os.path.join(data_root, 'ucfTrainTestlist')
    if os.path.exists(splits_root):
        print("\n2. Testing with official splits...")
        try:
            train_ds, test_ds = create_ucf101_with_official_splits(
                data_root=data_root,
                splits_root=splits_root
            )

            print(f"Train sequences: {len(train_ds)}")
            print(f"Test sequences: {len(test_ds)}")
            print("Official splits test passed!")

        except Exception as e:
            print(f"Error with official splits: {e}")
    else:
        print(f"\nOfficial splits not found at {splits_root}")
        print("Skipping official splits test")