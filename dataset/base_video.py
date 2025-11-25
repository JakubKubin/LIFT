"""
Base video dataset class with efficient frame extraction.

Supports multiple video formats and provides utilities for extracting
consecutive frames from videos for LIFT training.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
from torch.utils.data import Dataset


class VideoFrameExtractor:
    """
    Efficient video frame extraction utility.

    Extracts frames from video files without loading entire video into memory.
    """

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video information without loading frames.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info (fps, total_frames, width, height)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

        cap.release()
        return info

    @staticmethod
    def extract_frames(video_path: str,
                      start_frame: int = 0,
                      num_frames: int = 15,
                      target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Extract consecutive frames from video.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            num_frames: Number of frames to extract
            target_size: Optional (width, height) to resize frames

        Returns:
            List of frames as numpy arrays [H, W, 3] in RGB format
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if needed
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

            frames.append(frame)

        cap.release()

        # Pad with last frame if video is too short
        while len(frames) < num_frames:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                raise ValueError(f"Video {video_path} has no frames")

        return frames

    @staticmethod
    def get_valid_start_frames(video_path: str, num_frames: int = 15) -> List[int]:
        """
        Get all valid starting frame indices for extracting sequences.

        Args:
            video_path: Path to video file
            num_frames: Number of frames needed

        Returns:
            List of valid starting frame indices
        """
        info = VideoFrameExtractor.get_video_info(video_path)
        total_frames = info['total_frames']

        if total_frames < num_frames:
            # If video is too short, only one sequence possible with padding
            return [0]

        # All possible starting positions
        return list(range(total_frames - num_frames + 1))


class BaseVideoDataset(Dataset):
    """
    Base class for video frame interpolation datasets.

    Handles common functionality for extracting frames from videos.
    """

    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 num_frames: int = 15,
                 crop_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 cache_frames: bool = False,
                 input_scale: float = 1.0):
        """
        Args:
            data_root: Root directory of dataset
            mode: 'train', 'val', or 'test'
            num_frames: Number of consecutive frames to extract
            crop_size: Size for random crop (H, W)
            augment: Whether to apply data augmentation
            cache_frames: Whether to cache extracted frames (memory intensive!)
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.num_frames = num_frames
        self.is_odd = (num_frames % 2 != 0)
        self.crop_size = crop_size
        self.augment = augment and (mode == 'train')
        self.cache_frames = cache_frames
        self.input_scale = input_scale

        # Reference frames
        self.mid_idx = num_frames // 2

        if self.is_odd:
            self.ref_source_idx = [self.mid_idx - 1, self.mid_idx + 1]
        else:
            self.ref_source_idx = [self.mid_idx - 1, self.mid_idx]
        self.target_timestep = 0.5

        # Frame cache
        self._frame_cache = {} if cache_frames else None

        # To be filled by subclasses
        self.video_list = []

        # Extractor
        self.extractor = VideoFrameExtractor()

    def __len__(self):
        return len(self.video_list)

    def _get_frames_from_video(self, video_path: str, start_frame: int = 0) -> List[np.ndarray]:
        """
        Get frames from video with optional caching.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index

        Returns:
            List of frames as numpy arrays
        """
        cache_key = f"{video_path}_{start_frame}_{self.input_scale}"

        # Check cache
        if self._frame_cache is not None and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        # Extract frames
        frames = self.extractor.extract_frames(
            video_path,
            start_frame=start_frame,
            num_frames=self.num_frames,
            target_size=None  # Do not resize
        )

        # Resize if scale is not 1.0
        if self.input_scale != 1.0:
            frames = [cv2.resize(f, (0, 0), fx=self.input_scale, fy=self.input_scale, interpolation=cv2.INTER_LINEAR) for f in frames]

        # Cache if enabled
        if self._frame_cache is not None:
            self._frame_cache[cache_key] = frames

        return frames

    def _apply_augmentation(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply data augmentation to frame sequence.

        Consistent augmentation across all frames.
        """
        if not self.augment or len(frames) == 0:
            return frames

        h, w = frames[0].shape[:2]
        crop_h, crop_w = self.crop_size

        # Random crop
        if h > crop_h and w > crop_w:
            x = np.random.randint(0, h - crop_h + 1)
            y = np.random.randint(0, w - crop_w + 1)
            frames = [f[x:x+crop_h, y:y+crop_w] for f in frames]
        elif h != crop_h or w != crop_w:
            # Resize if dimensions do not match
            frames = [cv2.resize(f, (crop_w, crop_h)) for f in frames]

        # Random horizontal flip
        if random.random() < 0.5:
            frames = [f[:, ::-1].copy() for f in frames]

        # Random vertical flip
        if random.random() < 0.5:
            frames = [f[::-1].copy() for f in frames]

        # Random rotation (90, 180, 270)
        p = random.random()
        if p < 0.25:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif p < 0.5:
            rotate_code = cv2.ROTATE_180
        elif p < 0.75:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            rotate_code = None

        if rotate_code is not None:
            frames = [cv2.rotate(f, rotate_code) for f in frames]

        # Random temporal flip
        if random.random() < 0.5:
            frames = frames[::-1]

        return frames

    def _frames_to_tensors(self, frames: List[np.ndarray]) -> dict:
        """
        Convert frames to tensors and prepare output dictionary.

        Args:
            frames: List of numpy arrays [H, W, 3]

        Returns:
            Dictionary with tensors ready for model
        """
        # 1. Extract Ground Truth
        if self.is_odd:
            # Middle frame is real GT
            gt_frame = frames[self.mid_idx].copy()
        else:
            # Synthesize GT (average) for even sequences if real GT not available
            r1 = frames[self.ref_source_idx[0]].astype(np.float32)
            r2 = frames[self.ref_source_idx[1]].astype(np.float32)
            gt_frame = ((r1 + r2) / 2.0).astype(np.uint8)

        # 2. Prepare Model Inputs (Drop middle frame if odd)
        if self.is_odd:
            # Exclude the middle frame from input
            input_frames_list = frames[:self.mid_idx] + frames[self.mid_idx+1:]
        else:
            input_frames_list = frames

        # 3. Extract Reference Frames (for synthesis stage)
        # These are always the frames strictly adjacent to the target timestamp
        ref_frame_1 = frames[self.ref_source_idx[0]].copy()
        ref_frame_2 = frames[self.ref_source_idx[1]].copy()

        # 4. Tensor Conversion
        frames_tensor = torch.stack([
            torch.from_numpy(f.transpose(2, 0, 1).copy()).float() / 255.0
            for f in input_frames_list
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

    def __getitem__(self, idx):
        """
        To be implemented by subclasses.

        Should return output from _frames_to_tensors()
        """
        raise NotImplementedError("Subclasses must implement __getitem__")


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Efficiently stacks batches without unnecessary copies.
    """
    return {
        'frames': torch.stack([item['frames'] for item in batch]),
        'ref_frames': torch.stack([item['ref_frames'] for item in batch]),
        'gt': torch.stack([item['gt'] for item in batch]),
        'timestep': torch.stack([item['timestep'] for item in batch])
    }
