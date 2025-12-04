"""
Base video dataset class with efficient frame extraction.
Unifies logic for UCF101, X4K1000FPS and other video datasets.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt

try:
    sys.path.append(str(Path(__file__).parent.parent))
    from configs.default import Config
    default_config = Config()
except ImportError:
    default_config = None


class VideoFrameExtractor:
    """
    Efficient video frame extraction utility.
    """
    @staticmethod
    def get_video_info(video_path: str) -> dict:
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
    def extract_frames(video_path: str, start_frame: int, num_frames: int, target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()

        # Padding
        while len(frames) < num_frames:
            if len(frames) > 0: frames.append(frames[-1].copy())
            else: raise ValueError(f"Video {video_path} has no frames")
        return frames

    @staticmethod
    def get_valid_start_frames(video_path: str, num_frames: int) -> List[int]:
        info = VideoFrameExtractor.get_video_info(video_path)
        total = info['total_frames']
        if total < num_frames: return [0]
        return list(range(total - num_frames + 1))


class BaseVideoDataset(Dataset):
    """
    Base class containing shared logic for loading, splitting, and processing video datasets.
    """

    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 num_frames: int = 15,
                 crop_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 cache_frames: bool = False,
                 input_scale: float = 1.0,
                 stride: int | None = 1,
                 max_sequences: int | None = None):

        self.data_root = Path(data_root)
        self.mode = mode
        self.num_frames = num_frames
        self.is_odd = (num_frames % 2 != 0)
        self.crop_size = crop_size
        self.augment = augment and (mode == 'train')
        self.cache_frames = cache_frames
        self.input_scale = input_scale
        self.max_sequences = max_sequences

        if stride is None:
            self.stride = 1 if mode == 'train' else num_frames
        else:
            self.stride = stride

        # Reference frames logic
        self.mid_idx = num_frames // 2
        if self.is_odd:
            self.ref_source_idx = [self.mid_idx - 1, self.mid_idx + 1]
        else:
            self.ref_source_idx = [self.mid_idx - 1, self.mid_idx]
        self.target_timestep = 0.5

        self._frame_cache = {} if cache_frames else None
        self.video_list = [] # List of tuples: (video_path, start_frame)
        self.extractor = VideoFrameExtractor()

    def __len__(self):
        return len(self.video_list)

    def _scan_for_videos(self, extensions: List[str] = ['.mp4', '.avi']) -> List[Path]:
        """Scans data_root recursively for video files."""
        if not self.data_root.exists():
            raise ValueError(f"Dataset root not found: {self.data_root}")

        all_videos = []
        for ext in extensions:
            all_videos.extend(sorted(self.data_root.rglob(f"*{ext}")))

        if not all_videos:
            raise ValueError(f"No videos found in {self.data_root} with extensions {extensions}")
        return all_videos

    def _split_videos(self, all_videos: List[Path], train_split: float, val_split: float, seed: int = 42) -> List[Path]:
        """Splits a list of video paths into train/val/test sets securely."""
        # Fix seed for consistent splits across runs
        random.Random(seed).shuffle(all_videos)

        total = len(all_videos)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)

        if self.mode == 'train':
            return all_videos[:train_end]
        elif self.mode == 'val':
            return all_videos[train_end:val_end]
        else: # test
            return all_videos[val_end:]

    def _index_sequences(self, video_paths: List[Path]):
        """
        Iterates over provided videos and generates valid (path, start_frame) tuples.
        """
        self.video_list = []

        for video_path in video_paths:
            if self.max_sequences is not None and len(self.video_list) >= self.max_sequences:
                break

            try:
                valid_starts = self.extractor.get_valid_start_frames(str(video_path), self.num_frames)
                valid_starts = valid_starts[::self.stride]

                for start in valid_starts:
                    self.video_list.append((str(video_path), start))
                    if self.max_sequences is not None and len(self.video_list) >= self.max_sequences:
                        break
            except Exception as e:
                continue

    def _auto_build_dataset(self, extensions: List[str], train_split: float = 0.8, val_split: float = 0.1):
        """Helper to run the full Scan -> Split -> Index pipeline."""
        print(f"Scanning {self.data_root} for {extensions}...")
        all_videos = self._scan_for_videos(extensions)

        seed = default_config.seed if default_config else 42
        selected_videos = self._split_videos(all_videos, train_split, val_split, seed=seed)

        print(f"Indexing sequences from {len(selected_videos)} videos (Mode: {self.mode}, Stride: {self.stride})...")
        self._index_sequences(selected_videos)
        print(f"Dataset ready: {len(self.video_list)} sequences.")

    def __getitem__(self, idx):
        """Unified __getitem__ with robust retry logic."""
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts:
            try:
                video_path, start_frame = self.video_list[idx]
                frames = self._get_frames_from_video(video_path, start_frame)

                if self.augment:
                    frames = self._apply_augmentation(frames)
                else:
                    # Explicit augmentation bypass (handled by safety resize later)
                    pass 

                return self._frames_to_tensors(frames)

            except Exception as e:
                # print(f"Error loading {idx}: {e}. Retrying...")
                idx = random.randint(0, len(self.video_list) - 1)
                attempts += 1

        raise RuntimeError(f"Failed to load data after {max_attempts} attempts.")

    def _get_frames_from_video(self, video_path: str, start_frame: int) -> List[np.ndarray]:
        cache_key = f"{video_path}_{start_frame}_{self.input_scale}"
        if self._frame_cache is not None and cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        target_h, target_w = self.crop_size
        target_size_cv2 = (target_w, target_h)

        frames = self.extractor.extract_frames(
            video_path, start_frame, self.num_frames, target_size=target_size_cv2
        )

        if self.input_scale != 1.0:
            frames = [cv2.resize(f, (0, 0), fx=self.input_scale, fy=self.input_scale, interpolation=cv2.INTER_LINEAR) for f in frames]

        if self._frame_cache is not None:
            self._frame_cache[cache_key] = frames
        return frames

    def _frames_to_tensors(self, frames: List[np.ndarray]) -> dict:
        # Safety Resize (Crucial for validation mixing orientations)
        target_h, target_w = self.crop_size
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            if h != target_h or w != target_w:
                frames = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames]

        # 1. Ground Truth
        if self.is_odd:
            gt_frame = frames[self.mid_idx].copy()
        else:
            r1 = frames[self.ref_source_idx[0]].astype(np.float32)
            r2 = frames[self.ref_source_idx[1]].astype(np.float32)
            gt_frame = ((r1 + r2) / 2.0).astype(np.uint8)

        # 2. Input Frames
        if self.is_odd:
            input_frames_list = frames[:self.mid_idx] + frames[self.mid_idx+1:]
        else:
            input_frames_list = frames

        # 3. Reference Frames
        ref_frame_1 = frames[self.ref_source_idx[0]].copy()
        ref_frame_2 = frames[self.ref_source_idx[1]].copy()

        # 4. To Tensor
        frames_tensor = torch.stack([torch.from_numpy(f.transpose(2, 0, 1).copy()).float() / 255.0 for f in input_frames_list])
        ref_frames_tensor = torch.stack([
            torch.from_numpy(ref_frame_1.transpose(2, 0, 1).copy()).float() / 255.0,
            torch.from_numpy(ref_frame_2.transpose(2, 0, 1).copy()).float() / 255.0
        ])
        gt_tensor = torch.from_numpy(gt_frame.transpose(2, 0, 1).copy()).float() / 255.0
        timestep = torch.tensor(self.target_timestep).float()

        return {'frames': frames_tensor, 'ref_frames': ref_frames_tensor, 'gt': gt_tensor, 'timestep': timestep}

    def _apply_augmentation(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # (Tutaj wklej swoją istniejącą funkcję _apply_augmentation - jest bez zmian)
        if not self.augment or len(frames) == 0: return frames
        h, w = frames[0].shape[:2]
        crop_h, crop_w = self.crop_size

        # Crop logic
        if h > crop_h and w > crop_w:
            x = np.random.randint(0, h - crop_h + 1)
            y = np.random.randint(0, w - crop_w + 1)
            frames = [f[x:x+crop_h, y:y+crop_w] for f in frames]
        elif h != crop_h or w != crop_w:
             frames = [cv2.resize(f, (crop_w, crop_h)) for f in frames]

        # Flips & Rotations
        if random.random() < 0.5: frames = [f[:, ::-1].copy() for f in frames]
        if random.random() < 0.5: frames = [f[::-1].copy() for f in frames]
        p = random.random()
        if p < 0.25: frames = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]
        elif p < 0.5: frames = [cv2.rotate(f, cv2.ROTATE_180) for f in frames]
        elif p < 0.75: frames = [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]
        if random.random() < 0.5: frames = frames[::-1]

        return frames

    def visualize_samples(self, output_dir: str, num_samples: int = 5):
        """
        Generuje obrazki porównawcze 'Original' vs 'Augmented' dla losowych próbek.
        Zapisuje je w podanym folderze.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating visualization for {num_samples} samples in '{output_dir}'...")

        # Wybierz losowe indeksy
        total_samples = len(self.video_list)
        if total_samples == 0:
            print("Warning: Dataset is empty, cannot visualize.")
            return

        indices = random.sample(range(total_samples), min(total_samples, num_samples))

        for i, idx in enumerate(indices):
            video_path, start_frame = self.video_list[idx]
            
            # 1. Pobierz oryginał (Before)
            # Używamy copy(), aby nie modyfikować cache
            frames_orig = [f.copy() for f in self._get_frames_from_video(video_path, start_frame)]
            
            # 2. Zastosuj augmentację (After)
            # Wymuszamy augmentację = True, aby zobaczyć efekt nawet w trybie walidacji
            frames_aug = [f.copy() for f in frames_orig]
            
            old_aug_state = self.augment
            self.augment = True  # Force Enable
            try:
                frames_aug = self._apply_augmentation(frames_aug)
            finally:
                self.augment = old_aug_state # Restore state

            # 3. Rysowanie (Start, Middle, End)
            # Jeśli augmentacja zmieniła liczbę klatek (np. odwrócenie czasu), indeksy mogą się różnić
            # Ale dla LIFT liczba klatek jest stała.
            
            indices_to_show = [0, len(frames_orig)//2, len(frames_orig)-1]
            labels = ["First", "Middle (GT)", "Last"]
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            
            # Rząd 1: Oryginał
            for col, frame_idx in enumerate(indices_to_show):
                ax = axes[0, col]
                if frame_idx < len(frames_orig):
                    ax.imshow(frames_orig[frame_idx])
                    ax.set_title(f"Original: {labels[col]}\n{frames_orig[frame_idx].shape[:2]}")
                ax.axis('off')

            # Rząd 2: Augmentacja
            for col, frame_idx in enumerate(indices_to_show):
                ax = axes[1, col]
                if frame_idx < len(frames_aug):
                    ax.imshow(frames_aug[frame_idx])
                    ax.set_title(f"Augmented: {labels[col]}\n{frames_aug[frame_idx].shape[:2]}")
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"vis_sample_{i}_{Path(video_path).stem}.png")
            plt.savefig(save_path)
            plt.close()
        
        print(f"Visualization saved to {output_dir}")

def collate_fn(batch):
    return {
        'frames': torch.stack([item['frames'] for item in batch]),
        'ref_frames': torch.stack([item['ref_frames'] for item in batch]),
        'gt': torch.stack([item['gt'] for item in batch]),
        'timestep': torch.stack([item['timestep'] for item in batch])
    }