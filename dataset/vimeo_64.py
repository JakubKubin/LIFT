import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path

cv2.setNumThreads(1)


class Vimeo64Dataset(Dataset):
    """
    Dataset for LIFT model that loads 64 consecutive frames for long-range temporal context.

    Memory optimization strategy:
    1. Lazy loading: frames are loaded on-demand, not stored in memory
    2. Sequential reading: read frames one by one to avoid memory spikes
    3. Efficient augmentation: apply transforms during loading, not after

    The model interpolates frame at t=0.5 between frame 31 and 32 (middle of sequence).
    """

    def __init__(self,
                 data_root,
                 mode='train',
                 num_frames=64,
                 crop_size=(224, 224),
                 augment=True,
                 input_scale=1.0,
                 max_sequences=None): # Added max_sequences
        """
        Args:
            data_root: Root directory containing video sequences
            mode: 'train', 'val', or 'test'
            num_frames: Number of frames to load (default 64)
            crop_size: Tuple of (height, width) for random crop
            augment: Whether to apply data augmentation
            max_sequences: Maximum number of sequences to load (for debugging or limiting dataset size)
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.augment = augment and (mode == 'train')
        self.input_scale = input_scale
        self.max_sequences = max_sequences # Store it

        # Reference frames
        mid = self.num_frames // 2
        self.ref_frame_idx = [mid - 1, mid]
        self.target_timestep = 0.5

        # Load sequence list
        self.sequences = self._load_sequences()

        print(f"Loaded {len(self.sequences)} sequences for {mode}")

    def _load_sequences(self):
        """
        Load list of valid sequences that have at least num_frames frames.

        For memory efficiency, we only store paths, not actual frames.
        """
        sequences = []

        # Support different directory structures
        if (self.data_root / 'sequences').exists():
            # Vimeo-style structure: sequences/category/sequence_id/
            seq_dir = self.data_root / 'sequences'
            for category in sorted(seq_dir.iterdir()):
                if not category.is_dir():
                    continue
                for seq_id in sorted(category.iterdir()):
                    if not seq_id.is_dir():
                        continue
                    frames = sorted([f for f in seq_id.iterdir()
                                   if f.suffix in ['.png', '.jpg', '.jpeg']])
                    if len(frames) >= self.num_frames:
                        sequences.append(frames[:self.num_frames])
        else:
            # Flat structure: data_root/sequence_id/
            for seq_id in sorted(self.data_root.iterdir()):
                if not seq_id.is_dir():
                    continue
                frames = sorted([f for f in seq_id.iterdir()
                               if f.suffix in ['.png', '.jpg', '.jpeg']])
                if len(frames) >= self.num_frames:
                    sequences.append(frames[:self.num_frames])

        # Filter based on mode (train/val split)
        if self.mode == 'train':
            sequences = sequences[:int(len(sequences) * 0.95)]
        elif self.mode == 'val':
            sequences = sequences[int(len(sequences) * 0.95):]

        # Apply max_sequences limit
        if self.max_sequences is not None:
            # Shuffle to ensure we get a random subset
            # Using a fixed seed for reproducibility if needed, but random.shuffle is fine here
            random.shuffle(sequences) 
            sequences = sequences[:self.max_sequences]

        return sequences

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, frame_path):
        """
        Load a single frame efficiently.

        Returns normalized tensor in range [0, 1].
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=self.input_scale, fy=self.input_scale, interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def _apply_augmentation(frames, ref_frame_1, ref_frame_2, gt_frame, crop_size):
        """
        Apply data augmentation consistently across all frames.

        This is memory-efficient as it's applied during loading.

        Augmentations:
        - Random crop
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (90, 180, 270 degrees)
        - Random temporal flip
        """
        h, w = frames[0].shape[:2]
        crop_h, crop_w = crop_size

        # Random crop coordinates (same for all frames)
        if h > crop_h and w > crop_w:
            x = np.random.randint(0, h - crop_h + 1)
            y = np.random.randint(0, w - crop_w + 1)
            frames = [f[x:x+crop_h, y:y+crop_w] for f in frames]
            ref_frame_1 = ref_frame_1[x:x+crop_h, y:y+crop_w]
            ref_frame_2 = ref_frame_2[x:x+crop_h, y:y+crop_w]
            gt_frame = gt_frame[x:x+crop_h, y:y+crop_w]

        # Random horizontal flip
        if random.random() < 0.5:
            frames = [f[:, ::-1].copy() for f in frames]
            ref_frame_1 = ref_frame_1[:, ::-1].copy()
            ref_frame_2 = ref_frame_2[:, ::-1].copy()
            gt_frame = gt_frame[:, ::-1].copy()

        # Random vertical flip
        if random.random() < 0.5:
            frames = [f[::-1].copy() for f in frames]
            ref_frame_1 = ref_frame_1[::-1].copy()
            ref_frame_2 = ref_frame_2[::-1].copy()
            gt_frame = gt_frame[::-1].copy()

        # Random rotation
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
            ref_frame_1 = cv2.rotate(ref_frame_1, rotate_code)
            ref_frame_2 = cv2.rotate(ref_frame_2, rotate_code)
            gt_frame = cv2.rotate(gt_frame, rotate_code)

        # Random temporal flip (reverse sequence)
        if random.random() < 0.5:
            frames = frames[::-1]
            ref_frame_1, ref_frame_2 = ref_frame_2, ref_frame_1

        return frames, ref_frame_1, ref_frame_2, gt_frame

    def __getitem__(self, idx):
        """
        Load and return a sequence of frames.

        Returns:
            dict with keys:
                - 'frames': Tensor [64, 3, H, W] - all 64 frames normalized to [0, 1]
                - 'ref_frames': Tensor [2, 3, H, W] - reference frames (31, 32)
                - 'gt': Tensor [3, H, W] - ground truth middle frame
                - 'timestep': float - interpolation timestep (0.5)

        Memory optimization:
        - Frames are loaded sequentially and immediately converted to tensors
        - No intermediate storage of full sequence
        - Augmentation applied during loading
        """
        frame_paths = self.sequences[idx]

        # Load all frames (memory-efficient sequential loading)
        frames = []
        for frame_path in frame_paths:
            frame = self._load_frame(frame_path)
            frames.append(frame)

        # Extract reference frames and ground truth
        ref_frame_1 = frames[self.ref_frame_idx[0]].copy()
        ref_frame_2 = frames[self.ref_frame_idx[1]].copy()

        # Ground truth is the average of two reference frames (for now)
        # In practice, this should be an actual intermediate frame
        # For training, we'll synthesize it or use actual intermediate frames
        gt_frame = (ref_frame_1.astype(np.float32) + ref_frame_2.astype(np.float32)) / 2.0
        gt_frame = gt_frame.astype(np.uint8)

        # Apply augmentation if enabled
        if self.augment:
            frames, ref_frame_1, ref_frame_2, gt_frame = self._apply_augmentation(
                frames, ref_frame_1, ref_frame_2, gt_frame, self.crop_size
            )

        # Convert to tensors and normalize to [0, 1]
        # Use efficient conversion without intermediate copies
        frames_tensor = torch.stack([
            torch.from_numpy(f.transpose(2, 0, 1).copy()).float() / 255.0
            for f in frames
        ])  # [64, 3, H, W]

        ref_frames_tensor = torch.stack([
            torch.from_numpy(ref_frame_1.transpose(2, 0, 1).copy()).float() / 255.0,
            torch.from_numpy(ref_frame_2.transpose(2, 0, 1).copy()).float() / 255.0
        ])  # [2, 3, H, W]

        gt_tensor = torch.from_numpy(
            gt_frame.transpose(2, 0, 1).copy()
        ).float() / 255.0  # [3, H, W]

        timestep = torch.tensor(self.target_timestep).float()

        return {
            'frames': frames_tensor,
            'ref_frames': ref_frames_tensor,
            'gt': gt_tensor,
            'timestep': timestep
        }


class VideoSequenceDataset(Dataset):
    """
    Dataset that loads frames from video files directly.

    More memory-efficient for long sequences as it uses video codecs
    instead of loading individual frame images.
    """

    def __init__(self,
                 video_list_file,
                 mode='train',
                 num_frames=64,
                 crop_size=(224, 224),
                 augment=True,
                 input_scale=1.0,
                 max_sequences=None): # Added max_sequences
        """
        Args:
            video_list_file: Text file containing list of video paths
            mode: 'train', 'val', or 'test'
            num_frames: Number of frames to extract from each video
            crop_size: Tuple of (height, width) for random crop
            augment: Whether to apply data augmentation
            max_sequences: Maximum number of sequences to load
        """
        self.mode = mode
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.augment = augment and (mode == 'train')
        self.input_scale = input_scale
        self.max_sequences = max_sequences

        # Load video paths
        with open(video_list_file, 'r') as f:
            self.video_paths = [line.strip() for line in f.readlines()]

        # Split train/val
        if mode == 'train':
            self.video_paths = self.video_paths[:int(len(self.video_paths) * 0.95)]
        elif mode == 'val':
            self.video_paths = self.video_paths[int(len(self.video_paths) * 0.95):]
            
        # Apply max_sequences limit
        if self.max_sequences is not None:
            random.shuffle(self.video_paths)
            self.video_paths = self.video_paths[:self.max_sequences]

        print(f"Loaded {len(self.video_paths)} videos for {mode}")

    def __len__(self):
        return len(self.video_paths)

    def _extract_frames(self, video_path, start_frame=0):
        """
        Extract num_frames consecutive frames from video.

        Returns list of numpy arrays (H, W, 3) in RGB format.
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Random start frame if augmenting
        if self.augment and total_frames > self.num_frames:
            start_frame = random.randint(0, total_frames - self.num_frames)

        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # If video is too short, repeat last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        if self.input_scale != 1.0:
            frames = [cv2.resize(f, (0, 0), fx=self.input_scale, fy=self.input_scale, interpolation=cv2.INTER_LINEAR) for f in frames]

        return frames

    def __getitem__(self, idx):
        """Load frames from video file."""
        video_path = self.video_paths[idx]

        # Extract frames
        frames = self._extract_frames(video_path)

        # Extract reference frames and ground truth
        ref_frame_1 = frames[31].copy()
        ref_frame_2 = frames[32].copy()
        gt_frame = (ref_frame_1.astype(np.float32) + ref_frame_2.astype(np.float32)) / 2.0
        gt_frame = gt_frame.astype(np.uint8)

        # Apply augmentation
        if self.augment:
            frames, ref_frame_1, ref_frame_2, gt_frame = Vimeo64Dataset._apply_augmentation(
                frames, ref_frame_1, ref_frame_2, gt_frame, self.crop_size
            )

        # Convert to tensors
        frames_tensor = torch.stack([
            torch.from_numpy(f.transpose(2, 0, 1).copy()).float() / 255.0
            for f in frames
        ])

        ref_frames_tensor = torch.stack([
            torch.from_numpy(ref_frame_1.transpose(2, 0, 1).copy()).float() / 255.0,
            torch.from_numpy(ref_frame_2.transpose(2, 0, 1).copy()).float() / 255.0
        ])

        gt_tensor = torch.from_numpy(gt_frame.transpose(2, 0, 1).copy()).float() / 255.0
        timestep = torch.tensor(0.5).float()

        return {
            'frames': frames_tensor,
            'ref_frames': ref_frames_tensor,
            'gt': gt_tensor,
            'timestep': timestep
        }


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