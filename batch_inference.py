import os
import cv2
import torch
import argparse
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import LIFT
from configs.default import Config


class VideoFrameExtractor:
    """Efficient video frame extraction utility."""

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'fps': 0, 'total_frames': 0, 'width': 0, 'height': 0, 'valid': False}
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'valid': True
        }
        cap.release()
        return info

    @staticmethod
    def extract_frames(video_path: str, start_frame: int, num_frames: int,
                       target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        cap.release()
        return frames


class BatchEvaluator:
    """Handles metrics computation for batch evaluation."""

    def __init__(self, device: torch.device):
        self.device = device
        self._init_lpips()
        self._init_ssim_window()

    def _init_lpips(self):
        import warnings
        warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
        warnings.filterwarnings("ignore", message="Arguments other than a weight enum")

        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: LPIPS library not found. Metric will be 0.")
            self.lpips_model = None

    def _init_ssim_window(self, window_size: int = 11):
        from math import exp
        import torch.nn.functional as F

        def gaussian(size, sigma):
            gauss = torch.Tensor([exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
            return gauss / gauss.sum()

        _1D = gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        self.ssim_window_base = _2D
        self.ssim_window = None

    def _get_ssim_window(self, channels: int, device: torch.device, dtype: torch.dtype):
        if self.ssim_window is None or self.ssim_window.device != device:
            self.ssim_window = self.ssim_window_base.expand(channels, 1, -1, -1).contiguous().to(device).type(dtype)
        return self.ssim_window

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return -10 * torch.log10(mse).item()

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        import torch.nn.functional as F

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        padd = window_size // 2

        window = self._get_ssim_window(pred.size(1), pred.device, pred.dtype)

        mu1 = F.conv2d(pred, window, padding=padd, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=padd, groups=pred.size(1))

        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=padd, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=padd, groups=pred.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=padd, groups=pred.size(1)) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if self.lpips_model is None:
            return 0.0
        with torch.no_grad():
            return self.lpips_model(pred * 2 - 1, target * 2 - 1).mean().item()

    def compute_all(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred = pred.float().clamp(0, 1)
        target = target.float().clamp(0, 1)

        return {
            'PSNR': self.compute_psnr(pred, target),
            'SSIM': self.compute_ssim(pred, target),
            'LPIPS': self.compute_lpips(pred, target)
        }


class BatchInference:
    """Main batch inference pipeline."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda',
                 num_frames: int = 15, crop_size: Tuple[int, int] = (256, 256)):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.mid_idx = num_frames // 2

        self.config = Config()
        self.model = self._load_model(checkpoint_path)
        self.evaluator = BatchEvaluator(self.device)
        self.extractor = VideoFrameExtractor()

        self.results = []
        self.skipped = []

    def _load_model(self, checkpoint_path: str) -> LIFT:
        print(f"\n{'='*60}")
        print("LOADING MODEL CHECKPOINT")
        print(f"{'='*60}")
        print(f"Path: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'config' in checkpoint:
            print("\nRestoring config from checkpoint:")
            for key, value in checkpoint['config'].items():
                if hasattr(self.config, key):
                    old_val = getattr(self.config, key)
                    setattr(self.config, key, value)

        print(f"\nInitializing LIFT model (num_frames={self.config.num_frames})...")
        model = LIFT(self.config).to(self.device)

        model_state = model.state_dict()

        if 'model_state_dict' in checkpoint:
            checkpoint_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            checkpoint_state = checkpoint['state_dict']
        else:
            checkpoint_state = checkpoint

        new_state = {}
        for k, v in checkpoint_state.items():
            clean_key = k.replace('module.', '')
            if clean_key in model_state:
                if v.shape == model_state[clean_key].shape:
                    new_state[clean_key] = v

        model.load_state_dict(new_state, strict=False)
        model.eval()

        epoch = checkpoint.get('epoch', 'unknown')
        print(f"\nCheckpoint Epoch: {epoch}")
        print(f"{'='*60}\n")

        return model

    def _create_error_map(self, pred: torch.Tensor, target: torch.Tensor, amplify: float = 5.0) -> np.ndarray:
        """
        Create amplified error visualization using Blue->Red gradient.
        Returns: Numpy array [H, W, 3] uint8 ready for plotting.
        """
        # Obliczanie błędu na tensorach (według Twojego wzoru)
        # [B, 3, H, W] -> średnia po kanałach -> [B, 1, H, W]
        diff = torch.abs(pred - target).mean(dim=1, keepdim=True) * amplify
        diff = diff.clamp(0, 1)

        # Tworzenie mapy kolorów: R=błąd, G=0, B=1-błąd (Niebieski -> Czerwony)
        # Wynik: [B, 3, H, W]
        error_map_tensor = torch.cat([diff, torch.zeros_like(diff), 1 - diff], dim=1)

        # Konwersja do numpy dla matplotlib: [H, W, 3]
        error_map_np = (error_map_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return error_map_np

    def sample_sequences_lazy(self, data_dir: str, num_sequences: int, seed: int = 44) -> List[Tuple[Path, int]]:
        """
        Robustly samples sequences by checking directory structure depth.
        Supports:
        1. Flat video files
        2. UCF-101 style: Root/Category/Video.avi
        3. Vimeo style: Root/Sequence/Folder/*.png
        """
        random.seed(seed)
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Directory not found: {data_dir}")

        print(f"Sampling {num_sequences} sequences from {data_dir}...")

        # 1. Collect potential containers (Files or Directories)
        # We look at depth 1 and 2 to find where the "items" are.

        candidates = []

        # Check depth 0 (files directly in root)
        direct_files = list(data_path.glob('*.avi')) + list(data_path.glob('*.mp4')) + list(data_path.glob('*.mkv'))
        if direct_files:
            candidates.extend(direct_files)

        # Check depth 1 (folders in root)
        subdirs = [p for p in data_path.iterdir() if p.is_dir()]

        # Shuffle subdirs to avoid scanning everything linearly
        random.shuffle(subdirs)

        # Traverse subdirs to find videos or frame folders
        for sd in subdirs:
            if len(candidates) > num_sequences * 2:
                break

            # Check for videos inside subdir (UCF-101 style: Action/Video.avi)
            vids = list(sd.glob('*.avi')) + list(sd.glob('*.mp4'))
            if vids:
                candidates.extend(vids)
                continue

            # Check for frames inside subdir (Vimeo simple style)
            frames = list(sd.glob('*.png')) + list(sd.glob('*.jpg'))
            if frames:
                candidates.append(sd) # This dir is a sequence
                continue

            # Check depth 2 (Vimeo complex style: Seq/001/*.png)
            subsubdirs = [p for p in sd.iterdir() if p.is_dir()]
            if subsubdirs:
                for ssd in subsubdirs:
                    frames = list(ssd.glob('*.png')) + list(ssd.glob('*.jpg'))
                    if frames:
                         candidates.append(ssd)

        if not candidates:
            raise ValueError(f"No video files or frame directories found in {data_dir}")

        random.shuffle(candidates)

        # 2. Validate and Select
        valid_samples = []

        print(f"Validating candidates...")
        for item in tqdm(candidates, desc="Checking"):
            if len(valid_samples) >= num_sequences:
                break

            try:
                # Case A: Directory of frames
                if item.is_dir():
                    frames = sorted(list(item.glob('*.png')) + list(item.glob('*.jpg')))
                    if len(frames) >= self.num_frames:
                        start = random.randint(0, len(frames) - self.num_frames)
                        valid_samples.append((item, start))

                # Case B: Video file
                elif item.is_file():
                    info = self.extractor.get_video_info(str(item))
                    if info['valid'] and info['total_frames'] >= self.num_frames:
                        start = random.randint(0, info['total_frames'] - self.num_frames)
                        valid_samples.append((item, start))

            except Exception:
                continue

        if len(valid_samples) < num_sequences:
            print(f"Warning: Only found {len(valid_samples)} valid sequences.")

        return valid_samples

    def extract_and_prepare(self, video_path: Path, start_frame: int) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        """Extract frames and prepare tensors."""
        target_w, target_h = self.crop_size

        frames = self.extractor.extract_frames(
            str(video_path), start_frame, self.num_frames,
            target_size=(target_w, target_h)
        )

        if len(frames) < self.num_frames:
            raise ValueError(f"Could not extract {self.num_frames} frames")

        gt_frame = frames[self.mid_idx]
        input_frames = frames[:self.mid_idx] + frames[self.mid_idx + 1:]

        frames_tensor = torch.stack([
            torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
            for f in input_frames
        ]).unsqueeze(0)

        gt_tensor = torch.from_numpy(
            gt_frame.transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0

        return frames_tensor, gt_tensor, frames

    @torch.no_grad()
    def interpolate(self, frames: torch.Tensor, timestep: float = 0.5) -> torch.Tensor:
        """Run model inference."""
        frames = frames.to(self.device)
        output = self.model.inference(frames, timestep=timestep)
        return output

    def run_batch_evaluation(self, data_dir: str, num_sequences: int,
                             output_dir: str, seed: int = 42) -> Dict:
        """Main evaluation loop."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        sequences = self.sample_sequences_lazy(data_dir, num_sequences, seed)

        print(f"\n{'='*60}")
        print("STARTING BATCH EVALUATION")
        print(f"{'='*60}\n")

        all_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}

        for idx, (video_path, start_frame) in enumerate(tqdm(sequences, desc="Evaluating")):
            try:
                frames_tensor, gt_tensor, raw_frames = self.extract_and_prepare(
                    video_path, start_frame
                )

                pred_tensor = self.interpolate(frames_tensor)

                metrics = self.evaluator.compute_all(
                    pred_tensor.to(self.device),
                    gt_tensor.to(self.device)
                )

                for k, v in metrics.items():
                    all_metrics[k].append(v)

                result = {
                    'idx': idx,
                    'video': str(video_path.relative_to(data_dir)),
                    'start_frame': start_frame,
                    'metrics': metrics
                }
                self.results.append(result)

                if idx < 20 or idx % (len(sequences) // 10 + 1) == 0:
                    self._save_visualization(
                        raw_frames, pred_tensor, gt_tensor, metrics,
                        viz_dir / f"sample_{idx:04d}.png",
                        video_path.name, start_frame
                    )

            except Exception as e:
                self.skipped.append({
                    'video': str(video_path),
                    'start_frame': start_frame,
                    'reason': str(e)
                })
                continue

        summary = self._compute_summary(all_metrics)

        self._save_summary_visualization(all_metrics, summary, output_path / 'summary.png')

        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_dir': data_dir,
                'num_sequences': num_sequences,
                'num_frames': self.num_frames,
                'crop_size': self.crop_size,
                'seed': seed
            },
            'summary': summary,
            'results': self.results,
            'skipped': self.skipped
        }

        with open(output_path / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)

        self._print_summary(summary)

        return report

    def _compute_summary(self, all_metrics: Dict[str, List[float]]) -> Dict:
        """Compute summary statistics."""
        summary = {}

        for metric_name, values in all_metrics.items():
            if len(values) == 0:
                continue

            arr = np.array(values)
            summary[metric_name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'count': len(values)
            }

        return summary

    def _save_visualization(self, raw_frames: List[np.ndarray],
                            pred_tensor: torch.Tensor, gt_tensor: torch.Tensor,
                            metrics: Dict[str, float], save_path: Path,
                            video_name: str, start_frame: int):
        """Create comprehensive visualization: Inputs -> Result -> Error."""

        pred_np = (pred_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_np = (gt_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        error_map = self._create_error_map(pred_tensor, gt_tensor, amplify=5.0)

        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1.2, 0.2], wspace=0.05, hspace=0.15)

        idx_prev = self.mid_idx - 1  # 6
        idx_next = self.mid_idx + 1  # 8

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(raw_frames[idx_prev])
        ax1.set_title(f'Input Frame {idx_prev}', fontsize=16)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(raw_frames[idx_next])
        ax2.set_title(f'Input Frame {idx_next}', fontsize=16)
        ax2.axis('off')

        ax_gt = fig.add_subplot(gs[1, 0])
        ax_gt.imshow(gt_np)
        ax_gt.set_title(f'Ground Truth (Frame {self.mid_idx})', fontsize=16, fontweight='bold')
        ax_gt.axis('off')

        ax_pred = fig.add_subplot(gs[1, 1])
        ax_pred.imshow(pred_np)
        ax_pred.set_title('LIFT Interpolated', fontsize=16, fontweight='bold')
        ax_pred.axis('off')

        ax_err = fig.add_subplot(gs[2, :])
        ax_err.imshow(error_map)
        ax_err.set_title('Error Map (x5)', fontsize=16, fontweight='bold')
        ax_err.axis('off')

        ax_metrics = fig.add_subplot(gs[3, :])
        ax_metrics.axis('off')

        metrics_text = (
            f"Video: {video_name}   |   Start Frame: {start_frame}\n"
            f"PSNR: {metrics['PSNR']:.2f} dB   |   "
            f"SSIM: {metrics['SSIM']:.4f}   |   "
            f"LPIPS: {metrics['LPIPS']:.4f}"
        )

        psnr_color = '#27ae60' if metrics['PSNR'] > 30 else '#f39c12' if metrics['PSNR'] > 25 else '#e74c3c'

        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                        fontsize=16, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa',
                                  edgecolor=psnr_color, linewidth=2))

        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def _save_summary_visualization(self, all_metrics: Dict[str, List[float]],
                                    summary: Dict, save_path: Path):
        """Create summary visualization with distributions."""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        colors = {'PSNR': '#3498db', 'SSIM': '#2ecc71', 'LPIPS': '#e74c3c'}

        for idx, (metric_name, values) in enumerate(all_metrics.items()):
            if len(values) == 0:
                continue

            ax = axes[0, idx]
            ax.hist(values, bins=30, color=colors[metric_name], edgecolor='black', alpha=0.7)
            ax.axvline(summary[metric_name]['mean'], color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {summary[metric_name]['mean']:.3f}")
            ax.axvline(summary[metric_name]['median'], color='orange', linestyle=':',
                       linewidth=2, label=f"Median: {summary[metric_name]['median']:.3f}")
            ax.set_title(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel(metric_name)
            ax.set_ylabel('Count')
            ax.legend()

        ax_box = axes[1, 0]
        psnr_data = all_metrics['PSNR']
        # Fixed: Updated parameter name for matplotlib compatibility
        ax_box.boxplot([psnr_data], tick_labels=['PSNR'])
        ax_box.set_ylabel('dB')
        ax_box.set_title('PSNR Box Plot', fontweight='bold')

        ax_scatter = axes[1, 1]
        if len(all_metrics['PSNR']) > 0 and len(all_metrics['SSIM']) > 0:
            ax_scatter.scatter(all_metrics['PSNR'], all_metrics['SSIM'],
                               alpha=0.5, c=all_metrics['LPIPS'], cmap='RdYlGn_r')
            ax_scatter.set_xlabel('PSNR (dB)')
            ax_scatter.set_ylabel('SSIM')
            ax_scatter.set_title('PSNR vs SSIM (color=LPIPS)', fontweight='bold')

        ax_summary = axes[1, 2]
        ax_summary.axis('off')

        summary_text = "EVALUATION SUMMARY\n" + "="*30 + "\n\n"
        for metric_name, stats in summary.items():
            summary_text += f"{metric_name}:\n"
            summary_text += f"  Mean:   {stats['mean']:.4f}\n"
            summary_text += f"  Std:    {stats['std']:.4f}\n"
            summary_text += f"  Min:    {stats['min']:.4f}\n"
            summary_text += f"  Max:    {stats['max']:.4f}\n"
            summary_text += f"  Median: {stats['median']:.4f}\n\n"

        summary_text += f"Total Evaluated: {summary.get('PSNR', {}).get('count', 0)}\n"
        summary_text += f"Skipped: {len(self.skipped)}"

        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                        fontsize=11, family='monospace', va='top',
                        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _print_summary(self, summary: Dict):
        """Print formatted summary to console."""
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}\n")

        for metric_name, stats in summary.items():
            print(f"{metric_name}:")
            print(f"  Mean:   {stats['mean']:.4f}")
            print(f"  Std:    {stats['std']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()

        print(f"Total Evaluated: {summary.get('PSNR', {}).get('count', 0)}")
        print(f"Skipped Sequences: {len(self.skipped)}")
        print(f"{'='*60}\n")


def test_single_inference(pipeline: BatchInference, data_dir: str) -> bool:
    """Quick test with a single sample to verify everything works."""
    print(f"\n{'='*60}")
    print("RUNNING QUICK TEST (1 sample)")
    print(f"{'='*60}\n")

    try:
        samples = pipeline.sample_sequences_lazy(data_dir, num_sequences=1)
        if not samples:
            print("ERROR: No videos found!")
            return False

        test_video, test_start = samples[0]

        print(f"Test video: {test_video.name}")
        print(f"Start frame: {test_start}")

        frames_tensor, gt_tensor, raw_frames = pipeline.extract_and_prepare(
            test_video, test_start
        )

        print(f"Input tensor shape: {frames_tensor.shape}")
        print(f"GT tensor shape: {gt_tensor.shape}")

        print("Running inference...")
        pred_tensor = pipeline.interpolate(frames_tensor)
        print(f"Output tensor shape: {pred_tensor.shape}")

        print("Computing metrics...")
        metrics = pipeline.evaluator.compute_all(
            pred_tensor.to(pipeline.device),
            gt_tensor.to(pipeline.device)
        )

        print(f"\nTest Results:")
        print(f"  PSNR:  {metrics['PSNR']:.2f} dB")
        print(f"  SSIM:  {metrics['SSIM']:.4f}")
        print(f"  LPIPS: {metrics['LPIPS']:.4f}")

        print(f"\n{'='*60}")
        print("TEST PASSED! Ready for batch evaluation.")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_checkpoints(directory: str = '.') -> List[str]:
    """Find all .pth files in directory."""
    pth_files = list(Path(directory).rglob('*.pth'))
    return sorted([str(p) for p in pth_files])


def main():
    parser = argparse.ArgumentParser(
        description='LIFT Batch Inference with Metrics Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test first
  python batch_inference.py --checkpoint model.pth --data_dir ./data/UCF-101 --test

  # Full evaluation
  python batch_inference.py --checkpoint model.pth --data_dir ./data/UCF-101 --num_sequences 100

  # Find checkpoints in directory
  python batch_inference.py --find_checkpoints ./checkpoints

  # Custom settings
  python batch_inference.py --checkpoint best_model.pth --data_dir ./videos \\
      --num_sequences 200 --output ./results --crop_size 256 256
        """
    )

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing videos')
    parser.add_argument('--num_sequences', type=int, default=100,
                        help='Number of sequences to evaluate (default: 100)')
    parser.add_argument('--output', type=str, default='./batch_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--num_frames', type=int, default=15,
                        help='Number of frames per sequence (default: 15)')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256],
                        help='Crop size [H, W] (default: 256 256)')
    parser.add_argument('--seed', type=int, default=44,
                        help='Random seed for reproducibility')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test with single sample before full evaluation')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run quick test, skip full evaluation')
    parser.add_argument('--find_checkpoints', type=str, default=None,
                        help='Find all .pth files in directory')

    args = parser.parse_args()

    if args.find_checkpoints:
        print(f"\nSearching for .pth files in: {args.find_checkpoints}")
        checkpoints = find_checkpoints(args.find_checkpoints)
        if checkpoints:
            print(f"\nFound {len(checkpoints)} checkpoint(s):")
            for i, cp in enumerate(checkpoints, 1):
                size_mb = os.path.getsize(cp) / (1024 * 1024)
                print(f"  {i}. {cp} ({size_mb:.1f} MB)")
        else:
            print("No .pth files found.")
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required (use --find_checkpoints to locate .pth files)")

    if args.data_dir is None:
        parser.error("--data_dir is required")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")

        nearby = find_checkpoints(os.path.dirname(args.checkpoint) or '.')
        if nearby:
            print(f"\nDid you mean one of these?")
            for cp in nearby[:5]:
                print(f"  - {cp}")
        return

    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return

    print("\n" + "="*60)
    print("  LIFT BATCH INFERENCE")
    print("="*60)
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Num Sequences:  {args.num_sequences}")
    print(f"  Num Frames:     {args.num_frames}")
    print(f"  Crop Size:      {args.crop_size}")
    print(f"  Device:         {args.device}")
    print(f"  Output:         {args.output}")
    print("="*60 + "\n")

    pipeline = BatchInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_frames=args.num_frames,
        crop_size=tuple(args.crop_size)
    )

    if args.test or args.test_only:
        success = test_single_inference(pipeline, args.data_dir)
        if not success:
            print("\nAborting due to test failure.")
            return

        if args.test_only:
            return

    report = pipeline.run_batch_evaluation(
        data_dir=args.data_dir,
        num_sequences=args.num_sequences,
        output_dir=args.output,
        seed=args.seed
    )

    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"  Directory: {args.output}")
    print(f"  - report.json       : Detailed metrics for each sequence")
    print(f"  - summary.png       : Statistical visualizations")
    print(f"  - visualizations/   : Individual sample comparisons")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()