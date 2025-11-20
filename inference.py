"""
Inference script for LIFT model.

Load a trained model and interpolate frames from test sequences.
"""

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import LIFT
from configs.default import Config


def load_frames_from_directory(frame_dir, num_frames=64):
    """
    Load frames from directory.

    Args:
        frame_dir: Directory containing frames
        num_frames: Number of frames to load

    Returns:
        Tensor of frames [1, num_frames, 3, H, W]
    """
    frame_paths = sorted(list(Path(frame_dir).glob('*.png')) +
                        list(Path(frame_dir).glob('*.jpg')))

    if len(frame_paths) < num_frames:
        raise ValueError(f"Directory contains only {len(frame_paths)} frames, need {num_frames}")

    frames = []
    for frame_path in frame_paths[:num_frames]:
        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        frames.append(img_tensor)

    # Stack into single tensor [num_frames, 3, H, W]
    frames_tensor = torch.stack(frames, dim=0)

    # Add batch dimension [1, num_frames, 3, H, W]
    return frames_tensor.unsqueeze(0)


def save_frame(frame_tensor, output_path):
    """
    Save frame tensor as image.

    Args:
        frame_tensor: Frame tensor [3, H, W] in range [0, 1]
        output_path: Output path
    """
    # Convert to numpy [H, W, 3] and scale to [0, 255]
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), frame_bgr)


def interpolate_sequence(model, frames, timestep=0.5, device: str | torch.device = 'cuda'):
    """
    Interpolate frame from sequence.

    Args:
        model: LIFT model
        frames: Input frames [1, 64, 3, H, W]
        timestep: Interpolation timestep
        device: Device to use

    Returns:
        Interpolated frame [1, 3, H, W]
    """
    model.eval()
    frames = frames.to(device)

    with torch.no_grad():
        output = model.inference(frames, timestep=timestep)

    return output


def main():
    parser = argparse.ArgumentParser(description='LIFT Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing frames')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for interpolated frame')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='Interpolation timestep (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load config and model
    print("Loading model...")
    config = Config()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Load config from checkpoint if available
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LIFT(config).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    if 'val_psnr' in checkpoint:
        print(f"Validation PSNR: {checkpoint['val_psnr']:.2f} dB")

    # Load frames
    print(f"\nLoading frames from {args.input}...")
    frames = load_frames_from_directory(args.input, num_frames=config.num_frames)
    print(f"Loaded {frames.shape[1]} frames of size {frames.shape[3]}x{frames.shape[4]}")

    # Interpolate
    print(f"\nInterpolating at t={args.timestep}...")
    interpolated = interpolate_sequence(model, frames, timestep=args.timestep, device=device)

    # Save output
    output_path = Path(args.output) / f'interpolated_t{args.timestep:.2f}.png'
    save_frame(interpolated[0], output_path)
    print(f"Saved interpolated frame to {output_path}")

    # Also save reference frames for comparison
    ref_31 = frames[0, 31]
    ref_32 = frames[0, 32]
    save_frame(ref_31, Path(args.output) / 'reference_31.png')
    save_frame(ref_32, Path(args.output) / 'reference_32.png')
    print(f"Saved reference frames to {args.output}")


def batch_inference():
    """Process multiple sequences in batch."""
    parser = argparse.ArgumentParser(description='LIFT Batch Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_list', type=str, required=True,
                       help='Text file with list of input directories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for all results')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='Interpolation timestep')
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    config = Config()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LIFT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Read input list
    with open(args.input_list, 'r') as f:
        input_dirs = [line.strip() for line in f.readlines()]

    print(f"\nProcessing {len(input_dirs)} sequences...")

    # Process each sequence
    for i, input_dir in enumerate(tqdm(input_dirs)):
        try:
            # Load frames
            frames = load_frames_from_directory(input_dir, num_frames=config.num_frames)

            # Interpolate
            interpolated = interpolate_sequence(model, frames, timestep=args.timestep, device=device)

            # Save
            seq_name = Path(input_dir).name
            output_path = Path(args.output_dir) / f'{seq_name}_interpolated.png'
            save_frame(interpolated[0], output_path)

        except Exception as e:
            print(f"\nError processing {input_dir}: {e}")
            continue

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
