"""
Inference script for LIFT model.

Supports two modes:
1. Image Mode: Interpolate a single frame from a directory of images.
2. Video Mode: Interpolate a sequence and save as an MP4 video.
"""

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from model import LIFT
from configs.default import Config


def load_frames_from_directory(frame_dir, num_frames=None):
    """
    Load frames from directory.

    Args:
        frame_dir: Directory containing frames
        num_frames: Number of frames to load (optional). If None, loads all.

    Returns:
        Tensor of frames [1, T, 3, H, W]
    """
    # Find all image files
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(list(Path(frame_dir).glob(ext)))

    frame_paths = sorted(frame_paths)

    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    # Limit number of frames if requested
    if num_frames is not None:
        if len(frame_paths) < num_frames:
            # WARNING instead of Error: we will handle short sequences later
            print(f"Warning: Directory contains only {len(frame_paths)} frames, requested {num_frames}.")
        else:
            frame_paths = frame_paths[:num_frames]

    print(f"Loading {len(frame_paths)} frames...")

    frames = []
    for frame_path in tqdm(frame_paths, desc="Loading frames"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        frames.append(img_tensor)

    # Stack into single tensor [T, 3, H, W]
    frames_tensor = torch.stack(frames, dim=0)

    # Add batch dimension [1, T, 3, H, W]
    return frames_tensor.unsqueeze(0)


def save_frame(frame_tensor, output_path):
    """
    Save frame tensor as image.

    Args:
        frame_tensor: Frame tensor [3, H, W] in range [0, 1]
        output_path: Output path
    """
    # Convert to numpy [H, W, 3] and scale to [0, 255]
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), frame_bgr)


def interpolate_sequence(model, frames, timestep=0.5, device='cuda'):
    """
    Interpolate frame from sequence.

    Args:
        model: LIFT model
        frames: Input frames [1, T, 3, H, W]
        timestep: Interpolation timestep
        device: Device to use

    Returns:
        Interpolated frame [1, 3, H, W]
    """
    model.eval()
    frames = frames.to(device)

    with torch.no_grad():
        # LIFT inference handles the windowing internally or takes the whole sequence
        output = model.inference(frames, timestep=timestep)

    return output


def pad_sequence(frames, target_length):
    """
    Pad sequence to target length by repeating the last frame.

    Args:
        frames: [B, T, C, H, W]
        target_length: int

    Returns:
        Padded frames [B, target_length, C, H, W]
    """
    B, T, C, H, W = frames.shape
    if T >= target_length:
        return frames

    diff = target_length - T
    print(f"Padding input sequence from {T} to {target_length} frames (repeating last frame)...")

    # Create padding by repeating the last frame
    last_frame = frames[:, -1:].repeat(1, diff, 1, 1, 1)
    padded = torch.cat([frames, last_frame], dim=1)

    return padded


def run_image_mode(args, model, device, config):
    """Run inference to generate a single interpolated image."""
    print(f"\n--- Image Mode ---")
    print(f"Loading frames from {args.input}...")

    # If user specified num_frames, try to load that many.
    # If directory has fewer, load_frames_from_directory will load what's available.
    req_frames = args.num_frames if args.num_frames else config.num_frames

    frames = load_frames_from_directory(args.input, num_frames=req_frames)
    print(f"Loaded input tensor: {frames.shape}")

    # Ensure we meet the model's requirement (usually 15 frames)
    # If we have fewer frames than config.num_frames, the model will crash (index out of bounds)
    # because it expects reference frames at specific indices (e.g. 6 and 7).
    if frames.shape[1] < config.num_frames:
        frames = pad_sequence(frames, config.num_frames)
        print(f"Padded input tensor: {frames.shape}")

    # Interpolate
    print(f"Interpolating at t={args.timestep}...")
    interpolated = interpolate_sequence(model, frames, timestep=args.timestep, device=device)

    # Save output
    if os.path.isdir(args.output):
        output_path = Path(args.output) / f'interpolated_t{args.timestep:.2f}.png'
    else:
        output_path = Path(args.output)

    save_frame(interpolated[0], output_path)
    print(f"Saved interpolated frame to {output_path}")


def run_video_mode(args, model, device, config):
    """Run inference to generate a video with interpolated frames."""
    print(f"\n--- Video Mode ---")
    print(f"Loading frames from {args.input}...")

    # Load all frames
    frames = load_frames_from_directory(args.input, num_frames=args.num_frames)
    B, T, C, H, W = frames.shape
    print(f"Loaded sequence: {T} frames, {H}x{W}")

    # Setup video writer
    if os.path.isdir(args.output):
        output_path = Path(args.output) / 'output_video.mp4'
    else:
        output_path = Path(args.output)

    fps = 30 # Default FPS
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    print(f"Generating video to {output_path}...")

    window_size = config.num_frames # Should be 15 typically
    mid_idx = window_size // 2

    # Pre-convert to numpy for saving (original frames)
    frames_cpu = frames[0].permute(0, 2, 3, 1).cpu().numpy() # [T, H, W, 3]
    frames_cpu = (frames_cpu * 255).clip(0, 255).astype(np.uint8)

    valid_start = 0
    valid_end = T - window_size + 1

    if valid_end <= 0:
        print(f"Warning: Sequence length {T} is shorter than model window {window_size}.")
        print("Cannot perform standard sliding window inference.")
        # Optional: Handle short video with padding if needed, but usually video mode implies long sequence
        return

    # Write leading frames that we can't interpolate
    start_output_idx = mid_idx - 1
    for i in range(start_output_idx + 1):
        frame_bgr = cv2.cvtColor(frames_cpu[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    for i in tqdm(range(valid_start, valid_end)):
        # Extract window
        window = frames[:, i : i + window_size].to(device)

        # Interpolate
        interpolated = interpolate_sequence(model, window, timestep=args.timestep, device=device)

        # Convert interpolated to uint8
        interp_tensor = interpolated[0].detach().cpu()
        interp_np = (interp_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        interp_bgr = cv2.cvtColor(interp_np, cv2.COLOR_RGB2BGR)
        writer.write(interp_bgr)

        # Append Next Original Frame (Right reference)
        next_orig_idx = i + mid_idx
        if next_orig_idx < T:
            frame_bgr = cv2.cvtColor(frames_cpu[next_orig_idx], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    writer.release()
    print("Video generation complete.")


def main():
    parser = argparse.ArgumentParser(description='LIFT Inference')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video'],
                        help='Inference mode: image (single frame) or video (mp4 generation)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing frames')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path (directory for image, file/dir for video)')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Interpolation timestep (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to load from directory (optional override)')
    args = parser.parse_args()

    # Ensure output dir exists if it's a directory path
    if not args.output.endswith('.mp4') and not args.output.endswith('.png'):
        os.makedirs(args.output, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load config and model
    print("Loading model...")
    config = Config()

    # Safe weight loading (filters mismatched shapes like PE)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Load config from checkpoint if available
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LIFT(config).to(device)

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Safe weight loading (filters mismatched shapes like PE)
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']

    new_state_dict = {}
    for k, v in checkpoint_state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state[k].shape}")

    # Loading filtered weights (strict=False allows missing PE buffers)
    model.load_state_dict(new_state_dict, strict=False)

    if args.mode == 'image':
        run_image_mode(args, model, device, config)
    elif args.mode == 'video':
        run_video_mode(args, model, device, config)


if __name__ == '__main__':
    main()