"""
Inference script for LIFT model with Metrics Evaluation.

Supports two modes:
1. Image Mode: Interpolate a single frame from a directory of images and compare with GT.
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

# Import Evaluator for metrics
try:
    from utils.metrics import Evaluator
except ImportError:
    print("Warning: utils.metrics not found. Metrics calculation disabled.")
    Evaluator = None


def load_frames_from_directory(frame_dir, num_frames=None):
    """Load frames from directory [1, T, 3, H, W]."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(list(Path(frame_dir).glob(ext)))

    frame_paths = sorted(frame_paths)

    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    if num_frames is not None:
        if len(frame_paths) < num_frames:
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
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        frames.append(img_tensor)

    frames_tensor = torch.stack(frames, dim=0)
    return frames_tensor.unsqueeze(0)


def save_frame(frame_tensor, output_path):
    """Save frame tensor as image."""
    frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), frame_bgr)


def interpolate_sequence(model, frames, timestep=0.5, device='cuda'):
    """Interpolate frame from sequence."""
    model.eval()
    frames = frames.to(device)
    with torch.no_grad():
        output = model.inference(frames, timestep=timestep)
    return output


def pad_sequence(frames, target_length):
    """Pad sequence by repeating last frame."""
    B, T, C, H, W = frames.shape
    if T >= target_length:
        return frames

    diff = target_length - T
    print(f"Padding input sequence from {T} to {target_length} frames (repeating last frame)...")
    last_frame = frames[:, -1:].repeat(1, diff, 1, 1, 1)
    padded = torch.cat([frames, last_frame], dim=1)
    return padded


# def run_image_mode(args, model, device, config):
#     """
#     Run inference in Image Mode.
#     If full sequence is provided, calculates metrics against Ground Truth.
#     """
#     print(f"\n--- Image Mode ---")
#     print(f"Loading frames from {args.input}...")

#     req_frames = args.num_frames if args.num_frames else config.num_frames
#     frames = load_frames_from_directory(args.input, num_frames=req_frames)

#     # Check if we have Ground Truth
#     # LIFT expects 15 frames. The middle one (index 7) is the target.
#     # If we loaded 15 frames, we have the real GT at index 7.
#     has_gt = False
#     gt_frame = None

#     print(frames.shape[1], config.num_frames)

#     if frames.shape[1] == config.num_frames:
#         mid_idx = config.num_frames // 2
#         gt_frame = frames[:, mid_idx].to(device)
#         has_gt = True
#         print(f"Ground Truth detected at frame index {mid_idx}.")

#     # Padding if needed for model inference
#     if frames.shape[1] < config.num_frames:
#         input_frames = pad_sequence(frames, config.num_frames)
#         print(f"Padded input tensor: {input_frames.shape}")
#     else:
#         input_frames = frames

#     print(f"Interpolating at t={args.timestep}...")
#     interpolated = interpolate_sequence(model, input_frames, timestep=args.timestep, device=device)

#     # Calculate Metrics
#     if has_gt and Evaluator is not None:
#         print("\n--- Evaluation Metrics ---")
#         evaluator = Evaluator(device)
#         metrics = evaluator.compute_metrics(interpolated, gt_frame) # type: ignore

#         print(f"PSNR:  {metrics['PSNR']:.4f} dB")
#         print(f"SSIM:  {metrics['SSIM']:.4f}")
#         print(f"LPIPS: {metrics['LPIPS']:.4f}")

#         # Save metrics to file
#         if os.path.isdir(args.output):
#             metrics_path = Path(args.output) / "metrics.txt"
#             with open(metrics_path, "w") as f:
#                 f.write(f"Evaluation for: {args.input}\n")
#                 for k, v in metrics.items():
#                     f.write(f"{k}: {v:.6f}\n")
#             print(f"Metrics saved to {metrics_path}")
#     else:
#         if not has_gt:
#             print("No Ground Truth available for metrics calculation.")
#         if Evaluator is None:
#             print("Evaluator not available. Metrics calculation skipped.")

#     # Save output image
#     if os.path.isdir(args.output):
#         output_path = Path(args.output) / f'interpolated_t{args.timestep:.2f}.png'
#     else:
#         output_path = Path(args.output)

#     save_frame(interpolated[0], output_path)
#     print(f"Saved interpolated frame to {output_path}")

def run_image_mode(args, model, device, config):
    """
    Run inference in Image Mode.
    Handles specific padding logic for 7 and 3 frame datasets to match 15-frame requirement.
    """
    print(f"\n--- Image Mode ---")
    print(f"Loading frames from {args.input}...")

    # Load all available frames first to determine strategy
    frames = load_frames_from_directory(args.input, num_frames=args.num_frames)
    B, T, C, H, W = frames.shape
    
    print(f"Loaded {T} frames. Target input size is {config.num_frames} frames.")

    has_gt = False
    gt_frame = None
    input_frames = None

    # Logic for 7 Frames (GT is index 3)
    if T == 7:
        print("Detected 7-frame sequence. Applying 7x First / 7x Last padding strategy.")
        
        # Ground Truth is the middle frame of the original 7 (index 3)
        gt_frame = frames[:, 3].to(device)
        has_gt = True
        
        # Construct 15 frames: [First*7, GT, Last*7]
        first_frame = frames[:, 0:1] # Shape: [B, 1, C, H, W]
        last_frame = frames[:, 6:7]
        middle_frame = frames[:, 3:4] # This is the GT, kept in tensor for shape
        
        # Repeat first and last frames 7 times
        prefix = first_frame.repeat(1, 7, 1, 1, 1)
        suffix = last_frame.repeat(1, 7, 1, 1, 1)
        
        # Concatenate: 7 + 1 + 7 = 15 frames
        input_frames = torch.cat([prefix, middle_frame, suffix], dim=1)
        print(f"Constructed input tensor: {input_frames.shape} (7x First, 1x GT, 7x Last)")

    # Logic for 3 Frames (GT is index 1)
    elif T == 3:
        print("Detected 3-frame sequence. Applying 7x First / 7x Last padding strategy.")
        
        # Ground Truth is the middle frame (index 1)
        gt_frame = frames[:, 1].to(device)
        has_gt = True
        
        # Construct 15 frames: [First*7, GT, Last*7]
        first_frame = frames[:, 0:1]
        last_frame = frames[:, 2:3]
        middle_frame = frames[:, 1:2]
        
        prefix = first_frame.repeat(1, 7, 1, 1, 1)
        suffix = last_frame.repeat(1, 7, 1, 1, 1)
        
        input_frames = torch.cat([prefix, middle_frame, suffix], dim=1)
        print(f"Constructed input tensor: {input_frames.shape} (7x First, 1x GT, 7x Last)")

    # Logic for 15 Frames (Standard LIFT requirement)
    elif T == config.num_frames:
        print("Detected full 15-frame sequence.")
        mid_idx = config.num_frames // 2
        gt_frame = frames[:, mid_idx].to(device)
        has_gt = True
        input_frames = frames

    # Fallback for other frame counts (Generic Padding)
    else:
        print(f"Frame count {T} does not match specific test cases (3, 7, or 15). using generic padding.")
        if T < config.num_frames:
            input_frames = pad_sequence(frames, config.num_frames)
        else:
            input_frames = frames[:, :config.num_frames]

        # Try to define a GT if possible (simple middle), though accuracy is not guaranteed for metrics
        if T % 2 == 1:
            gt_frame = frames[:, T // 2].to(device)
            has_gt = True

    # Run Inference
    print(f"Interpolating at t={args.timestep}...")
    interpolated = interpolate_sequence(model, input_frames, timestep=args.timestep, device=device)

    # Calculate Metrics
    if has_gt and Evaluator is not None:
        print("\n--- Evaluation Metrics ---")
        evaluator = Evaluator(device)
        metrics = evaluator.compute_metrics(interpolated, gt_frame)

        print(f"PSNR:  {metrics['PSNR']:.4f} dB")
        print(f"SSIM:  {metrics['SSIM']:.4f}")
        print(f"LPIPS: {metrics['LPIPS']:.4f}")

        if os.path.isdir(args.output):
            metrics_path = Path(args.output) / "metrics.txt"
            with open(metrics_path, "w") as f:
                f.write(f"Evaluation for: {args.input}\n")
                f.write(f"Original Frames: {T}\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
            print(f"Metrics saved to {metrics_path}")
    else:
        if not has_gt:
            print("No Ground Truth available for metrics calculation.")
        if Evaluator is None:
            print("Evaluator not available. Metrics calculation skipped.")

    # Save output image
    if os.path.isdir(args.output):
        output_path = Path(args.output) / f'interpolated_t{args.timestep:.2f}.png'
    else:
        output_path = Path(args.output)

    save_frame(interpolated[0], output_path)
    print(f"Saved interpolated frame to {output_path}")

def run_video_mode(args, model, device, config):
    """Run inference to generate a video."""
    print(f"\n--- Video Mode ---")
    print(f"Loading frames from {args.input}...")

    frames = load_frames_from_directory(args.input, num_frames=args.num_frames)
    B, T, C, H, W = frames.shape
    print(f"Loaded sequence: {T} frames, {H}x{W}")

    if os.path.isdir(args.output):
        output_path = Path(args.output) / 'output_video.mp4'
    else:
        output_path = Path(args.output)

    fps = 30
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    print(f"Generating video to {output_path}...")

    window_size = config.num_frames
    mid_idx = window_size // 2
    frames_cpu = frames[0].permute(0, 2, 3, 1).cpu().numpy()
    frames_cpu = (frames_cpu * 255).clip(0, 255).astype(np.uint8)

    valid_start = 0
    valid_end = T - window_size + 1

    if valid_end <= 0:
        print(f"Warning: Sequence too short ({T} < {window_size}). Cannot interpolate.")
        return

    start_output_idx = mid_idx - 1
    for i in range(start_output_idx + 1):
        frame_bgr = cv2.cvtColor(frames_cpu[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    for i in tqdm(range(valid_start, valid_end)):
        window = frames[:, i : i + window_size].to(device)
        interpolated = interpolate_sequence(model, window, timestep=args.timestep, device=device)

        interp_tensor = interpolated[0].detach().cpu()
        interp_np = (interp_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        interp_bgr = cv2.cvtColor(interp_np, cv2.COLOR_RGB2BGR)
        writer.write(interp_bgr)

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
                        help='Output path')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Interpolation timestep (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Override number of frames to load')
    args = parser.parse_args()

    if not args.output.endswith('.mp4') and not args.output.endswith('.png'):
        os.makedirs(args.output, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading model...")
    config = Config()

    # Load Checkpoint with safety settings
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LIFT(config).to(device)

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Filter incompatible weights
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    new_state = {}
    for k, v in checkpoint_state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                new_state[k] = v

    model.load_state_dict(new_state, strict=False)

    if args.mode == 'image':
        run_image_mode(args, model, device, config)
    elif args.mode == 'video':
        run_video_mode(args, model, device, config)

if __name__ == '__main__':
    main()