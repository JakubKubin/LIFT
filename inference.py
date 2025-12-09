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
import matplotlib.pyplot as plt

from model import LIFT
from configs.default import Config

from utils.metrics import Evaluator

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
    """
    Pad sequence symmetrically to preserve the center frame.
    Repeats the first frame at the beginning and the last frame at the end.
    """
    B, T, C, H, W = frames.shape
    if T >= target_length:
        return frames

    diff = target_length - T

    # Dzielimy różnicę na pół: dla 7->15 diff=8, więc front=4, back=4
    pad_front = diff // 2
    pad_back = diff - pad_front

    print(f"Padding sequence: {T} -> {target_length} (Front: {pad_front} frames, Back: {pad_back} frames)...")

    # Kopiowanie pierwszej klatki na przód
    front_frames = frames[:, :1].repeat(1, pad_front, 1, 1, 1)

    # Kopiowanie ostatniej klatki na tył
    back_frames = frames[:, -1:].repeat(1, pad_back, 1, 1, 1)

    # Sklejanie: [Kopie_Początku, Oryginał, Kopie_Końca]
    padded = torch.cat([front_frames, frames, back_frames], dim=1)
    return padded

def visualize_results(pred_tensor, gt_tensor, metrics_dict, output_path):
    """
    Displays visual comparison and saves it as an image.
    Accepts tensors [1, 3, H, W]
    """
    # Convert to numpy for matplotlib
    def tensor_to_img(t):
        img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img * 255).clip(0, 255).astype(np.uint8)

    pred_img = tensor_to_img(pred_tensor)
    gt_img = tensor_to_img(gt_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_img)
    axes[1].set_title('Interpolated', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    metrics_text = f"PSNR: {metrics_dict['PSNR']:.2f} dB  |  SSIM: {metrics_dict['SSIM']:.4f}  |  LPIPS: {metrics_dict['LPIPS']:.4f}"

    fig.text(0.5, 0.05, metrics_text, ha='center', fontsize=12,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    plt.tight_layout(rect=(0, 0.1, 1, 1))

    # Save visualization with _comparison suffix
    # Handle png/mp4 suffix correctly if output_path is a full file path
    if str(output_path).lower().endswith(('.png', '.jpg', '.jpeg')):
        viz_path = str(output_path).rsplit('.', 1)[0] + '_comparison.png'
    else:
        viz_path = str(output_path) + '_comparison.png'
    plt.savefig(viz_path)
    plt.close(fig)
    print(f"Comparison saved to {viz_path}")

def run_image_mode(args, model, device, config):
    """
    Run inference in Image Mode.
    If full sequence is provided, calculates metrics against Ground Truth.
    """
    print(f"\n--- Image Mode ---")

    if os.path.isdir(args.output):
        output_path = Path(args.output) / f'interpolated_t{args.timestep:.2f}.png'
    else:
        output_path = Path(args.output)

    print(f"Loading frames from {args.input}...")

    # If user specified num_frames, try to load that many.
    # If directory has fewer, load_frames_from_directory will load what's available.
    req_frames = args.num_frames if args.num_frames else config.num_frames
    frames = load_frames_from_directory(args.input, num_frames=req_frames)
    print(f"Loaded input tensor: {frames.shape}")

    gt_frame = None

    if frames.shape[1] == config.num_frames:
        mid_idx = config.num_frames // 2
        gt_frame = frames[:, mid_idx].to(device)
        print(f"Ground Truth detected at frame index {mid_idx}.")

    # Check if we loaded exactly 3 frames (triplet), common for VFI testing
    elif frames.shape[1] == 3 and config.num_frames > 3:
        gt_frame = frames[:, 1].to(device) # Middle frame is GT
        # For inference we need more context, so we might need padding or
        # specifically crafting input if user provided only 3 frames but model needs 15.
        # Here we assume user provides full sequence or we pad.
        print("Loaded 3 frames, assuming middle is GT.")

    # Ensure we meet the model's requirement (usually 15 frames)
    if frames.shape[1] < config.num_frames:
        input_frames = pad_sequence(frames, config.num_frames)
        print(f"Padded input tensor: {input_frames.shape}")
    else:
        input_frames = frames

    print(f"Interpolating at t={args.timestep}...")
    interpolated = interpolate_sequence(model, input_frames, timestep=args.timestep, device=device)

    if gt_frame is not None:
        print("\n--- Evaluation Metrics ---")
        evaluator = Evaluator(device)
        metrics = evaluator.compute_metrics(interpolated, gt_frame)

        print(f"PSNR:  {metrics['PSNR']:.4f} dB")
        print(f"SSIM:  {metrics['SSIM']:.4f}")
        print(f"LPIPS: {metrics['LPIPS']:.4f}")

        visualize_results(interpolated, gt_frame, metrics, output_path)

        # Save metrics to text file
        metrics_file = str(output_path).rsplit('.', 1)[0] + '_metrics.txt'
        with open(metrics_file, "w") as f:
            f.write(f"Source: {args.input}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
        print(f"Metrics saved to {metrics_file}")

    else:
        print("No Ground Truth available for metrics evaluation.")

    # Save output image
    save_frame(interpolated[0], output_path)
    print(f"Saved interpolated frame to {output_path}")


def run_video_mode(args, model, device, config):
    """Run inference to generate a video with interpolated frames."""
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

    # Load Checkpoint with safety settings (weights_only=False for scalar configs)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
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
            else:
                print(f"Skipping {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state[k].shape}")

    model.load_state_dict(new_state, strict=False)

    if args.mode == 'image':
        run_image_mode(args, model, device, config)
    elif args.mode == 'video':
        run_video_mode(args, model, device, config)


if __name__ == '__main__':
    main()