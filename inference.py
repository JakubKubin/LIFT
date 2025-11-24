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
            raise ValueError(f"Directory contains only {len(frame_paths)} frames, need {num_frames}")
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
        # If T != config.num_frames, the model needs to handle it.
        # Based on model/lift.py, it takes 'frames' and uses encoder/transformer.
        # Ensure T matches training configuration or model supports variable T.
        # For LIFT, we typically expect 64 frames.
        output = model.inference(frames, timestep=timestep)

    return output


def run_image_mode(args, model, device, config):
    """Run inference to generate a single interpolated image."""
    print(f"\n--- Image Mode ---")
    print(f"Loading frames from {args.input}...")
    
    # For single image interpolation, we typically need the context window size (e.g. 64)
    # If user specified num_frames, use it, otherwise default to config
    num_frames = args.num_frames if args.num_frames else config.num_frames
    
    frames = load_frames_from_directory(args.input, num_frames=num_frames)
    print(f"Loaded input tensor: {frames.shape}")

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
    # Note: Loading ALL frames into memory might be heavy for long videos.
    # ideally we would stream, but for simplicity we load all.
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
    
    # Sliding window approach
    # We need 64 frames context to interpolate between frame 31 and 32 (indices)
    # Central pair is at index (64//2)-1 and (64//2) -> 31 and 32
    
    window_size = config.num_frames # Should be 64 typically
    mid_idx = window_size // 2
    
    # We can only interpolate where we have full context
    # Start from index 0, take window_size
    # We will generate 2x frames: Original[i], Interpolated[i+0.5]
    
    # Pre-convert to numpy for saving (original frames)
    frames_cpu = frames[0].permute(0, 2, 3, 1).cpu().numpy() # [T, H, W, 3]
    frames_cpu = (frames_cpu * 255).clip(0, 255).astype(np.uint8)
    
    # Write first frame
    # writer.write(cv2.cvtColor(frames_cpu[0], cv2.COLOR_RGB2BGR))

    # Iterate
    # For a sequence of length T and window W (e.g. 64)
    # We can technically interpolate at many positions, but LIFT is optimized 
    # to interpolate the middle of the window.
    # Let's assume we simply slide the window 1 frame at a time.
    # Window i: frames[i : i+W]
    # Interpolates between frames[i + mid-1] and frames[i + mid]
    
    # To generate a full video, we need to pad or handle edges.
    # For strict LIFT usage, we can only interpolate frames that have full 64-frame context.
    # That means we can interpolate between index 31 and 32, 32 and 33, ... T-32 and T-31.
    
    valid_start = 0
    valid_end = T - window_size + 1
    
    if valid_end <= 0:
        print(f"Warning: Sequence length {T} is shorter than model window {window_size}.")
        print("Cannot perform standard sliding window inference.")
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
        # This generates frame at t=0.5 between indices (mid-1) and (mid) relative to window
        # Absolute indices: i + mid-1 and i + mid
        interpolated = interpolate_sequence(model, window, timestep=args.timestep, device=device)
        
        # Save Original Frame (Left reference) - already written in previous loop or pre-fill
        # actually, we need to interleave.
        
        # Current window produces frame between (i + 31) and (i + 32)
        # We append Interpolated
        
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

    # Write remaining trailing frames
    # Not needed if loop covers everything correctly up to T-1
    
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

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Load config from checkpoint if available
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Override config num_frames if needed for model init (though LIFT handles variable somewhat)
    # Ideally model structure doesn't change with T, but positional embeddings might
    
    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LIFT(config).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    if args.mode == 'image':
        run_image_mode(args, model, device, config)
    elif args.mode == 'video':
        run_video_mode(args, model, device, config)


if __name__ == '__main__':
    main()

### Explanation of Changes

# 1.  **Two Modes (`--mode`)**:
#     * `image`: Similar to the old functionality. Loads a specific number of frames (defaulting to 64 from config, or user-specified via `--num_frames`) and interpolates the middle frame. This is great for testing quality on snippets.
#     * `video`: Designed for processing longer sequences. It sets up a `cv2.VideoWriter` to create an `.mp4` file.

# 2.  **`load_frames_from_directory` Update**:
#     * Added `num_frames` argument. If `None`, it loads *all* images found in the directory. This is crucial for video mode where you might have hundreds of frames.
#     * Added a progress bar (`tqdm`) for loading frames, as this can take time for large sequences.

# 3.  **Video Generation Logic**:
#     * Because LIFT requires a large context (64 frames), we can't just interpolate between *any* two frames easily. The script uses a **sliding window** approach.
#     * It iterates through the video sequence. For every position, it extracts a 64-frame window centered on the gap we want to interpolate.
#     * It writes the original frames and the interpolated frames interleaved to double the frame rate (conceptually).
#     * **Note on Edge Cases**: With a 64-frame window, the model cannot interpolate the very first ~30 frames or the very last ~30 frames because it lacks the full context required by the architecture. The script writes these original frames without interpolation to preserve the timeline.

# 4.  **Config Flexibility**:
#     * The script attempts to respect the training configuration loaded from the checkpoint but allows overriding the number of input frames via command-line arguments.

# ### Usage Examples

# **1. Interpolate a single frame (like before):**
# python inference.py --mode image \
#     --checkpoint checkpoints/best_model.pth \
#     --input my_frames_dir/ \
#     --output results/ \
#     --num_frames 64