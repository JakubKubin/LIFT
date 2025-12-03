import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

def calculate_metrics(gt_img, pred_img):
    psnr_value = psnr(gt_img, pred_img, data_range=255)

    ssim_value = ssim(gt_img, pred_img, channel_axis=2, data_range=255)

    loss_fn = lpips.LPIPS(net='alex')

    gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0
    pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).float() / 255.0

    gt_tensor = gt_tensor.unsqueeze(0) * 2 - 1
    pred_tensor = pred_tensor.unsqueeze(0) * 2 - 1

    with torch.no_grad():
        lpips_value = loss_fn(gt_tensor, pred_tensor).item()

    return psnr_value, ssim_value, lpips_value

def visualize_comparison(gt_path, pred_path):
    gt_img = load_image(gt_path)
    pred_img = load_image(pred_path)

    psnr_val, ssim_val, lpips_val = calculate_metrics(gt_img, pred_img)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_img)
    axes[1].set_title('Interpolated', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    metrics_text = f'PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  LPIPS: {lpips_val:.4f}'
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=14,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # type: ignore
    plt.show()

    print(f"\n{'='*60}")
    print(f"METRICS RESULTS:")
    print(f"{'='*60}")
    print(f"PSNR:  {psnr_val:.4f} dB")
    print(f"SSIM:  {ssim_val:.4f}")
    print(f"LPIPS: {lpips_val:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare GT and Interpolated frames')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth image')
    parser.add_argument('--pred', type=str, required=True, help='Path to interpolated image')

    args = parser.parse_args()

    visualize_comparison(args.gt, args.pred)