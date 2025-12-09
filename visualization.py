"""
Visualization Grid Utility for LIFT Human Evaluation.

Creates comprehensive comparison grids for visual assessment.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Dict, Optional
import json


def create_comparison_grid(results_dir: str, output_path: str,
                           num_samples: int = 16, sort_by: str = 'PSNR'):
    """
    Create a grid of best/worst samples for human evaluation.

    Args:
        results_dir: Directory containing batch inference results
        output_path: Path to save the grid image
        num_samples: Number of samples to show (half best, half worst)
        sort_by: Metric to sort by ('PSNR', 'SSIM', or 'LPIPS')
    """
    results_path = Path(results_dir)
    report_file = results_path / 'report.json'

    if not report_file.exists():
        raise FileNotFoundError(f"Report not found: {report_file}")

    with open(report_file, 'r') as f:
        report = json.load(f)

    results = report['results']

    reverse = sort_by != 'LPIPS'
    sorted_results = sorted(results, key=lambda x: x['metrics'][sort_by], reverse=reverse)

    half = num_samples // 2
    best_samples = sorted_results[:half]
    worst_samples = sorted_results[-half:]

    viz_dir = results_path / 'visualizations'

    fig = plt.figure(figsize=(20, 24))

    fig.suptitle(f'LIFT Evaluation Grid (Sorted by {sort_by})\nTop Row: Best | Bottom Row: Worst',
                 fontsize=16, fontweight='bold', y=0.98)

    cols = 4
    rows_per_section = (half + cols - 1) // cols

    for section_idx, (samples, label) in enumerate([(best_samples, 'BEST'), (worst_samples, 'WORST')]):
        for i, sample in enumerate(samples):
            row = section_idx * rows_per_section + i // cols
            col = i % cols

            ax = fig.add_subplot(rows_per_section * 2, cols, row * cols + col + 1)

            viz_file = viz_dir / f"sample_{sample['idx']:04d}.png"
            if viz_file.exists():
                img = plt.imread(str(viz_file))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')

            metrics = sample['metrics']
            title = f"{label} #{i+1}\nPSNR:{metrics['PSNR']:.1f} SSIM:{metrics['SSIM']:.3f}"

            color = '#27ae60' if label == 'BEST' else '#e74c3c'
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
            ax.axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison grid saved to {output_path}")


def create_metrics_dashboard(results_dir: str, output_path: str):
    """
    Create a comprehensive metrics dashboard.
    """
    results_path = Path(results_dir)
    report_file = results_path / 'report.json'

    with open(report_file, 'r') as f:
        report = json.load(f)

    results = report['results']
    summary = report['summary']

    psnr_vals = [r['metrics']['PSNR'] for r in results]
    ssim_vals = [r['metrics']['SSIM'] for r in results]
    lpips_vals = [r['metrics']['LPIPS'] for r in results]

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(psnr_vals, bins=40, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(summary['PSNR']['mean'], color='red', linestyle='--', linewidth=2)
    ax1.set_title('PSNR Distribution', fontweight='bold')
    ax1.set_xlabel('PSNR (dB)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ssim_vals, bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax2.axvline(summary['SSIM']['mean'], color='red', linestyle='--', linewidth=2)
    ax2.set_title('SSIM Distribution', fontweight='bold')
    ax2.set_xlabel('SSIM')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(lpips_vals, bins=40, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax3.axvline(summary['LPIPS']['mean'], color='blue', linestyle='--', linewidth=2)
    ax3.set_title('LPIPS Distribution', fontweight='bold')
    ax3.set_xlabel('LPIPS (lower is better)')

    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(psnr_vals, ssim_vals, c=lpips_vals, cmap='RdYlGn_r', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax4, label='LPIPS')
    ax4.set_xlabel('PSNR (dB)')
    ax4.set_ylabel('SSIM')
    ax4.set_title('PSNR vs SSIM (color = LPIPS)', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(psnr_vals, lpips_vals, c=ssim_vals, cmap='RdYlGn', alpha=0.6, s=30)
    ax5.set_xlabel('PSNR (dB)')
    ax5.set_ylabel('LPIPS')
    ax5.set_title('PSNR vs LPIPS', fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(ssim_vals, lpips_vals, c=psnr_vals, cmap='RdYlGn', alpha=0.6, s=30)
    ax6.set_xlabel('SSIM')
    ax6.set_ylabel('LPIPS')
    ax6.set_title('SSIM vs LPIPS', fontweight='bold')

    ax7 = fig.add_subplot(gs[2, 0])
    ax7.boxplot([psnr_vals, ssim_vals])
    ax7.set_xticklabels(['PSNR (scaled)', 'SSIM'])
    ax7.set_title('Box Plots', fontweight='bold')

    ax8 = fig.add_subplot(gs[2, 1])
    indices = range(len(psnr_vals))
    ax8.plot(indices, psnr_vals, alpha=0.7, linewidth=0.5)
    ax8.axhline(summary['PSNR']['mean'], color='red', linestyle='--')
    ax8.set_xlabel('Sample Index')
    ax8.set_ylabel('PSNR (dB)')
    ax8.set_title('PSNR Over Samples', fontweight='bold')

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    stats_text = "SUMMARY STATISTICS\n" + "="*35 + "\n\n"

    for metric in ['PSNR', 'SSIM', 'LPIPS']:
        s = summary[metric]
        stats_text += f"{metric}:\n"
        stats_text += f"  Mean:   {s['mean']:.4f}\n"
        stats_text += f"  Std:    {s['std']:.4f}\n"
        stats_text += f"  Median: {s['median']:.4f}\n"
        stats_text += f"  Range:  [{s['min']:.4f}, {s['max']:.4f}]\n\n"

    stats_text += f"Total Samples: {len(results)}\n"
    stats_text += f"Skipped: {len(report.get('skipped', []))}"

    ax9.text(0.1, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=11, family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Metrics dashboard saved to {output_path}")


def create_failure_analysis(results_dir: str, output_path: str, threshold_psnr: float = 25.0):
    """
    Analyze and visualize failure cases (low PSNR samples).
    """
    results_path = Path(results_dir)
    report_file = results_path / 'report.json'

    with open(report_file, 'r') as f:
        report = json.load(f)

    results = report['results']

    failures = [r for r in results if r['metrics']['PSNR'] < threshold_psnr]
    failures = sorted(failures, key=lambda x: x['metrics']['PSNR'])

    if len(failures) == 0:
        print(f"No failures found (PSNR < {threshold_psnr})")
        return

    print(f"Found {len(failures)} failure cases (PSNR < {threshold_psnr} dB)")

    viz_dir = results_path / 'visualizations'

    num_to_show = min(12, len(failures))
    cols = 3
    rows = (num_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f'Failure Analysis (PSNR < {threshold_psnr} dB)',
                 fontsize=14, fontweight='bold')

    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, sample in enumerate(failures[:num_to_show]):
        ax = axes[i]

        viz_file = viz_dir / f"sample_{sample['idx']:04d}.png"
        if viz_file.exists():
            img = plt.imread(str(viz_file))
            ax.imshow(img)

        metrics = sample['metrics']
        ax.set_title(f"#{sample['idx']} | PSNR: {metrics['PSNR']:.2f} dB\n{sample['video']}",
                     fontsize=9, color='red')
        ax.axis('off')

    for i in range(num_to_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Failure analysis saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate visualization grids from batch results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing batch inference results')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['grid', 'dashboard', 'failures', 'all'],
                        help='Visualization mode')
    parser.add_argument('--output_prefix', type=str, default='viz',
                        help='Output filename prefix')

    args = parser.parse_args()

    results_path = Path(args.results_dir)

    if args.mode in ['grid', 'all']:
        create_comparison_grid(
            args.results_dir,
            str(results_path / f'{args.output_prefix}_grid.png')
        )

    if args.mode in ['dashboard', 'all']:
        create_metrics_dashboard(
            args.results_dir,
            str(results_path / f'{args.output_prefix}_dashboard.png')
        )

    if args.mode in ['failures', 'all']:
        create_failure_analysis(
            args.results_dir,
            str(results_path / f'{args.output_prefix}_failures.png')
        )