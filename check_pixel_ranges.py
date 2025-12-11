#!/usr/bin/env python3
"""
Szybki skrypt do sprawdzenia zakresów wartości pikseli w datasecie.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dataset import UCF101Dataset
import numpy as np

def check_raw_video_values(dataset, num_samples=10):
    """Sprawdza wartości przed i po normalizacji."""
    print("\n" + "="*80)
    print("🔍 SPRAWDZANIE WARTOŚCI PIKSELI")
    print("="*80)

    all_frames_max = []
    all_ref_max = []
    all_gt_max = []

    print(f"\nAnalizuję {num_samples} sekwencji...\n")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        frames_max = sample['frames'].max().item()
        ref_max = sample['ref_frames'].max().item()
        gt_max = sample['gt'].max().item()

        all_frames_max.append(frames_max)
        all_ref_max.append(ref_max)
        all_gt_max.append(gt_max)

        print(f"Sekwencja {i+1:2d}: "
              f"frames={frames_max:.4f}  "
              f"ref={ref_max:.4f}  "
              f"gt={gt_max:.4f}")

    print("\n" + "-"*80)
    print("📊 STATYSTYKI:")
    print("-"*80)

    print(f"\nframes:")
    print(f"  • min max: {min(all_frames_max):.4f}")
    print(f"  • max max: {max(all_frames_max):.4f}")
    print(f"  • średnia max: {np.mean(all_frames_max):.4f}")
    print(f"  • Ile ma max=1.0? {sum(1 for x in all_frames_max if x >= 0.999)}/{len(all_frames_max)}")

    print(f"\nref_frames:")
    print(f"  • min max: {min(all_ref_max):.4f}")
    print(f"  • max max: {max(all_ref_max):.4f}")
    print(f"  • średnia max: {np.mean(all_ref_max):.4f}")
    print(f"  • Ile ma max=1.0? {sum(1 for x in all_ref_max if x >= 0.999)}/{len(all_ref_max)}")

    print(f"\ngt:")
    print(f"  • min max: {min(all_gt_max):.4f}")
    print(f"  • max max: {max(all_gt_max):.4f}")
    print(f"  • średnia max: {np.mean(all_gt_max):.4f}")
    print(f"  • Ile ma max=1.0? {sum(1 for x in all_gt_max if x >= 0.999)}/{len(all_gt_max)}")

    print("\n" + "="*80)
    print("💡 INTERPRETACJA:")
    print("="*80)

    if max(all_frames_max) >= 0.999 and max(all_ref_max) >= 0.999:
        print("✅ Wszystko w porządku!")
        print("   Niektóre sekwencje mają max=1.0, inne nie - to normalne.")
        print("   Zależy od jasności sceny w konkretnym wideo.")
    elif max(all_frames_max) < 0.95:
        print("⚠️  Uwaga: Wszystkie wartości są poniżej 0.95")
        print("   Możliwy problem z normalizacją lub bardzo ciemne wideo.")
    else:
        print("✅ Wartości wyglądają prawidłowo.")
        print("   Różnice wynikają z jasności poszczególnych scen.")

    print("\n" + "="*80 + "\n")


def check_specific_sequence(dataset, idx=0):
    """Szczegółowa analiza pojedynczej sekwencji."""
    print("\n" + "="*80)
    print(f"🔬 SZCZEGÓŁOWA ANALIZA SEKWENCJI #{idx}")
    print("="*80)

    sample = dataset[idx]

    # Statystyki dla frames
    frames = sample['frames']
    print(f"\n📊 frames shape: {frames.shape}")
    print(f"  • min:  {frames.min():.6f}")
    print(f"  • max:  {frames.max():.6f}")
    print(f"  • mean: {frames.mean():.6f}")
    print(f"  • std:  {frames.std():.6f}")

    # Policz ile pikseli ma wartość >=0.99
    high_vals = (frames >= 0.99).sum().item()
    total_vals = frames.numel()
    print(f"  • Pikseli >=0.99: {high_vals} / {total_vals} ({100*high_vals/total_vals:.2f}%)")

    # To samo dla ref_frames
    ref_frames = sample['ref_frames']
    print(f"\n📊 ref_frames shape: {ref_frames.shape}")
    print(f"  • min:  {ref_frames.min():.6f}")
    print(f"  • max:  {ref_frames.max():.6f}")
    print(f"  • mean: {ref_frames.mean():.6f}")
    print(f"  • std:  {ref_frames.std():.6f}")

    high_vals_ref = (ref_frames >= 0.99).sum().item()
    total_vals_ref = ref_frames.numel()
    print(f"  • Pikseli >=0.99: {high_vals_ref} / {total_vals_ref} ({100*high_vals_ref/total_vals_ref:.2f}%)")

    # GT
    gt = sample['gt']
    print(f"\n📊 gt shape: {gt.shape}")
    print(f"  • min:  {gt.min():.6f}")
    print(f"  • max:  {gt.max():.6f}")
    print(f"  • mean: {gt.mean():.6f}")
    print(f"  • std:  {gt.std():.6f}")

    high_vals_gt = (gt >= 0.99).sum().item()
    total_vals_gt = gt.numel()
    print(f"  • Pikseli >=0.99: {high_vals_gt} / {total_vals_gt} ({100*high_vals_gt/total_vals_gt:.2f}%)")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'x4k'])
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--detailed', type=int, default=None,
                        help='Index of sequence for detailed analysis')
    args = parser.parse_args()

    print(f"\n🔬 Ładuję dataset: {args.dataset}")

    if args.dataset == 'ucf101':
        from dataset import UCF101Dataset
        dataset = UCF101Dataset(mode='train', num_frames=15, max_sequences=100)
    else:
        from dataset import X4K1000FPSDataset
        from configs.default import Config
        config = Config()
        dataset = X4K1000FPSDataset(
            data_root=config.data_root,
            mode='train',
            num_frames=15,
            max_sequences=100
        )

    # Sprawdź wiele sekwencji
    check_raw_video_values(dataset, num_samples=args.num_samples)

    # Opcjonalnie szczegółowa analiza
    if args.detailed is not None:
        check_specific_sequence(dataset, idx=args.detailed)

    print("💡 Tip: Uruchom z --detailed 0 aby zobaczyć szczegóły pojedynczej sekwencji")
