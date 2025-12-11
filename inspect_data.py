#!/usr/bin/env python3
"""
Skrypt do inspekcji i debugowania danych wejściowych do modelu LIFT.

Użycie:
    python inspect_data.py --dataset ucf101 --num_sequences 3
    python inspect_data.py --dataset x4k --num_sequences 5 --show_all_frames
    python inspect_data.py --dataset ucf101 --batch_inspection
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import UCF101Dataset, X4K1000FPSDataset, collate_fn
from configs.default import Config
from utils.data_inspector import (
    print_dataset_stats,
    visualize_model_inputs,
    inspect_batch,
    compare_sequences
)


def get_dataset_class(dataset_name):
    if dataset_name.lower() == 'x4k':
        return X4K1000FPSDataset
    elif dataset_name.lower() == 'ucf101':
        return UCF101Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Inspect LIFT dataset inputs')
    parser.add_argument('--dataset', type=str, default='ucf101',
                        choices=['ucf101', 'x4k'],
                        help='Dataset to inspect')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root')
    parser.add_argument('--num_sequences', type=int, default=3,
                        help='Number of sequences to visualize')
    parser.add_argument('--show_all_frames', action='store_true',
                        help='Show all frames instead of just key frames')
    parser.add_argument('--batch_inspection', action='store_true',
                        help='Also inspect DataLoader batch')
    parser.add_argument('--compare_sequences', type=int, nargs='+',
                        help='Compare specific sequences by indices (e.g., --compare_sequences 0 5 10)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset mode')
    parser.add_argument('--num_frames', type=int, default=15,
                        help='Number of frames per sequence')
    parser.add_argument('--output_dir', type=str, default='data_inspection',
                        help='Output directory for visualizations')
    parser.add_argument('--max_sequences', type=int, default=100,
                        help='Max sequences to load (for faster testing)')

    args = parser.parse_args()

    # Load config
    config = Config()
    if args.data_root:
        config.data_root = args.data_root

    print("\n" + "="*80)
    print("🔬 LIFT DATA INSPECTOR")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Num frames: {args.num_frames}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)

    # Create dataset
    DatasetClass = get_dataset_class(args.dataset)

    try:
        dataset = DatasetClass(
            data_root=config.data_root,
            mode=args.mode,
            num_frames=args.num_frames,
            crop_size=config.crop_size,
            augment=(args.mode == 'train'),
            input_scale=config.input_scale,
            max_sequences=args.max_sequences,
            stride=config.train_stride if args.mode == 'train' else config.val_stride
        )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        return

    # 1. Print statistics
    print_dataset_stats(dataset, name=f"{args.dataset.upper()} ({args.mode})")

    # 2. Visualize sequences
    if args.num_sequences > 0:
        visualize_model_inputs(
            dataset,
            num_sequences=args.num_sequences,
            output_dir=args.output_dir,
            show_all_frames=args.show_all_frames
        )

    # 3. Compare specific sequences
    if args.compare_sequences:
        print(f"\n🔄 Porównuję sekwencje: {args.compare_sequences}")
        compare_sequences(dataset, args.compare_sequences, args.output_dir)

    # 4. Batch inspection
    if args.batch_inspection:
        print(f"\n📦 Tworzę DataLoader i inspekcję batcha...")

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        # Get first batch
        batch = next(iter(dataloader))
        inspect_batch(batch, batch_idx=0)

        # Optionally inspect on GPU
        if torch.cuda.is_available():
            print("\n🖥️  Sprawdzam batch na GPU...")
            batch_gpu = {k: v.cuda() for k, v in batch.items()}
            inspect_batch(batch_gpu, batch_idx=0)

    print("\n" + "="*80)
    print("✅ INSPEKCJA ZAKOŃCZONA")
    print("="*80)
    print(f"\n📂 Wyniki zapisane w: {args.output_dir}/")
    print("\n💡 Wskazówki:")
    print("  • Sprawdź wygenerowane obrazki aby zweryfikować poprawność danych")
    print("  • Reference frames powinny być sąsiadami GT")
    print("  • Input frames nie zawierają GT (dla nieparzystych)")
    print("  • Wszystkie wartości powinny być w zakresie [0, 1]")
    print("")


if __name__ == '__main__':
    main()
