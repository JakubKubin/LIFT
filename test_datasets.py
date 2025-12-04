"""
Example script demonstrating X4K1000FPS and UCF-101 dataset usage.
"""

import torch
from torch.utils.data import DataLoader
import argparse

from dataset import (
    X4K1000FPSDataset,
    UCF101Dataset,
    collate_fn
)


def test_x4k1000fps(data_root='/data/X4K1000FPS', batch_size=2, num_samples=5):
    """Test X4K1000FPS dataset."""
    print("\n" + "="*60)
    print("Testing X4K1000FPS Dataset")
    print("="*60)

    try:
        # Create dataset
        dataset = X4K1000FPSDataset(
            data_root=data_root,
            mode='train',
            num_frames=15,
            crop_size=(224, 224),
            augment=True,
            cache_frames=False
        )

        print(f"\nDataset created successfully!")
        print(f"Total sequences: {len(dataset)}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )

        print(f"\nLoading {num_samples} batches...")

        import time
        total_time = 0

        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            start = time.time()

            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            frames = batch['frames'].to(device)
            ref_frames = batch['ref_frames'].to(device)
            gt = batch['gt'].to(device)

            load_time = time.time() - start
            total_time += load_time

            print(f"\nBatch {i+1}:")
            print(f"  Time: {load_time:.3f}s")
            print(f"  Frames: {frames.shape}")
            print(f"  Ref frames: {ref_frames.shape}")
            print(f"  GT: {gt.shape}")
            print(f"  Memory: {frames.element_size() * frames.nelement() / 1e6:.1f} MB")

        avg_time = total_time / num_samples
        print(f"\nAverage loading time: {avg_time:.3f}s per batch")
        print(f"Throughput: {batch_size / avg_time:.1f} sequences/second")

        print("\nX4K1000FPS test PASSED!")
        return True

    except Exception as e:
        print(f"\nX4K1000FPS test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ucf101(data_root='/data/UCF-101', batch_size=2, num_samples=5):
    """Test UCF-101 dataset."""
    print("\n" + "="*60)
    print("Testing UCF-101 Dataset")
    print("="*60)

    try:
        # Create dataset
        dataset = UCF101Dataset(
            data_root=data_root,
            mode='train',
            num_frames=15,
            crop_size=(224, 224),
            augment=True,
            cache_frames=False
        )

        print(f"\nDataset created successfully!")
        print(f"Total sequences: {len(dataset)}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )

        print(f"\nLoading {num_samples} batches...")

        import time
        total_time = 0

        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            start = time.time()

            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            frames = batch['frames'].to(device)
            ref_frames = batch['ref_frames'].to(device)
            gt = batch['gt'].to(device)

            load_time = time.time() - start
            total_time += load_time

            print(f"\nBatch {i+1}:")
            print(f"  Time: {load_time:.3f}s")
            print(f"  Frames: {frames.shape}")
            print(f"  Ref frames: {ref_frames.shape}")
            print(f"  GT: {gt.shape}")
            print(f"  Memory: {frames.element_size() * frames.nelement() / 1e6:.1f} MB")

        avg_time = total_time / num_samples
        print(f"\nAverage loading time: {avg_time:.3f}s per batch")
        print(f"Throughput: {batch_size / avg_time:.1f} sequences/second")

        print("\nUCF-101 test PASSED!")
        return True

    except Exception as e:
        print(f"\nUCF-101 test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test video datasets')
    parser.add_argument('--dataset', type=str, choices=['x4k', 'ucf101', 'both'],
                       default='both', help='Which dataset to test')
    parser.add_argument('--x4k_root', type=str, default='/data/X4K1000FPS',
                       help='X4K1000FPS root directory')
    parser.add_argument('--ucf101_root', type=str, default='/data/UCF-101',
                       help='UCF-101 root directory')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of batches to test')
    args = parser.parse_args()

    print("="*60)
    print("LIFT Dataset Testing")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    results = {}

    # Test X4K1000FPS
    if args.dataset in ['x4k', 'both']:
        results['x4k'] = test_x4k1000fps(
            args.x4k_root,
            args.batch_size,
            args.num_samples
        )

    # Test UCF-101
    if args.dataset in ['ucf101', 'both']:
        results['ucf101'] = test_ucf101(
            args.ucf101_root,
            args.batch_size,
            args.num_samples
        )

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for dataset_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{dataset_name.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed! Datasets are ready for training.")
    else:
        print("\nSome tests failed. Please check the error messages above.")

    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
