import torch
import torch.cuda
import gc
from model import LIFT, LIFTLoss
from configs.default import Config

def check_batch_size(max_batch_to_try=32):
    # Configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cpu':
        print("CUDA not available. Memory checking is only relevant for GPU.")
        return

    print(f"Checking max batch size on {torch.cuda.get_device_name(0)}...")

    # Create dummy model and loss
    model = LIFT(config).to(device)
    loss_fn = LIFTLoss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    # Dimensions based on your config (256x256 resolution, 15 frames)
    H, W = 256, 256 # or config.crop_size if different
    T = config.num_frames

    # Binary search or linear scan for max batch size
    for batch_size in range(1, max_batch_to_try + 1):
        try:
            print(f"Testing batch_size = {batch_size}...", end="", flush=True)

            # Clear memory from previous iteration
            gc.collect()
            torch.cuda.empty_cache()

            # Reset max memory stats
            torch.cuda.reset_peak_memory_stats()

            # Create dummy batch
            frames = torch.rand(batch_size, T, 3, H, W).to(device)
            ref_frames = torch.rand(batch_size, 2, 3, H, W).to(device)
            gt = torch.rand(batch_size, 3, H, W).to(device)
            timestep = torch.tensor([0.5] * batch_size).to(device)

            # Forward pass
            outputs = model(frames, ref_frames, timestep[0].item())
            pred = outputs['prediction']

            # Loss computation
            losses = loss_fn(
                pred, gt,
                flow1=outputs['flows']['flow_31'],
                flow2=outputs['flows']['flow_32'],
                occ1=outputs['occlusions']['occ_31'],
                occ2=outputs['occlusions']['occ_32']
            )
            loss = losses['total']

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Get memory usage
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f" OK. Max Alloc: {max_mem:.2f} GB / {total_mem:.2f} GB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(" OOM!")
                print(f"\nMaximum successful batch size: {batch_size - 1}")
                return batch_size - 1
            else:
                print(f" Failed with error: {e}")
                raise e
        except Exception as e:
            print(f" Failed with error: {e}")
            raise e

if __name__ == "__main__":
    check_batch_size()