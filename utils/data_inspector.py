"""
Data Inspector - narzędzie do debugowania i wizualizacji danych wejściowych modelu LIFT.

Funkcje:
- print_dataset_stats: Wypisuje szczegółowe statystyki datasetu
- visualize_model_inputs: Wizualizuje dokładnie co wchodzi do modelu
- inspect_batch: Analizuje batch z DataLoadera
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional
import os


def print_dataset_stats(dataset, name: str = "Dataset"):
    """
    Wypisuje szczegółowe statystyki datasetu.

    Args:
        dataset: BaseVideoDataset instance
        name: Nazwa datasetu do wyświetlenia
    """
    print("\n" + "="*80)
    print(f"📊 STATYSTYKI: {name}")
    print("="*80)

    # Podstawowe info
    print(f"\n🔢 Podstawowe informacje:")
    print(f"  • Liczba sekwencji: {len(dataset)}")
    print(f"  • Tryb: {dataset.mode}")
    print(f"  • Liczba klatek na sekwencję: {dataset.num_frames}")
    print(f"  • Typ (parzyste/nieparzyste): {'NIEPARZYSTE' if dataset.is_odd else 'PARZYSTE'}")
    print(f"  • Rozmiar crop: {dataset.crop_size}")
    print(f"  • Augmentacja: {'TAK' if dataset.augment else 'NIE'}")
    print(f"  • Input scale: {dataset.input_scale}")
    print(f"  • Stride: {dataset.stride}")
    print(f"  • Min motion threshold: {dataset.min_motion_threshold}")

    # Struktura danych
    print(f"\n🎯 Struktura danych wejściowych:")
    print(f"  • Indeks środkowy (mid_idx): {dataset.mid_idx}")
    print(f"  • Indeksy klatek referencyjnych: {dataset.ref_source_idx}")
    print(f"  • Timestep: {dataset.target_timestep}")

    if dataset.is_odd:
        print(f"\n📐 Konfiguracja dla NIEPARZYSTYCH klatek ({dataset.num_frames}):")
        print(f"  • Ground Truth: klatka nr {dataset.mid_idx}")
        print(f"  • Reference frames: klatki nr {dataset.ref_source_idx}")
        excluded_frames = [dataset.mid_idx]
        input_frames = [i for i in range(dataset.num_frames) if i not in excluded_frames]
        print(f"  • Input frames (do modelu): {input_frames}")
        print(f"  • Liczba input frames: {len(input_frames)}")
    else:
        print(f"\n📐 Konfiguracja dla PARZYSTYCH klatek ({dataset.num_frames}):")
        print(f"  • Ground Truth: interpolacja między klatkami {dataset.ref_source_idx}")
        print(f"  • Reference frames: klatki nr {dataset.ref_source_idx}")
        print(f"  • Input frames (do modelu): wszystkie {dataset.num_frames} klatek")

    # Przykładowa sekwencja
    if len(dataset) > 0:
        print(f"\n🔍 Testowanie pierwszej sekwencji...")
        sample = dataset[0]
        print(f"  ✓ frames shape: {sample['frames'].shape} (T, C, H, W)")
        print(f"  ✓ ref_frames shape: {sample['ref_frames'].shape} (2, C, H, W)")
        print(f"  ✓ gt shape: {sample['gt'].shape} (C, H, W)")
        print(f"  ✓ timestep: {sample['timestep'].item()}")

        print(f"\n📊 Zakresy wartości (0-1):")
        print(f"  • frames: [{sample['frames'].min():.3f}, {sample['frames'].max():.3f}]")
        print(f"  • ref_frames: [{sample['ref_frames'].min():.3f}, {sample['ref_frames'].max():.3f}]")
        print(f"  • gt: [{sample['gt'].min():.3f}, {sample['gt'].max():.3f}]")

    # Przykłady ścieżek wideo
    print(f"\n📁 Przykładowe ścieżki wideo:")
    for i in range(min(3, len(dataset.video_list))):
        video_path, start_frame = dataset.video_list[i]
        video_name = Path(video_path).name
        print(f"  {i+1}. {video_name} (start: {start_frame})")

    print("\n" + "="*80 + "\n")


def visualize_model_inputs(
    dataset,
    num_sequences: int = 3,
    output_dir: str = "data_inspection",
    show_all_frames: bool = False
):
    """
    Wizualizuje dokładnie co wchodzi do modelu dla n sekwencji.

    Args:
        dataset: BaseVideoDataset instance
        num_sequences: Ile sekwencji zwizualizować
        output_dir: Folder na wyniki
        show_all_frames: Czy pokazać wszystkie klatki czy tylko kluczowe
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🎨 Generowanie wizualizacji dla {num_sequences} sekwencji...")
    print(f"📂 Zapisuję do: {output_dir}/")

    num_sequences = min(num_sequences, len(dataset))

    for seq_idx in range(num_sequences):
        sample = dataset[seq_idx]
        video_path, start_frame = dataset.video_list[seq_idx]
        video_name = Path(video_path).stem

        print(f"\n  Sekwencja {seq_idx+1}/{num_sequences}: {video_name} (start: {start_frame})")

        # Konwersja do numpy do wyświetlania
        frames = sample['frames']  # (T-1, C, H, W) dla odd lub (T, C, H, W) dla even
        ref_frames = sample['ref_frames']  # (2, C, H, W)
        gt = sample['gt']  # (C, H, W)

        # Do numpy i transpose do (H, W, C)
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        ref_frames_np = ref_frames.permute(0, 2, 3, 1).cpu().numpy()
        gt_np = gt.permute(1, 2, 0).cpu().numpy()

        if show_all_frames:
            _visualize_all_frames(
                frames_np, ref_frames_np, gt_np, dataset,
                seq_idx, video_name, start_frame, output_dir
            )
        else:
            _visualize_key_frames(
                frames_np, ref_frames_np, gt_np, dataset,
                seq_idx, video_name, start_frame, output_dir
            )

    print(f"\n✅ Wizualizacja zapisana w: {output_dir}/")


def _visualize_key_frames(
    frames_np, ref_frames_np, gt_np, dataset,
    seq_idx, video_name, start_frame, output_dir
):
    """Wizualizuje tylko kluczowe klatki (ref + GT + context)."""

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.2)

    # Tytuł główny
    fig.suptitle(
        f'Sekwencja #{seq_idx+1}: {video_name} (start_frame={start_frame})\n'
        f'num_frames={dataset.num_frames}, is_odd={dataset.is_odd}, '
        f'mid_idx={dataset.mid_idx}, ref_idx={dataset.ref_source_idx}',
        fontsize=14, fontweight='bold'
    )

    # Kolorystyka
    color_ref = '#2ecc71'    # zielony dla reference
    color_gt = '#e74c3c'     # czerwony dla GT
    color_input = '#3498db'  # niebieski dla input

    # === RZĄD 1: Reference Frames ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ref_frames_np[0])
    ax1.set_title(f'REF #1\nKlatka {dataset.ref_source_idx[0]}',
                  fontsize=12, fontweight='bold', color=color_ref)
    ax1.axis('off')
    ax1.add_patch(mpatches.Rectangle((0, 0), gt_np.shape[1]-1, gt_np.shape[0]-1,
                                     linewidth=4, edgecolor=color_ref, facecolor='none'))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ref_frames_np[1])
    ax2.set_title(f'REF #2\nKlatka {dataset.ref_source_idx[1]}',
                  fontsize=12, fontweight='bold', color=color_ref)
    ax2.axis('off')
    ax2.add_patch(mpatches.Rectangle((0, 0), gt_np.shape[1]-1, gt_np.shape[0]-1,
                                     linewidth=4, edgecolor=color_ref, facecolor='none'))

    # === RZĄD 2: Ground Truth ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(gt_np)
    if dataset.is_odd:
        gt_text = f'GROUND TRUTH\nKlatka {dataset.mid_idx}'
    else:
        gt_text = f'GROUND TRUTH\nInterpolacja klatek {dataset.ref_source_idx}'
    ax3.set_title(gt_text, fontsize=12, fontweight='bold', color=color_gt)
    ax3.axis('off')
    ax3.add_patch(mpatches.Rectangle((0, 0), gt_np.shape[1]-1, gt_np.shape[0]-1,
                                     linewidth=4, edgecolor=color_gt, facecolor='none'))

    # === RZĄD 3: Przykładowe Input Frames ===
    # Pokazujemy: pierwszą, ~1/3, ~2/3, ostatnią
    num_input = frames_np.shape[0]
    sample_indices = [
        0,
        num_input // 3,
        2 * num_input // 3,
        num_input - 1
    ]

    # Mapowanie indeksów z input_frames do oryginalnych indeksów
    if dataset.is_odd:
        # Input frames to wszystkie oprócz mid_idx
        original_indices = [i for i in range(dataset.num_frames) if i != dataset.mid_idx]
    else:
        # Input frames to wszystkie
        original_indices = list(range(dataset.num_frames))

    for col_idx, inp_idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[2, col_idx])
        ax.imshow(frames_np[inp_idx])
        orig_idx = original_indices[inp_idx]
        ax.set_title(f'INPUT #{inp_idx}\nOryg klatka {orig_idx}',
                     fontsize=10, color=color_input)
        ax.axis('off')
        ax.add_patch(mpatches.Rectangle((0, 0), gt_np.shape[1]-1, gt_np.shape[0]-1,
                                        linewidth=3, edgecolor=color_input, facecolor='none'))

    # === INFO BOX ===
    info_ax = fig.add_subplot(gs[:, 3:])
    info_ax.axis('off')

    if dataset.is_odd:
        input_list = [i for i in range(dataset.num_frames) if i != dataset.mid_idx]
    else:
        input_list = list(range(dataset.num_frames))

    info_text = f"""
📋 SZCZEGÓŁOWE INFORMACJE

🎯 Konfiguracja sekwencji:
  • Całkowita liczba klatek: {dataset.num_frames}
  • Typ: {'NIEPARZYSTE' if dataset.is_odd else 'PARZYSTE'}
  • Środkowy indeks: {dataset.mid_idx}

🔴 Ground Truth (GT):
  • Indeks klatki: {dataset.mid_idx if dataset.is_odd else f"interpolacja {dataset.ref_source_idx}"}
  • Shape: {gt_np.shape}
  • Zakres wartości: [{gt_np.min():.3f}, {gt_np.max():.3f}]

🟢 Reference Frames (2 klatki):
  • Indeksy: {dataset.ref_source_idx}
  • Shape: {ref_frames_np.shape}
  • Zakres: [{ref_frames_np.min():.3f}, {ref_frames_np.max():.3f}]
  • Użycie: Bazowe klatki dla optical flow

🔵 Input Frames (do modelu):
  • Liczba: {num_input}
  • Indeksy oryginalnych klatek:
    {input_list}
  • Shape: {frames_np.shape}
  • Zakres: [{frames_np.min():.3f}, {frames_np.max():.3f}]

⏱️ Timestep: {dataset.target_timestep}

📐 Wymiary:
  • Crop size: {dataset.crop_size}
  • Input scale: {dataset.input_scale}

💡 Jak to działa:
  1. Model otrzymuje {num_input} klatek (wszystkie oprócz GT)
  2. Reference frames ({dataset.ref_source_idx}) są używane
     do obliczania optical flow
  3. Model interpoluje klatkę {dataset.mid_idx}
  4. GT jest używane tylko do obliczenia loss
"""

    info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes,
                 fontsize=11, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Zapisz
    save_path = os.path.join(output_dir, f'seq_{seq_idx+1:03d}_{video_name}_key_frames.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Zapisano: {save_path}")


def _visualize_all_frames(
    frames_np, ref_frames_np, gt_np, dataset,
    seq_idx, video_name, start_frame, output_dir
):
    """Wizualizuje wszystkie klatki w sekwencji."""

    num_frames = frames_np.shape[0]

    # Dynamiczny układ - maksymalnie 8 kolumn
    ncols = min(8, num_frames + 3)  # +3 dla ref1, ref2, gt
    nrows = int(np.ceil((num_frames + 3) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.5, nrows*3))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'ALL FRAMES - Sekwencja #{seq_idx+1}: {video_name}\n'
        f'num_frames={dataset.num_frames}, mid_idx={dataset.mid_idx}',
        fontsize=14, fontweight='bold'
    )

    # Mapowanie indeksów
    if dataset.is_odd:
        original_indices = [i for i in range(dataset.num_frames) if i != dataset.mid_idx]
    else:
        original_indices = list(range(dataset.num_frames))

    plot_idx = 0

    # Reference frames
    for ref_idx in range(2):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]
        ax.imshow(ref_frames_np[ref_idx])
        ax.set_title(f'REF #{ref_idx+1}\n(Frame {dataset.ref_source_idx[ref_idx]})',
                     fontsize=10, color='green', fontweight='bold')
        ax.axis('off')
        ax.add_patch(mpatches.Rectangle((0, 0), gt_np.shape[1]-1, gt_np.shape[0]-1,
                                        linewidth=3, edgecolor='green', facecolor='none'))
        plot_idx += 1

    # Ground Truth
    row, col = divmod(plot_idx, ncols)
    ax = axes[row, col]
    ax.imshow(gt_np)
    ax.set_title(f'GT\n(Frame {dataset.mid_idx})', fontsize=10, color='red', fontweight='bold')
    ax.axis('off')
    ax.add_patch(mpatches.Rectangle((0, 0), gt_np.shape[1]-1, gt_np.shape[0]-1,
                                    linewidth=3, edgecolor='red', facecolor='none'))
    plot_idx += 1

    # Input frames
    for inp_idx in range(num_frames):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]
        ax.imshow(frames_np[inp_idx])
        orig_idx = original_indices[inp_idx]
        ax.set_title(f'Input #{inp_idx}\n(Frame {orig_idx})', fontsize=9, color='blue')
        ax.axis('off')
        plot_idx += 1

    # Wyłącz pozostałe osie
    for idx in range(plot_idx, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'seq_{seq_idx+1:03d}_{video_name}_all_frames.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Zapisano: {save_path}")


def inspect_batch(batch: Dict[str, torch.Tensor], batch_idx: int = 0):
    """
    Analizuje batch z DataLoadera i wypisuje szczegółowe informacje.

    Args:
        batch: Dict z kluczami: frames, ref_frames, gt, timestep
        batch_idx: Indeks batcha (do wyświetlenia)
    """
    print("\n" + "="*80)
    print(f"🔍 INSPEKCJA BATCHA #{batch_idx}")
    print("="*80)

    batch_size = batch['frames'].shape[0]

    print(f"\n📦 Rozmiary tensors:")
    print(f"  • frames:     {batch['frames'].shape}     (B, T, C, H, W)")
    print(f"  • ref_frames: {batch['ref_frames'].shape} (B, 2, C, H, W)")
    print(f"  • gt:         {batch['gt'].shape}         (B, C, H, W)")
    print(f"  • timestep:   {batch['timestep'].shape}   (B,)")

    print(f"\n📊 Statystyki wartości (0-1):")
    for key in ['frames', 'ref_frames', 'gt']:
        tensor = batch[key]
        print(f"  • {key:12s}: min={tensor.min():.4f}, max={tensor.max():.4f}, "
              f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")

    print(f"\n⏱️ Timesteps:")
    timesteps = batch['timestep'].cpu().numpy()
    print(f"  • Wartości: {timesteps}")
    print(f"  • Wszystkie równe? {np.all(timesteps == timesteps[0])}")

    print(f"\n💾 Pamięć GPU (jeśli na CUDA):")
    if batch['frames'].is_cuda:
        for key in ['frames', 'ref_frames', 'gt']:
            tensor = batch[key]
            memory_mb = tensor.element_size() * tensor.nelement() / (1024**2)
            print(f"  • {key:12s}: {memory_mb:.2f} MB")

    print(f"\n🔢 Batch size: {batch_size}")
    print("="*80 + "\n")


def compare_sequences(dataset, indices: List[int], output_dir: str = "sequence_comparison"):
    """
    Porównuje wiele sekwencji obok siebie (tylko GT + ref frames).

    Args:
        dataset: BaseVideoDataset instance
        indices: Lista indeksów sekwencji do porównania
        output_dir: Folder na wynik
    """
    os.makedirs(output_dir, exist_ok=True)

    num_seqs = len(indices)
    fig, axes = plt.subplots(3, num_seqs, figsize=(num_seqs*4, 12))
    if num_seqs == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle('Porównanie sekwencji - Reference frames + GT', fontsize=16, fontweight='bold')

    for col_idx, seq_idx in enumerate(indices):
        sample = dataset[seq_idx]
        video_path, start_frame = dataset.video_list[seq_idx]
        video_name = Path(video_path).stem

        ref_frames_np = sample['ref_frames'].permute(0, 2, 3, 1).cpu().numpy()
        gt_np = sample['gt'].permute(1, 2, 0).cpu().numpy()

        # Ref 1
        axes[0, col_idx].imshow(ref_frames_np[0])
        axes[0, col_idx].set_title(f'Seq {seq_idx}\n{video_name}\nRef frame {dataset.ref_source_idx[0]}',
                                    fontsize=10)
        axes[0, col_idx].axis('off')

        # GT
        axes[1, col_idx].imshow(gt_np)
        axes[1, col_idx].set_title(f'GT (frame {dataset.mid_idx})', fontsize=10, color='red')
        axes[1, col_idx].axis('off')

        # Ref 2
        axes[2, col_idx].imshow(ref_frames_np[1])
        axes[2, col_idx].set_title(f'Ref frame {dataset.ref_source_idx[1]}', fontsize=10)
        axes[2, col_idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'comparison_{len(indices)}_sequences.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Porównanie zapisane: {save_path}")
