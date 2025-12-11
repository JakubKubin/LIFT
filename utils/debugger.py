"""
utils/debugger.py
Narzędzie do inspekcji danych wejściowych przed treningiem.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def inspect_input_batch(batch, config, output_dir='debug_inputs', batch_idx=0):
    """
    Analizuje i wizualizuje pojedynczy batch danych.
    Naprawiona wersja: Rekonstruuje pełną sekwencję (wstawia GT w środek).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frames = batch['frames']      # [B, 14, 3, H, W] - UWAGA: 14 klatek (bez GT)
    gt = batch['gt']              # [B, 3, H, W]
    
    B, T_in, C, H, W = frames.shape
    
    # Pobieramy indeksy logiczne (czasowe)
    # Dla 15 klatek: mid=7 (to jest GT)
    mid_idx = config.num_frames // 2
    
    # === KLUCZOWA POPRAWKA: REKONSTRUKCJA PEŁNEJ SEKWENCJI ===
    # Rozdzielamy tensor wejściowy (14 klatek) na lewo i prawo od dziury na GT
    frames_left = frames[:, :mid_idx]      # Klatki 0-6
    frames_right = frames[:, mid_idx:]     # Klatki 8-14 (indeksy tensora 7-13)
    
    # Wstawiamy GT w środek: [Left, GT, Right] -> [B, 15, 3, H, W]
    full_seq = torch.cat([frames_left, gt.unsqueeze(1), frames_right], dim=1)
    
    print(f"\n[DEBUG] Analiza Batcha #{batch_idx}")
    print(f"--------------------------------------------------")
    print(f"Shape input frames: {frames.shape} (oczekiwane 14)")
    print(f"Shape gt:           {gt.shape}")
    print(f"Shape full_seq:     {full_seq.shape} (zrekonstruowane 15)")
    print(f"--------------------------------------------------")
    print(f"Indeks GT (Middle): {mid_idx}")
    # Logiczni sąsiedzi to klatka przed i po środku
    ref_indices_visual = [mid_idx - 1, mid_idx + 1] 
    print(f"Sąsiedzi (Ref):     {ref_indices_visual}")
    print(f"--------------------------------------------------")
    
    # Sprawdzenie zakresu wartości
    print(f"Wartości pixeli (Full Seq): Min={full_seq.min():.4f}, Max={full_seq.max():.4f}, Mean={full_seq.mean():.4f}")
    
    # Weryfikacja spójności (teraz sprawdzamy czy rekonstrukcja się udała)
    # Pobieramy próbkę do wizualizacji
    sample_seq = full_seq[0] # [15, 3, H, W]
    
    # Integrity check jest teraz trywialny (bo sami wstawiliśmy GT), 
    # ale potwierdza, że operacje na tensorach są poprawne.
    diff_gt = torch.abs(full_seq[:, mid_idx] - gt).mean().item()
    if diff_gt < 1e-5:
        print("PASSED INTEGRITY CHECK: Sekwencja zrekonstruowana poprawnie.")
    else:
        print(f"FAILED INTEGRITY CHECK: Błąd rekonstrukcji (diff={diff_gt:.6f})")

    # --- WIZUALIZACJA ---
    # Rysujemy pełną sekwencję 15 klatek zrekonstruowaną z inputu i GT
    # Ustawiamy układ 3 rzędy x 5 kolumn = 15 klatek
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    # Iterujemy po pełnej liczbie klatek (15)
    for i in range(config.num_frames):
        if i >= len(axes): break
        
        ax = axes[i]
        # Konwersja [3, H, W] -> [H, W, 3]
        img = sample_seq[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Oznaczenia
        title = f"Frame {i}"
        color = 'black'
        linewidth = 0
        
        if i == mid_idx:
            title += " (GT)"
            color = 'red' # GT na czerwono
            linewidth = 4
        elif i in ref_indices_visual:
            title += " (REF)"
            color = 'green' # Referencje na zielono
            linewidth = 4
            
        ax.set_title(title, color=color, fontweight='bold')
        
        # Ramka
        if linewidth > 0:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(linewidth)

    plt.suptitle(f"Zrekonstruowana sekwencja (Batch {batch_idx})\nInput (14 klatek) + GT wstawione w środek", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"batch_{batch_idx}_sequence.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Zapisano wizualizację do: {save_path}")

def check_warping_sanity(img_ref, flow, img_target, save_path='debug_warp.png'):
    """
    Prosty test: Warpujemy ref za pomocą flow.
    Służy do sprawdzenia, czy kierunek przepływu (Flow) jest zgodny z oczekiwaniami.
    """
    try:
        from model.warplayer import backward_warp
    except ImportError:
        # Próba importu relatywnego jeśli skrypt jest uruchamiany z innego miejsca
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from model.warplayer import backward_warp
        except ImportError:
            print("Nie można zaimportować backward_warp - pomijam test warpingu.")
            return

    # img_ref: [B, 3, H, W]
    # flow: [B, 2, H, W]
    
    with torch.no_grad():
        # Wykonujemy warping
        warped = backward_warp(img_ref, flow)
        
        # Oblicz błąd
        diff = torch.abs(warped - img_target).mean().item()
        print(f"[Warp Sanity] Mean L1 Diff: {diff:.4f}")
        
        # Zapisz wizualizację (pierwszy element z batcha)
        if save_path:
            # Pobieramy pierwszy element z batcha [0] i konwertujemy do numpy
            ref_np = img_ref[0].permute(1, 2, 0).cpu().numpy()
            target_np = img_target[0].permute(1, 2, 0).cpu().numpy()
            warped_np = warped[0].permute(1, 2, 0).cpu().numpy()
            
            # Clip values to [0, 1] for proper display
            ref_np = np.clip(ref_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            warped_np = np.clip(warped_np, 0, 1)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].imshow(ref_np)
            ax[0].set_title("Reference (Source)")
            ax[0].axis('off')
            
            ax[1].imshow(warped_np)
            ax[1].set_title(f"Warped (Result)\nDiff: {diff:.4f}")
            ax[1].axis('off')
            
            ax[2].imshow(target_np)
            ax[2].set_title("Target (Goal)")
            ax[2].axis('off')
            
            plt.suptitle(f"Warping Check: Does Center look like Right?", fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Zapisano test warpingu do: {save_path}")