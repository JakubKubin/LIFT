"""
PRZYKŁAD: Jak dodać inspekcję danych do train.py

Ten plik pokazuje jak zintegrować data_inspector z train.py
aby mieć lepszą kontrolę nad danymi wejściowymi.
"""

# ============================================================================
# OPCJA 1: Dodanie inspekcji na początku train.py
# ============================================================================

def main_with_inspection():
    """
    Dodaj ten kod w train.py po stworzeniu datasetu (około linii 280).
    """
    import argparse
    from dataset import UCF101Dataset, X4K1000FPSDataset
    from configs.default import Config
    from utils.data_inspector import (
        print_dataset_stats,
        visualize_model_inputs,
        inspect_batch
    )
    from torch.utils.data import DataLoader
    from dataset import collate_fn

    # ... (twój istniejący kod argparse i config) ...
    config = Config()

    # Stwórz dataset (twój istniejący kod)
    train_dataset = UCF101Dataset(
        data_root=config.data_root,
        mode='train',
        num_frames=config.num_frames,
        crop_size=config.crop_size,
        augment=True,
        max_sequences=config.max_sequences
    )

    val_dataset = UCF101Dataset(
        data_root=config.data_root,
        mode='val',
        num_frames=config.num_frames,
        crop_size=config.crop_size,
        augment=False,
        max_sequences=config.max_val_sequences
    )

    # ========================================================================
    # DODAJ TO: Inspekcja danych przed treningiem
    # ========================================================================

    print("\n" + "="*80)
    print("🔬 DATA INSPECTION MODE")
    print("="*80)

    # 1. Wypisz statystyki dla train i val
    print_dataset_stats(train_dataset, name="Training Dataset")
    print_dataset_stats(val_dataset, name="Validation Dataset")

    # 2. Wizualizuj przykładowe sekwencje (tylko przy pierwszym uruchomieniu)
    # Możesz to skomentować po pierwszej weryfikacji
    visualize_model_inputs(
        train_dataset,
        num_sequences=3,
        output_dir='inspection/train',
        show_all_frames=False
    )

    visualize_model_inputs(
        val_dataset,
        num_sequences=3,
        output_dir='inspection/val',
        show_all_frames=False
    )

    # 3. Sprawdź batch z DataLoadera
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    print("\n📦 Checking first training batch...")
    first_batch = next(iter(train_loader))
    inspect_batch(first_batch, batch_idx=0)

    print("\n✅ Data inspection complete! Check 'inspection/' folder for visualizations.")
    print("="*80 + "\n")

    # Dalej normalny trening...
    # ... (reszta kodu train.py) ...


# ============================================================================
# OPCJA 2: Warunkowa inspekcja (tylko gdy flaga --inspect_data)
# ============================================================================

def main_with_optional_inspection():
    """
    Dodaj argument --inspect_data do argparse i włączaj inspekcję opcjonalnie.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train LIFT model')
    # ... (twoje istniejące argumenty) ...

    # DODAJ TO:
    parser.add_argument('--inspect_data', action='store_true',
                        help='Run data inspection before training')
    parser.add_argument('--inspect_sequences', type=int, default=3,
                        help='Number of sequences to visualize in inspection')

    args = parser.parse_args()

    # ... (stwórz datasety) ...

    # DODAJ TO:
    if args.inspect_data:
        from utils.data_inspector import print_dataset_stats, visualize_model_inputs

        print("\n🔬 Running data inspection...")
        print_dataset_stats(train_dataset, name="Training Dataset")
        visualize_model_inputs(
            train_dataset,
            num_sequences=args.inspect_sequences,
            output_dir='inspection/train'
        )
        print("✅ Inspection complete! Starting training...\n")

    # Normalny trening...


# ============================================================================
# OPCJA 3: Funkcja callback do debugowania w trakcie treningu
# ============================================================================

def debug_batch_callback(batch, step, output_dir='debug_batches'):
    """
    Wywołaj tę funkcję w pętli treningowej aby debugować problemy.

    Użycie w train.py (w funkcji train_epoch):
        if step % 100 == 0:  # Co 100 kroków
            debug_batch_callback(batch, step)
    """
    from utils.data_inspector import inspect_batch
    import os

    print(f"\n🐛 DEBUG: Batch at step {step}")
    inspect_batch(batch, batch_idx=step)


# ============================================================================
# OPCJA 4: Standalone script do inspekcji istniejącego checkpointa
# ============================================================================

def inspect_checkpoint_data():
    """
    Sprawdź dane używane przez zapisany checkpoint.
    """
    import torch
    from configs.default import Config
    from dataset import UCF101Dataset
    from utils.data_inspector import print_dataset_stats, visualize_model_inputs

    # Załaduj config z checkpointa
    checkpoint_path = 'checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Odtwórz dataset z tą samą konfiguracją
        dataset = UCF101Dataset(
            data_root=config_dict.get('data_root'),
            num_frames=config_dict.get('num_frames', 15),
            crop_size=config_dict.get('crop_size', (224, 224)),
            mode='train'
        )

        print(f"📊 Inspecting data from checkpoint: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")

        print_dataset_stats(dataset, name=f"Dataset from {checkpoint_path}")
        visualize_model_inputs(dataset, num_sequences=5, output_dir='checkpoint_inspection')


# ============================================================================
# PRZYKŁAD: Kompletna integracja w train.py
# ============================================================================

COMPLETE_INTEGRATION_EXAMPLE = """
# W train.py, dodaj na początku (po importach):

from utils.data_inspector import print_dataset_stats, visualize_model_inputs, inspect_batch

# W funkcji main(), po stworzeniu datasetów (po linii 276):

    # ===== DATA INSPECTION =====
    if not args.checkpoint:  # Tylko przy treningu od zera
        print("\\n" + "="*80)
        print("🔬 DATA INSPECTION")
        print("="*80)

        # Statystyki
        print_dataset_stats(train_dataset, name="Training Dataset")
        print_dataset_stats(val_dataset, name="Validation Dataset")

        # Wizualizacja pierwszych sekwencji
        visualize_model_inputs(train_dataset, num_sequences=3,
                               output_dir='inspection/train')
        visualize_model_inputs(val_dataset, num_sequences=3,
                               output_dir='inspection/val')

        # Sprawdź pierwszy batch
        temp_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                 collate_fn=collate_fn)
        inspect_batch(next(iter(temp_loader)), batch_idx=0)

        print("✅ Inspection complete! Check 'inspection/' folder.")
        print("="*80 + "\\n")

        # Opcjonalnie: czekaj na potwierdzenie użytkownika
        # input("Press Enter to start training...")
    # ===========================

    # Dalej normalny kod train.py...
    train_loader = DataLoader(...)
    # itd.

# UŻYCIE:
# python train.py --dataset ucf101    # Automatyczna inspekcja przy pierwszym uruchomieniu
# python train.py --checkpoint ...    # Pomiń inspekcję przy wznowieniu
"""

if __name__ == '__main__':
    print(COMPLETE_INTEGRATION_EXAMPLE)
    print("\n" + "="*80)
    print("💡 INSTRUKCJE:")
    print("="*80)
    print("1. Wybierz jedną z opcji powyżej")
    print("2. Dodaj odpowiedni kod do train.py")
    print("3. Uruchom trening i sprawdź folder 'inspection/'")
    print("4. Zweryfikuj czy dane są ładowane poprawnie")
    print("="*80)
