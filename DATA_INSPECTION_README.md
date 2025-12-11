# 📊 Data Inspection & Debugging - LIFT

Narzędzia do szczegółowej inspekcji i wizualizacji danych wejściowych do modelu LIFT.

## 🎯 Cel

Ten moduł pomaga:
- ✅ Zweryfikować poprawność ładowania danych
- ✅ Zrozumieć strukturę danych wejściowych
- ✅ Sprawdzić która klatka jest GT, które są reference
- ✅ Zwizualizować augmentacje
- ✅ Debugować problemy z danymi

## 📁 Pliki

```
utils/data_inspector.py  - Moduł z funkcjami do inspekcji
inspect_data.py         - Skrypt CLI do uruchomienia inspekcji
```

## 🚀 Jak używać

### 1. Podstawowa inspekcja (3 sekwencje)

```bash
python inspect_data.py --dataset ucf101 --num_sequences 3
```

**Wynik:**
- Statystyki datasetu wypisane w konsoli
- Obrazki w folderze `data_inspection/` pokazujące:
  - Reference frames (zielone ramki)
  - Ground Truth (czerwona ramka)
  - Przykładowe input frames (niebieskie ramki)
  - Szczegółowe informacje o strukturze danych

### 2. Inspekcja z większą liczbą sekwencji

```bash
python inspect_data.py --dataset ucf101 --num_sequences 10
```

### 3. Pokazanie WSZYSTKICH klatek (nie tylko kluczowych)

```bash
python inspect_data.py --dataset ucf101 --num_sequences 3 --show_all_frames
```

### 4. Inspekcja batcha z DataLoadera

```bash
python inspect_data.py --dataset ucf101 --batch_inspection
```

**Wynik:**
- Rozmiary tensors (B, T, C, H, W)
- Zakresy wartości (min/max/mean/std)
- Zużycie pamięci GPU
- Weryfikacja timesteps

### 5. Porównanie konkretnych sekwencji

```bash
python inspect_data.py --dataset ucf101 --compare_sequences 0 5 10 15
```

### 6. Własny output folder

```bash
python inspect_data.py --dataset ucf101 --output_dir my_debug_folder
```

### 7. X4K dataset

```bash
python inspect_data.py --dataset x4k --num_sequences 5 --data_root /path/to/X4K1000FPS
```

### 8. Walidacja dataset

```bash
python inspect_data.py --dataset ucf101 --mode val --num_sequences 5
```

## 📖 Użycie w kodzie Python

### Import i podstawowe użycie

```python
from dataset import UCF101Dataset
from utils.data_inspector import (
    print_dataset_stats,
    visualize_model_inputs,
    inspect_batch
)

# Załaduj dataset
dataset = UCF101Dataset(
    data_root='data/UCF-101',
    mode='train',
    num_frames=15,
    max_sequences=100
)

# 1. Wypisz statystyki
print_dataset_stats(dataset, name="UCF101 Train")

# 2. Wizualizuj pierwsze 3 sekwencje
visualize_model_inputs(
    dataset,
    num_sequences=3,
    output_dir='my_inspection'
)

# 3. Inspekcja batcha
from torch.utils.data import DataLoader
from dataset import collate_fn

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))
inspect_batch(batch, batch_idx=0)
```

### Integracja z train.py

Możesz dodać inspekcję na początku treningu w `train.py`:

```python
# W train.py, po stworzeniu datasetu (około linii 280)
from utils.data_inspector import print_dataset_stats, visualize_model_inputs

# Wypisz statystyki
print_dataset_stats(train_dataset, name="Training Dataset")
print_dataset_stats(val_dataset, name="Validation Dataset")

# Wizualizuj przykłady (tylko przy pierwszym uruchomieniu)
if not args.checkpoint:  # Tylko przy treningu od zera
    visualize_model_inputs(train_dataset, num_sequences=3, output_dir='train_inspection')
    visualize_model_inputs(val_dataset, num_sequences=3, output_dir='val_inspection')
```

## 📊 Co pokazują wizualizacje?

### Struktura dla num_frames=15 (NIEPARZYSTE)

```
Oryginalne klatki:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                                          ↑
                                         GT (mid_idx=7)

Reference frames:   [6, 8]              (sąsiedzi GT)
Input frames:       [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]  (bez GT!)
Ground Truth:       [7]                 (do obliczenia loss)
```

### Wizualizacja pokazuje:

1. **🟢 Reference Frames (2 klatki)** - klatki 6 i 8
   - Używane jako baza do optical flow
   - Sąsiedzi klatki GT

2. **🔴 Ground Truth (1 klatka)** - klatka 7
   - Środkowa klatka
   - Używana tylko do loss (NIE wchodzi do modelu!)

3. **🔵 Input Frames (14 klatek)** - [0,1,2,3,4,5,6,8,9,10,11,12,13,14]
   - Wszystkie klatki OPRÓCZ GT
   - To dostaje model jako input

4. **📋 Info Box** - szczegółowe informacje:
   - Konfiguracja sekwencji
   - Indeksy klatek
   - Shape tensorów
   - Zakresy wartości

## 🔍 Co sprawdzić w wizualizacjach?

### ✅ Checklist weryfikacji:

- [ ] Reference frames są sąsiadami GT (indeksy 6 i 8 dla mid_idx=7)
- [ ] GT wygląda na sensowną interpolację między ref frames
- [ ] Input frames NIE zawierają GT (brakuje klatki 7)
- [ ] Wszystkie obrazki mają ten sam rozmiar (crop_size)
- [ ] Wartości są w zakresie [0, 1]
- [ ] Augmentacje działają poprawnie (flip, rotate, crop)
- [ ] Brak artefaktów/błędów w klatkach

### ❌ Czerwone flagi:

- ❌ GT jest identyczne z którąś z ref frames → problem z indeksowaniem
- ❌ Input frames zawierają GT → błąd w logice
- ❌ Wartości poza zakresem [0, 1] → problem z normalizacją
- ❌ Różne rozmiary klatek → problem z crop/resize
- ❌ Artefakty w obrazkach → problem z wczytywaniem wideo

## 🎨 Kolory w wizualizacjach

- 🟢 **Zielona ramka** - Reference frames (bazowe dla optical flow)
- 🔴 **Czerwona ramka** - Ground Truth (target do predykcji)
- 🔵 **Niebieska ramka** - Input frames (wchodzą do modelu)

## 📈 Przykładowy output (konsola)

```
================================================================================
📊 STATYSTYKI: UCF101 (train)
================================================================================

🔢 Podstawowe informacje:
  • Liczba sekwencji: 1000
  • Tryb: train
  • Liczba klatek na sekwencję: 15
  • Typ (parzyste/nieparzyste): NIEPARZYSTE
  • Rozmiar crop: (224, 224)
  • Augmentacja: TAK
  • Input scale: 1.0
  • Stride: 1

🎯 Struktura danych wejściowych:
  • Indeks środkowy (mid_idx): 7
  • Indeksy klatek referencyjnych: [6, 8]
  • Timestep: 0.5

📐 Konfiguracja dla NIEPARZYSTYCH klatek (15):
  • Ground Truth: klatka nr 7
  • Reference frames: klatki nr [6, 8]
  • Input frames (do modelu): [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
  • Liczba input frames: 14

🔍 Testowanie pierwszej sekwencji...
  ✓ frames shape: torch.Size([14, 3, 224, 224]) (T, C, H, W)
  ✓ ref_frames shape: torch.Size([2, 3, 224, 224]) (2, C, H, W)
  ✓ gt shape: torch.Size([3, 224, 224]) (C, H, W)
  ✓ timestep: 0.5

📊 Zakresy wartości (0-1):
  • frames: [0.000, 1.000]
  • ref_frames: [0.000, 1.000]
  • gt: [0.000, 1.000]
================================================================================
```

## 💡 Wskazówki

### Szybka weryfikacja przed treningiem:

```bash
# 1. Sprawdź czy dane się ładują
python inspect_data.py --dataset ucf101 --num_sequences 1

# 2. Zweryfikuj augmentacje
python inspect_data.py --dataset ucf101 --num_sequences 3 --mode train

# 3. Sprawdź batch
python inspect_data.py --dataset ucf101 --batch_inspection
```

### Debug konkretnego problemu:

```python
# Jeśli podejrzewasz problem z konkretną sekwencją
from dataset import UCF101Dataset
from utils.data_inspector import visualize_model_inputs

dataset = UCF101Dataset(mode='train', max_sequences=1000)

# Wizualizuj podejrzane sekwencje
visualize_model_inputs(dataset, num_sequences=1, output_dir='debug_seq_42')
```

### Porównanie train vs val:

```bash
python inspect_data.py --dataset ucf101 --mode train --num_sequences 5 --output_dir train_vis
python inspect_data.py --dataset ucf101 --mode val --num_sequences 5 --output_dir val_vis
```

## 🐛 Troubleshooting

**Problem:** `ValueError: No videos found`
```bash
# Sprawdź ścieżkę do danych
python inspect_data.py --dataset ucf101 --data_root /correct/path/to/UCF-101
```

**Problem:** `Out of memory` przy dużej liczbie sekwencji
```bash
# Ogranicz liczbę
python inspect_data.py --dataset ucf101 --num_sequences 3 --max_sequences 100
```

**Problem:** Wizualizacje są puste/czarne
```python
# Sprawdź zakresy wartości
from utils.data_inspector import inspect_batch
# ... (zobacz przykład wyżej)
```

## 📚 API Reference

### `print_dataset_stats(dataset, name)`
Wypisuje szczegółowe statystyki datasetu.

### `visualize_model_inputs(dataset, num_sequences, output_dir, show_all_frames)`
Tworzy wizualizacje pokazujące co wchodzi do modelu.

### `inspect_batch(batch, batch_idx)`
Analizuje batch z DataLoadera (shapes, wartości, pamięć).

### `compare_sequences(dataset, indices, output_dir)`
Porównuje wiele sekwencji obok siebie.

## 🎓 Zrozumienie struktury danych

Model LIFT działa na zasadzie **frame interpolation**:

1. **Dostaje:** N-1 klatek (dla N=15 → 14 klatek)
2. **Przewiduje:** Brakującą środkową klatkę
3. **Używa:** 2 reference frames do obliczenia optical flow
4. **Sprawdza:** Prediction vs Ground Truth (loss)

To podejście wymusza na modelu naukę:
- Ruchu (optical flow)
- Okluzji (co jest zasłonięte)
- Syntezy (jak połączyć informacje)

---

**Autor:** Claude
**Data:** 2025-12-11
**Wersja:** 1.0
