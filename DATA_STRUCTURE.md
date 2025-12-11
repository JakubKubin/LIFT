# 📐 Struktura danych LIFT - Szczegółowy opis

## 🎯 Przegląd wysokopoziomowy

LIFT to model do **frame interpolation** - przewiduje brakującą środkową klatkę na podstawie otaczających klatek.

```
Input:  [Frame 0, 1, 2, ..., 6, ❌, 8, ..., 14]   (14 klatek - brakuje środkowej)
Output: [Frame 7] ✓                               (przewidziana środkowa klatka)
GT:     [Frame 7]                                 (prawdziwa środkowa klatka - do loss)
```

---

## 📊 Struktura dla num_frames=15 (NIEPARZYSTE)

### Indeksowanie klatek

```
Oryginalna sekwencja wideo (15 klatek):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬────┬────┬────┬────┬────┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ 10 │ 11 │ 12 │ 13 │ 14 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴────┴────┴────┴────┴────┘
                          ↑   ↑   ↑
                          │   │   └─ ref_source_idx[1] = 8
                          │   └───── GT (mid_idx = 7)
                          └───────── ref_source_idx[0] = 6
```

### Podział danych

```python
mid_idx = 7                    # Środkowy indeks
ref_source_idx = [6, 8]        # Indeksy klatek referencyjnych (sąsiedzi GT)
target_timestep = 0.5          # Timestep interpolacji (połowa między ref frames)
```

### Co wchodzi do modelu?

#### 1. **frames** - Input frames (14 klatek)
```
Tensor shape: (14, 3, 224, 224)  → (T-1, C, H, W)

Indeksy oryginalnych klatek: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
                                                    ↑
                                        BRAKUJE klatki 7 (GT)!

┌─────────────────────────────────────────────────────────────┐
│  INPUT FRAMES (do modelu)                                   │
├───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬────┬────┬────┬────┤
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 8 │ 9 │10 │ 11 │ 12 │ 13 │ 14 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴────┴────┴────┴────┘
                          ▲       ▲
                          └───────┘
                      Reference frames
```

**Użycie:** Model dostaje te klatki jako kontekst do przewidzenia brakującej klatki 7.

#### 2. **ref_frames** - Reference frames (2 klatki)
```
Tensor shape: (2, 3, 224, 224)  → (2, C, H, W)

Indeksy: [6, 8]  (sąsiedzi GT)

┌─────────────────────────────────┐
│  REFERENCE FRAMES                │
├───────────────┬─────────────────┤
│   Frame 6     │    Frame 8      │
│  (ref[0])     │   (ref[1])      │
└───────────────┴─────────────────┘
```

**Użycie:**
- Bazowe klatki do obliczenia **optical flow**
- Model uczy się ruchu między nimi: `flow_6→7` i `flow_8→7`
- Używane do warpowania (przesunięcia pikseli zgodnie z ruchem)

#### 3. **gt** - Ground Truth (1 klatka)
```
Tensor shape: (3, 224, 224)  → (C, H, W)

Indeks: 7

┌─────────────────────────┐
│   GROUND TRUTH          │
│                         │
│     Frame 7             │
│   (prawdziwa            │
│    środkowa klatka)     │
│                         │
└─────────────────────────┘
```

**Użycie:**
- **NIE wchodzi** do modelu jako input!
- Używane tylko do obliczenia **loss** (porównanie z predykcją)
- Cel do osiągnięcia przez model

#### 4. **timestep** - Timestep interpolacji (skalar)
```
Tensor shape: ()  → skalar
Wartość: 0.5

Znaczenie:
  0.0 = Frame 6 (ref[0])
  0.5 = Frame 7 (GT) - środek między ref frames
  1.0 = Frame 8 (ref[1])
```

**Użycie:** Informuje model, w którym momencie między ref frames ma interpolować.

---

## 🔄 Przepływ danych przez model

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DO MODELU                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─→ frames (14 klatek)
                            ├─→ ref_frames (2 klatki)
                            └─→ timestep (0.5)
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LIFT MODEL                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Encoder: Ekstraktuje features z frames           │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. IFNet: Oblicza optical flow                      │   │
│  │     - flow_6→7 (z ref[0] do GT)                     │   │
│  │     - flow_8→7 (z ref[1] do GT)                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Warper: Przesuwa piksele zgodnie z flow         │   │
│  │     - warped_6 = warp(ref[0], flow_6→7)             │   │
│  │     - warped_8 = warp(ref[1], flow_8→7)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. Occlusion: Wykrywa okluzje (zasłonięcia)        │   │
│  │     - occ_6→7 (co jest zasłonięte)                  │   │
│  │     - occ_8→7                                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  5. Synthesis: Łączy warped frames + context        │   │
│  │     prediction = f(warped_6, warped_8, context)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  6. Refinement: Dopracowuje szczegóły               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
                    ┌───────────────┐
                    │  PREDICTION   │  (przewidziana Frame 7)
                    └───────────────┘
                            │
                            ↓
                    ┌───────────────────────────┐
                    │   LOSS COMPUTATION        │
                    │   loss = L(pred, gt)      │
                    └───────────────────────────┘
```

---

## 📦 Format batcha w DataLoader

Gdy używasz `DataLoader` z `batch_size=4`:

```python
batch = {
    'frames':     torch.Size([4, 14, 3, 224, 224]),  # (B, T-1, C, H, W)
    'ref_frames': torch.Size([4, 2, 3, 224, 224]),   # (B, 2, C, H, W)
    'gt':         torch.Size([4, 3, 224, 224]),      # (B, C, H, W)
    'timestep':   torch.Size([4])                    # (B,)
}
```

Gdzie:
- `B` = batch size (4)
- `T` = num_frames (15), więc input ma T-1 = 14
- `C` = channels (3 dla RGB)
- `H, W` = height, width (224x224)

---

## 🔍 Jak to zweryfikować?

### Użyj narzędzi inspekcji:

```bash
# 1. Wypisz statystyki
python inspect_data.py --dataset ucf101 --num_sequences 3

# 2. Zobacz wizualizacje
ls data_inspection/  # Sprawdź wygenerowane obrazki

# 3. Sprawdź batch
python inspect_data.py --dataset ucf101 --batch_inspection
```

### W kodzie Python:

```python
from dataset import UCF101Dataset
from utils.data_inspector import print_dataset_stats, inspect_batch

dataset = UCF101Dataset(mode='train', num_frames=15)

# Sprawdź jedną sekwencję
sample = dataset[0]
print(f"frames shape: {sample['frames'].shape}")        # (14, 3, 224, 224)
print(f"ref_frames shape: {sample['ref_frames'].shape}") # (2, 3, 224, 224)
print(f"gt shape: {sample['gt'].shape}")                # (3, 224, 224)
print(f"timestep: {sample['timestep'].item()}")         # 0.5

# Sprawdź dataset
print_dataset_stats(dataset)
```

---

## ⚠️ Częste pułapki

### ❌ BŁĄD: GT w input frames
```python
# ZŁE - GT (klatka 7) jest w input!
input_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # 15 klatek
# Model by się ściągał - widzi odpowiedź!
```

### ✅ POPRAWNE: GT pominięte
```python
# DOBRE - GT (klatka 7) jest pominięta
input_frames = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]  # 14 klatek
# Model musi się nauczyć interpolować
```

### ❌ BŁĄD: Złe indeksy reference frames
```python
# ZŁE - ref frames nie są sąsiadami GT
ref_source_idx = [0, 14]  # Zbyt daleko od GT (7)
```

### ✅ POPRAWNE: Sąsiedzi GT
```python
# DOBRE - ref frames bezpośrednio sąsiadują z GT
ref_source_idx = [6, 8]  # Klatki przed i po GT (7)
```

---

## 📚 Kod w base_video.py

Kluczowe fragmenty:

```python
# Linia 108-113: Definicja indeksów
self.mid_idx = num_frames // 2  # 15 // 2 = 7
if self.is_odd:
    self.ref_source_idx = [self.mid_idx - 1, self.mid_idx + 1]  # [6, 8]
else:
    self.ref_source_idx = [self.mid_idx - 1, self.mid_idx]

# Linia 313-318: Wyznaczenie GT
if self.is_odd:
    gt_frame = frames[self.mid_idx].copy()  # Klatka 7
else:
    r1 = frames[self.ref_source_idx[0]].astype(np.float32)
    r2 = frames[self.ref_source_idx[1]].astype(np.float32)
    gt_frame = ((r1 + r2) / 2.0).astype(np.uint8)  # Średnia

# Linia 321-324: Input frames (bez GT)
if self.is_odd:
    input_frames_list = frames[:self.mid_idx] + frames[self.mid_idx+1:]
    # [0:7] + [8:15] = [0,1,2,3,4,5,6] + [8,9,10,11,12,13,14]
else:
    input_frames_list = frames  # Wszystkie klatki

# Linia 327-328: Reference frames
ref_frame_1 = frames[self.ref_source_idx[0]].copy()  # Frame 6
ref_frame_2 = frames[self.ref_source_idx[1]].copy()  # Frame 8
```

---

## 🎓 Podsumowanie

| Komponent | Shape | Indeksy oryginalnych klatek | Rola |
|-----------|-------|-----------------------------|------|
| **frames** | `(14, 3, 224, 224)` | `[0,1,2,3,4,5,6,8,9,10,11,12,13,14]` | Input do modelu |
| **ref_frames** | `(2, 3, 224, 224)` | `[6, 8]` | Bazowe klatki dla optical flow |
| **gt** | `(3, 224, 224)` | `[7]` | Target (tylko do loss) |
| **timestep** | `()` | `-` | Moment interpolacji (0.5) |

**Kluczowe zasady:**
1. ✅ Input frames **NIE zawierają** GT
2. ✅ Reference frames **sąsiadują** z GT
3. ✅ GT używane **tylko do loss**, nie wchodzi do modelu
4. ✅ Model **przewiduje** klatkę 7 na podstawie pozostałych 14 klatek

---

**Autor:** Claude
**Data:** 2025-12-11
**Plik źródłowy:** `dataset/base_video.py`
