# LIFT Training Module

Moduł treningowy dla modelu **LIFT** (Long-range Interpolation with Far Temporal context) - sieci neuronowej do interpolacji klatek wideo wykorzystującej rozszerzony kontekst czasowy.

## Przegląd

Skrypt `train.py` implementuje kompletny pipeline treningowy obejmujący ładowanie danych, konfigurację eksperymentów, trening z mixed precision oraz szczegółowe logowanie metryk i wizualizacji.

## Architektura danych

### Klasa bazowa

Datasety UCF-101 i X4K1000FPS dziedziczą po `BaseVideoDataset` (base_video), która zapewnia:

- ekstrakcję sekwencji klatek z zachowaniem temporalnej spójności,
- pipeline augmentacji (losowe przycinanie, odbicia, kolorystyka),
- konwersję klatek do tensorów PyTorch,
- generowanie zestawów: klatki wejściowe -> referencje -> ground truth.

### Obsługiwane datasety

| Dataset | Klasa | Opis |
|---------|-------|------|
| Vimeo-15 | `Vimeo15Dataset` (vimeo90k) | Sekwencje 7-klatkowe |
| UCF-101 | `UCF101Dataset` (ucf101) | Akcje ludzkie |
| X4K1000FPS | `X4K1000FPSDataset` (x4k1000fps) | Wysokorozdzielcze sekwencje 4K |

Wybór datasetu odbywa się dynamicznie poprzez fabrykę `get_dataset_class()`.

## Konfiguracja

Parametry eksperymentu definiuje klasa `Config` (default). Skrypt wczytuje domyślne wartości i nadpisuje je argumentami z linii poleceń.

### Główne parametry

| Kategoria | Parametry |
|-----------|-----------|
| Trening | batch size, liczba epok, gradient accumulation |
| Dane | liczba klatek wejściowych, rozmiar crop, augmentacje |
| Optymalizator | learning rate, weight decay, scheduler |
| Ścieżki | katalog logów, checkpointy, dane |

## Pipeline treningowy

Skrypt `train.py` (train) realizuje:

1. **Inicjalizację** - budowa modelu LIFT, funkcji straty i optymalizatora
2. **Ładowanie danych** - tworzenie DataLoaderów z prefetchingiem
3. **Pętlę treningową** - mixed precision (AMP), gradient clipping, akumulacja gradientów
4. **Walidację** - obliczanie PSNR i SSIM na zbiorze walidacyjnym
5. **Checkpointing** - zapis najlepszego modelu i okresowe snapshoty

## Logowanie i wizualizacja

Klasa `TensorBoardLogger` (tensorboard_logger) zapisuje:

- **Metryki** - loss treningowy/walidacyjny, PSNR, SSIM
- **Parametry** - learning rate, czas batchy, zużycie pamięci
- **Wizualizacje** - predykcje vs ground truth, klatki referencyjne
- **Przepływy optyczne** - mapy flow i błędu - `visualization.py`

### Uruchomienie TensorBoard

```bash
tensorboard --logdir logs --bind_all
```

Interfejs dostępny pod adresem `http://localhost:6006`.

## Szybki start

```bash
python train.py \
    --dataset vimeo \
    --batch_size 8 \
    --epochs 100 \
    --max_sequences 10000
```

## Struktura katalogów

```
├── train.py              # Główny skrypt treningowy
├── configs/default.py    # Definicja klasy Config
├── datasets/
│   ├── base_video.py     # Klasa bazowa datasetów
│   ├── vimeo_64.py       # Vimeo-15 dataset
│   ├── ucf101.py         # UCF-101 dataset
│   └── x4k1000fps.py     # X4K1000FPS dataset
├── utils/
│   ├── tensorboard_logger.py
│   └── visualization.py
└── logs/                 # Logi TensorBoard
```
