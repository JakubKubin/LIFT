# LIFT-15: Long-range Interpolation with Far Temporal context
## Specyfikacja Architektury (15 klatek)

**Wersja:** 2.1  
**Data:** 2025-01  
**Autor:** [Twoje dane]  
**Cel:** Master Thesis - Video Frame Interpolation z rozszerzonym kontekstem czasowym

---

## 1. Przegląd ogólny

### 1.1 Hipoteza badawcza

Wykorzystanie szerokiego kontekstu czasowego (15 klatek vs standardowe 2-4) pozwala na lepszą interpolację klatek wideo, szczególnie w scenach z:
- Okluzjami (obiekty przysłaniające się nawzajem)
- Szybkim ruchem
- Powtarzającymi się wzorcami (koła, nogi w ruchu)
- Zmianami oświetlenia

### 1.2 Konfiguracja wejścia/wyjścia

```
TRENING:
┌─────────────────────────────────────────────────────────────────┐
│  I₀   I₁   I₂   I₃   I₄   I₅   I₆   I₇   [I₈]   I₉   ...  I₁₄  │
│  ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓     GT    ↓         ↓   │
│  Kontekst lewy (7 klatek)      │      Kontekst prawy (6 klatek)│
│                                ▼                               │
│                         Î₈ (predykcja)                         │
└─────────────────────────────────────────────────────────────────┘

INFERENCE:
┌─────────────────────────────────────────────────────────────────┐
│  I₀   I₁   I₂   I₃   I₄   I₅   I₆   I₇    ?    I₉   ...  I₁₄  │
│                                ▼                               │
│                         Î₈ (generowana)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Kluczowe parametry

| Parametr | Wartość | Uwagi |
|----------|---------|-------|
| Liczba klatek wejściowych | 15 | I₀ do I₁₄ |
| Klatki przetwarzane (trening) | 14 | Bez I₈ (GT) |
| Klatki referencyjne | I₇, I₉ | Najbliższe sąsiedztwo |
| Generowana klatka | I₈ | t = 0.5 między I₇ a I₉ |
| Rozdzielczość bazowa | 256×256 | Trening |
| Rozdzielczość docelowa | 256×448 | Vimeo90K |

---

## 2. STAGE 1: Ekstrakcja cech (Feature Extraction)

### 2.1 Cel
Wydobycie wieloskalowych map cech z każdej z 14 klatek wejściowych przy użyciu współdzielonego encodera konwolucyjnego. Dla klatek referencyjnych I₇ i I₉ dodatkowo ekstrahujemy cechy w pełnej rozdzielczości (s1) dla zachowania maksymalnej ilości detali.

### 2.2 Architektura encodera

```
Input: Iₖ ∈ ℝ^(B×3×H×W)

Encoder (współdzielony, bazowany na RIFE):
├── Conv2d(3 → 32, k=3, s=1, p=1) + LeakyReLU
├── Conv2d(32 → 32, k=3, s=1, p=1) + LeakyReLU
├── ─────────────────────────────────────────── → Fₖˢ¹ ∈ ℝ^(B×32×H×W)        [TYLKO I₇, I₉]
├── Conv2d(32 → 64, k=3, s=2, p=1) + LeakyReLU
├── Conv2d(64 → 64, k=3, s=1, p=1) + LeakyReLU
├── ─────────────────────────────────────────── → Fₖˢ⁴ ∈ ℝ^(B×128×H/4×W/4)   [TYLKO I₇, I₉]
├── Conv2d(64 → 128, k=3, s=2, p=1) + LeakyReLU
├── Conv2d(128 → 128, k=3, s=1, p=1) + LeakyReLU
├── ─────────────────────────────────────────── → Fₖˢ⁸ ∈ ℝ^(B×192×H/8×W/8)   [TYLKO I₇, I₉]
├── Conv2d(128 → 192, k=3, s=2, p=1) + LeakyReLU
├── Conv2d(192 → 192, k=3, s=1, p=1) + LeakyReLU
└── Conv2d(192 → 256, k=3, s=2, p=1) + LeakyReLU → Fₖˢ¹⁶ ∈ ℝ^(B×256×H/16×W/16) [WSZYSTKIE]
```

**Uwaga:** Wczesne warstwy (przed s4) mają mniej kanałów (32→64) niż w oryginalnej wersji, aby zoptymalizować pamięć przy zachowaniu cech s1.

### 2.3 Wymiary tensorów

| Skala | Wymiary | Kanały | Użycie | Klatki |
|-------|---------|--------|--------|--------|
| **s1 (1/1)** | **H × W** | **32** | **STAGE 5 - full-res refinement** | **tylko I₇, I₉** |
| s4 (1/4) | H/4 × W/4 | 128 | STAGE 3, STAGE 5 | tylko I₇, I₉ |
| s8 (1/8) | H/8 × W/8 | 192 | STAGE 3 | tylko I₇, I₉ |
| s16 (1/16) | H/16 × W/16 | 256 | STAGE 2 (transformer) | wszystkie 14 klatek |

### 2.4 Kodowanie pozycyjne (Positional Encoding)

```python
def sinusoidal_pe(k, C, max_len=15):
    """
    k: indeks klatki (0-14, z pominięciem 8 podczas treningu)
    C: liczba kanałów (32/128/192/256)
    
    Zachowujemy ORYGINALNE indeksy - model "wie" o brakującej klatce 8
    """
    pe = zeros(C)
    for i in range(0, C, 2):
        pe[i] = sin(k / (10000 ** (i / C)))
        pe[i+1] = cos(k / (10000 ** (i / C)))
    return pe  # Broadcastowane do wymiarów przestrzennych
```

**Decyzja projektowa:** Używamy oryginalnych indeksów (0,1,...,7,9,...,14) zamiast renumeracji, aby model miał jawną informację o pozycji interpolowanego momentu.

### 2.5 Wyjścia STAGE 1

```
Dla transformera (STAGE 2):
    F_temporal = [F₀ˢ¹⁶, F₁ˢ¹⁶, ..., F₇ˢ¹⁶, F₉ˢ¹⁶, ..., F₁₄ˢ¹⁶]
    Wymiar: ℝ^(B×14×256×H/16×W/16)

Dla przepływu (STAGE 3):
    F₇ˢ⁴, F₉ˢ⁴ ∈ ℝ^(B×128×H/4×W/4)
    F₇ˢ⁸, F₉ˢ⁸ ∈ ℝ^(B×192×H/8×W/8)

Dla refinera pełnej rozdzielczości (STAGE 5):
    F₇ˢ¹, F₉ˢ¹ ∈ ℝ^(B×32×H×W)           ← NOWE! Pełna rozdzielczość
    F₇ˢ⁴, F₉ˢ⁴ ∈ ℝ^(B×128×H/4×W/4)
```

### 2.6 Strategia zamrażania wag

```
Epoki 1-10:  Encoder ZAMROŻONY (wykorzystanie pretrenowanych wag RIFE)
Epoki 11+:  Stopniowe odmrażanie z niskim LR (lr_encoder = 0.1 × lr_base)
```

**Uwaga:** Warstwy s1 (3→32→32) są NOWE i nie mają pretrenowanych wag - należy je trenować od początku lub zainicjalizować z wag RIFE po dostosowaniu wymiarów.

### 2.7 Optymalizacja pamięci

Cechy s1 są przechowywane TYLKO dla I₇ i I₉:
```python
# Pseudokod forward pass
for k, frame in enumerate(input_frames):
    f_s16 = encoder_full(frame)  # Zawsze do s16
    features_s16.append(f_s16)
    
    if k in [7, 9]:  # Tylko klatki referencyjne
        f_s1 = encoder.get_s1_features(frame)
        f_s4 = encoder.get_s4_features(frame)
        f_s8 = encoder.get_s8_features(frame)
        ref_features[k] = {'s1': f_s1, 's4': f_s4, 's8': f_s8}
```

### 2.8 TODO implementacyjne

- [ ] Zaimplementować `FeatureEncoder` z czterema wyjściami skalowymi (s1, s4, s8, s16)
- [ ] Dodać `SinusoidalPositionalEncoding` z obsługą nieciągłych indeksów
- [ ] Załadować pretrenowane wagi z RIFE (IFNet encoder) - dostosować do nowych warstw s1
- [ ] Implementacja mechanizmu zamrażania/odmrażania
- [ ] Selektywne przechowywanie cech (s1 tylko dla I₇, I₉)
- [ ] Test pamięci: 14×256×H/16×W/16 + 2×32×H×W

---

## 3. STAGE 2: Agregacja czasowa (Temporal Aggregation Transformer)

### 3.1 Cel
Modelowanie zależności czasowych między 14 klatkami i agregacja do pojedynczej mapy kontekstowej F_ctx.

### 3.2 Kluczowa zmiana vs 64-klatkowa wersja

```
64 klatki: Okienkowa uwaga (W=8) → O(T·W²) = O(64·64) = 4096 operacji
15 klatek: PEŁNA UWAGA możliwa → O(T²) = O(14²) = 196 operacji

Redukcja: ~20× mniej operacji! Można użyć pełnej uwagi bez okien.
```

### 3.3 Parametry transformera

| Parametr | Wartość | Uzasadnienie |
|----------|---------|--------------|
| Liczba warstw L | 3 | Krótsza sekwencja = mniej warstw |
| Wymiar modelu D | 256 | Zgodny z kanałami s16 |
| Liczba głów h | 8 | Standard dla D=256 |
| Uwaga czasowa | **PEŁNA** | T=14 pozwala na pełną uwagę |
| Rozmiar patcha P | 2×2 | Tokenizacja przestrzenna |
| FFN expansion | 4× | D → 4D → D |
| Dropout | 0.1 | Regularyzacja |

### 3.4 Tokenizacja przestrzenna

```
Input:  F_temporal ∈ ℝ^(B×14×256×H/16×W/16)

Dla H=W=256:
    Spatial size @ s16: 16×16
    Patch size: 2×2
    Tokens per frame: (16/2)×(16/2) = 64
    
Output po patchify:
    tokens ∈ ℝ^(B×14×64×256)  czyli (B, T, L, D)
    gdzie T=14 (klatki), L=64 (patche przestrzenne), D=256 (wymiar)
```

### 3.5 Architektura warstwy transformera

```
Dla każdej z L=3 warstw:

┌─────────────────────────────────────────────────────────────────┐
│ 1. TEMPORAL SELF-ATTENTION (pełna, nie okienkowa!)             │
│    ├── Input: (B, T, L, D) → reshape → (B×L, T, D)             │
│    ├── MultiHeadAttention(D, heads=8)                          │
│    ├── Każdy patch "widzi" wszystkie 14 klatek                 │
│    └── Output: (B×L, T, D) → reshape → (B, T, L, D)            │
├─────────────────────────────────────────────────────────────────┤
│ 2. SPATIAL PROCESSING (DepthwiseSeparable Conv)                │
│    ├── Reshape: (B, T, L, D) → (B×T, D, 8, 8)                  │
│    ├── DepthwiseConv2d(D, D, k=3, groups=D)                    │
│    ├── PointwiseConv2d(D, D, k=1)                              │
│    ├── GroupNorm(8 groups) + residual                          │
│    └── Reshape back: (B×T, D, 8, 8) → (B, T, L, D)             │
├─────────────────────────────────────────────────────────────────┤
│ 3. FEED-FORWARD NETWORK                                        │
│    ├── LayerNorm                                               │
│    ├── Linear(D → 4D) + GELU                                   │
│    ├── Linear(4D → D)                                          │
│    ├── Dropout(0.1)                                            │
│    └── Residual connection                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 Adaptacyjna agregacja czasowa

```
Po 3 warstwach transformera:
    F_agg ∈ ℝ^(B×14×64×256)

Agregacja do pojedynczego kontekstu:

1. Global Average Pooling per klatka:
   gₖ = mean(F_agg[k], dim=spatial) ∈ ℝ^D    dla k ∈ {0..7, 9..14}

2. Importance scoring (MLP):
   αₖ_raw = MLP([g₀, g₁, ..., g₇, g₉, ..., g₁₄])
   MLP: D → D/4 → 1 (per klatka)

3. Softmax normalization:
   αₖ = softmax(α_raw)    gdzie Σαₖ = 1

4. Weighted aggregation:
   F_ctx = Σₖ αₖ · F_agg[k] ∈ ℝ^(B×256×H/16×W/16)
```

**Output pomocniczy:** Wektor wag αₖ do wizualizacji (które klatki model uznał za najważniejsze).

### 3.7 Wyjście STAGE 2

```
F_ctx ∈ ℝ^(B×256×H/16×W/16)

Oczekiwane zachowanie wag α:
- Wyższe dla klatek bliskich t=8 (czyli I₇, I₉)
- Wyższe dla klatek z istotnymi zdarzeniami ruchu
- Niższe dla klatek statycznych/redundantnych
```

### 3.8 TODO implementacyjne

- [ ] Zaimplementować `TemporalTransformer` z pełną uwagą (nie okienkową!)
- [ ] `PatchEmbedding` - konwersja map cech na tokeny
- [ ] `DepthwiseSeparableConv2d` dla przetwarzania przestrzennego
- [ ] `AdaptiveTemporalAggregation` z MLP do ważenia
- [ ] Zachować αₖ jako output do TensorBoard/wizualizacji
- [ ] Test: porównanie pamięci pełna vs okienkowa uwaga

---

## 4. STAGE 3: Wieloskalowe szacowanie przepływu (Flow Estimation)

### 4.1 Cel
Oszacowanie przepływów optycznych i map okluzji dla klatki I₈ względem I₇ i I₉.

### 4.2 Architektura kaskady 2-skalowej

```
                    ┌─────────────┐
                    │   SKALA s8  │
                    │  (gruba)    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   flow₇ˢ⁸           flow₉ˢ⁸          logit_O₇ˢ⁸, logit_O₉ˢ⁸
   ∈ ℝ^(B×2×H/8×W/8)                  ∈ ℝ^(B×1×H/8×W/8)
        │                  │                  │
        │    ×2 upsample + scale              │    bilinear upsample
        ▼                  ▼                  ▼
                    ┌─────────────┐
                    │   SKALA s4  │
                    │ (refinement)│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   flow₇ˢ⁴           flow₉ˢ⁴           O₇ˢ⁴, O₉ˢ⁴
   ∈ ℝ^(B×2×H/4×W/4)                  ∈ [0,1]^(B×1×H/4×W/4)
```

### 4.3 Skala s8 - szacowanie grube

**Wejście:**
```
F₇ˢ⁸ ∈ ℝ^(B×192×H/8×W/8)      - cechy klatki I₇
F₉ˢ⁸ ∈ ℝ^(B×192×H/8×W/8)      - cechy klatki I₉
F_ctxˢ⁸ = AvgPool2d(F_ctx)     - kontekst ∈ ℝ^(B×256×H/8×W/8)
t_chan = 0.5 · 𝟙              - czas ∈ ℝ^(B×1×H/8×W/8)

input_s8 = concat([F₇ˢ⁸, F₉ˢ⁸, F_ctxˢ⁸, t_chan])
         ∈ ℝ^(B×641×H/8×W/8)
```

**Sieć IFNet-like:**
```
Conv(641 → 256, k=3) + LeakyReLU
ResBlock(256) × 3
Conv(256 → 6, k=3)  # 2+2+1+1 = 6 kanałów wyjściowych

Output:
├── flow₇ˢ⁸ ∈ ℝ^(B×2×H/8×W/8)      - przepływ do I₇
├── flow₉ˢ⁸ ∈ ℝ^(B×2×H/8×W/8)      - przepływ do I₉
├── logit_O₇ˢ⁸ ∈ ℝ^(B×1×H/8×W/8)   - logit okluzji I₇
└── logit_O₉ˢ⁸ ∈ ℝ^(B×1×H/8×W/8)   - logit okluzji I₉
```

### 4.4 Skala s4 - refinement

**Upsampling z s8:**
```python
flow₇_up = 2 × bilinear_upsample(flow₇ˢ⁸, scale=2)  # ×2 bo większa rozdzielczość
flow₉_up = 2 × bilinear_upsample(flow₉ˢ⁸, scale=2)
logit_O₇_up = bilinear_upsample(logit_O₇ˢ⁸, scale=2)  # bez skalowania wartości
logit_O₉_up = bilinear_upsample(logit_O₉ˢ⁸, scale=2)
```

**Wejście do refinera:**
```
F₇ˢ⁴ ∈ ℝ^(B×128×H/4×W/4)
F₉ˢ⁴ ∈ ℝ^(B×128×H/4×W/4)
F_ctxˢ⁴ = bilinear_upsample(F_ctx, scale=4) ∈ ℝ^(B×256×H/4×W/4)
t_chan_s4 = 0.5 · 𝟙 ∈ ℝ^(B×1×H/4×W/4)

refine_input = concat([F₇ˢ⁴, F₉ˢ⁴, F_ctxˢ⁴, 
                       flow₇_up, flow₉_up,
                       logit_O₇_up, logit_O₉_up,
                       t_chan_s4])
             ∈ ℝ^(B×519×H/4×W/4)
```

**Sieć refinująca:**
```
Conv(519 → 128, k=3) + LeakyReLU
ResBlock(128) × 2
Conv(128 → 6, k=3)  # delta dla flow i logit

Output (residualne!):
├── Δflow₇ ∈ ℝ^(B×2×H/4×W/4)
├── Δflow₉ ∈ ℝ^(B×2×H/4×W/4)
├── Δlogit_O₇ ∈ ℝ^(B×1×H/4×W/4)
└── Δlogit_O₉ ∈ ℝ^(B×1×H/4×W/4)

Finalne wartości:
flow₇ˢ⁴ = flow₇_up + Δflow₇
flow₉ˢ⁴ = flow₉_up + Δflow₉
logit_O₇ˢ⁴ = logit_O₇_up + Δlogit_O₇
logit_O₉ˢ⁴ = logit_O₉_up + Δlogit_O₉

Sigmoid NA KOŃCU:
O₇ˢ⁴ = σ(logit_O₇ˢ⁴) ∈ [0,1]^(B×1×H/4×W/4)
O₉ˢ⁴ = σ(logit_O₉ˢ⁴) ∈ [0,1]^(B×1×H/4×W/4)
```

### 4.5 Wyjście STAGE 3

```
flow₇ˢ⁴ ∈ ℝ^(B×2×H/4×W/4)      - przepływ optyczny I₈→I₇
flow₉ˢ⁴ ∈ ℝ^(B×2×H/4×W/4)      - przepływ optyczny I₈→I₉
O₇ˢ⁴ ∈ [0,1]^(B×1×H/4×W/4)     - mapa okluzji dla I₇
O₉ˢ⁴ ∈ [0,1]^(B×1×H/4×W/4)     - mapa okluzji dla I₉
```

### 4.6 TODO implementacyjne

- [ ] Zaimplementować `FlowEstimatorS8` (IFNet-like)
- [ ] Zaimplementować `FlowRefinerS4` (ResBlocks)
- [ ] `ResBlock` z GroupNorm
- [ ] Upsampling z prawidłowym skalowaniem flow (×2)
- [ ] Sigmoid TYLKO na końcu (logity w środku!)
- [ ] Wizualizacja flow jako color wheel

---

## 5. STAGE 4: Synteza klatki zgrubnej (Coarse Synthesis)

### 5.1 Cel
Wygenerowanie zgrubnej klatki pośredniej I₈^coarse w rozdzielczości s4.

### 5.2 Backward Warping

```python
# Downscale klatek referencyjnych
I₇ˢ⁴ = bilinear_downsample(I₇, scale=0.25)  # ∈ ℝ^(B×3×H/4×W/4)
I₉ˢ⁴ = bilinear_downsample(I₉, scale=0.25)

# Backward warp - przenosimy piksele z I₇/I₉ do pozycji I₈
I₈_from_7 = backward_warp(I₇ˢ⁴, flow₇ˢ⁴)  # grid_sample z flow
I₈_from_9 = backward_warp(I₉ˢ⁴, flow₉ˢ⁴)
```

### 5.3 Occlusion-aware Blending

```python
# Ważone łączenie z użyciem map okluzji
ε = 1e-8
I₈_blend = (O₇ˢ⁴ * I₈_from_7 + O₉ˢ⁴ * I₈_from_9) / (O₇ˢ⁴ + O₉ˢ⁴ + ε)

# I₈_blend ∈ ℝ^(B×3×H/4×W/4)
```

**Interpretacja:**
- Wysoka O₇ → region dobrze widoczny w I₇ → więcej wagi z I₇
- Wysoka O₉ → region dobrze widoczny w I₉ → więcej wagi z I₉
- Obie niskie → okluzja w obu → średnia (lub potrzebny inpainting)

### 5.4 Context Injection (ContextNet)

```
Input:
├── I₈_blend ∈ ℝ^(B×3×H/4×W/4)
└── F_ctxˢ⁴ ∈ ℝ^(B×256×H/4×W/4)

ctx_input = concat([I₈_blend, F_ctxˢ⁴]) ∈ ℝ^(B×259×H/4×W/4)

ContextNet (bardzo lekka!):
├── Conv(259 → 64, k=3, p=1) + ReLU
└── Conv(64 → 3, k=3, p=1)           # residual w przestrzeni obrazu

Output:
residual ∈ ℝ^(B×3×H/4×W/4)
I₈_coarse = I₈_blend + residual
```

### 5.5 Wyjście STAGE 4

```
I₈_coarse ∈ ℝ^(B×3×H/4×W/4)

Klatka zgrubna ale czasowo spójna - główna praca semantyczna
wykonana przez transformer, tu tylko dopasowanie szczegółów.
```

### 5.6 TODO implementacyjne

- [ ] Zaimplementować `backward_warp` używając `F.grid_sample`
- [ ] `OcclusionBlender` z obsługą epsilon
- [ ] `ContextNet` (2 warstwy conv)
- [ ] Test: czy blend wygląda sensownie przed ContextNet?

---

## 6. STAGE 5: Refinement pełnej rozdzielczości (Full-res Refinement)

### 6.1 Cel
Dopracowanie detali w pełnej rozdzielczości H×W z wykorzystaniem cech s1 z klatek referencyjnych.

### 6.2 Upsample klatki zgrubnej

```python
I₈_up = bilinear_upsample(I₈_coarse, scale=4)  # ∈ ℝ^(B×3×H×W)
```

### 6.3 Wykorzystanie cech pełnej rozdzielczości (s1)

**KLUCZOWA ZMIANA:** Zamiast upsamplować cechy s4, używamy bezpośrednio cech s1 - zero utraty informacji!

```python
# Cechy s1 już są w pełnej rozdzielczości - nie trzeba upsamplować!
F₇ˢ¹ ∈ ℝ^(B×32×H×W)  # bezpośrednio z STAGE 1
F₉ˢ¹ ∈ ℝ^(B×32×H×W)  # bezpośrednio z STAGE 1

# Nie ma redukcji kanałów ani upsamplingu - pełne detale zachowane!
```

**Porównanie z poprzednią wersją:**
```
STARA WERSJA (bez s1):
F₇ˢ⁴ ∈ ℝ^(B×128×H/4×W/4) → Conv1x1 → ℝ^(B×32×H/4×W/4) → upsample ×4 → ℝ^(B×32×H×W)
                                      ↑ utrata informacji przez upsample!

NOWA WERSJA (z s1):
F₇ˢ¹ ∈ ℝ^(B×32×H×W) → bezpośrednio do refinera
                      ↑ pełne detale, zero utraty!
```

### 6.4 Lightweight Refinement Network

```
Input:
refine_input = concat([I₈_up, F₇ˢ¹, F₉ˢ¹]) ∈ ℝ^(B×67×H×W)
                       ↑      ↑     ↑
                       3     32    32  = 67 kanałów

RefineNet:
├── Conv(67 → 64, k=3, p=1) + GroupNorm(8) + ReLU
├── ResBlock(64 → 64) × 2
│   └── Conv(64→64, k=3) + GN + ReLU + Conv(64→64, k=3) + GN + residual
├── Conv(64 → 32, k=3, p=1) + GroupNorm(4) + ReLU
└── Conv(32 → 3, k=3, p=1)  # bez aktywacji - residual

Output:
residual ∈ ℝ^(B×3×H×W)
I₈_final = I₈_up + residual
```

### 6.5 Wyjście końcowe

```
I₈_final ∈ ℝ^(B×3×H×W)  - finalna interpolowana klatka

Clamp do [0, 1] przed zapisem/wizualizacją!
```

### 6.6 Zalety użycia s1

| Aspekt | Bez s1 (stara wersja) | Z s1 (nowa wersja) |
|--------|----------------------|-------------------|
| Rozdzielczość cech | H/4 × W/4 → upsample | H × W (natywna) |
| Utrata detali | Tak (przez upsample) | Nie |
| Kanały | 128 → 32 (redukcja) | 32 (od razu) |
| Pamięć | Mniej | +~16MB (akceptowalne) |
| Jakość krawędzi | Rozmyte | **Ostre** |

### 6.7 TODO implementacyjne

- [ ] Zmodyfikować `FullResRefiner` na przyjmowanie F₇ˢ¹, F₉ˢ¹ zamiast upsamplowanych s4
- [ ] Usunąć `ChannelReducer` i upsampling dla s4 (niepotrzebne!)
- [ ] ResBlocks z GroupNorm
- [ ] Sprawdzić czy final output jest w [0,1]
- [ ] Porównanie wizualne: I₈_coarse vs I₈_final vs GT
- [ ] **Test A/B:** wersja z s1 vs bez s1 - spodziewana poprawa na krawędziach

---

## 7. Funkcje strat (Loss Functions)

### 7.1 Główne straty

```python
# L1 Reconstruction Loss
L_rec = L1(I₈_final, I₈_GT)

# Perceptual Loss (LPIPS lub VGG)
L_perc = LPIPS(I₈_final, I₈_GT)  # lub VGG feature matching

# Total Loss
L_total = L_rec + λ_perc × L_perc

# Sugerowane wagi:
λ_perc = 0.1
```

### 7.2 Opcjonalne straty dodatkowe

```python
# Census Loss (dla robustności na zmiany oświetlenia)
L_census = census_transform_loss(I₈_final, I₈_GT)

# Flow Smoothness (regularyzacja przepływu)
L_smooth = smoothness_loss(flow₇ˢ⁴) + smoothness_loss(flow₉ˢ⁴)

# Warping Loss (czy warped frames są sensowne)
L_warp = L1(I₈_from_7, I₈_GT) + L1(I₈_from_9, I₈_GT)
```

### 7.3 TODO implementacyjne

- [ ] Implementacja L1 + LPIPS
- [ ] Rozważyć Census Loss dla trudnych scen
- [ ] Logowanie poszczególnych składników loss do TensorBoard

---

## 8. Pipeline treningu

### 8.1 Przygotowanie danych

```python
# Wczytanie 15-klatkowego klipu
frames = load_clip(video, start_idx, length=15)  # [I₀, ..., I₁₄]

# Separacja
input_frames = frames[[0,1,2,3,4,5,6,7,9,10,11,12,13,14]]  # 14 klatek
gt_frame = frames[8]  # Ground Truth

# Augmentacje
# - Random crop do 256×256
# - Random horizontal flip
# - Random temporal flip (odwrócenie kolejności)
# - Color jitter (ostrożnie!)
```

### 8.2 Forward pass

```python
def forward(input_frames, t=0.5):
    # STAGE 1 - ekstrakcja cech
    features_s16 = []
    ref_features = {}
    
    for k, frame in enumerate(input_frames):
        f_s16 = encoder.forward_s16(frame)
        features_s16.append(f_s16)
        
        if k in [7, 9]:  # Klatki referencyjne
            ref_features[k] = {
                's1': encoder.get_s1(frame),   # ← NOWE! Pełna rozdzielczość
                's4': encoder.get_s4(frame),
                's8': encoder.get_s8(frame),
            }
    
    # STAGE 2 - agregacja czasowa
    F_ctx, alphas = temporal_transformer(torch.stack(features_s16, dim=1))
    
    # STAGE 3 - szacowanie przepływu
    flows, occlusions = flow_estimator(
        ref_features[7]['s8'], ref_features[9]['s8'],
        ref_features[7]['s4'], ref_features[9]['s4'],
        F_ctx, t
    )
    
    # STAGE 4 - synteza zgrubna
    I_coarse = coarse_synthesis(
        input_frames[7], input_frames[9],  # I₇, I₉
        flows, occlusions, F_ctx
    )
    
    # STAGE 5 - refinement z cechami s1
    I_final = full_res_refiner(
        I_coarse,
        ref_features[7]['s1'],  # ← NOWE! Cechy pełnej rozdzielczości
        ref_features[9]['s1']   # ← NOWE!
    )
    
    return I_final, alphas, flows, occlusions
```

### 8.3 Harmonogram treningu

| Etap | Epoki | Co trenujemy | LR | Uwagi |
|------|-------|--------------|-----|-------|
| 1 | 1-10 | Wszystko POZA encoderem | 1e-4 | Encoder zamrożony |
| 2 | 11-30 | Wszystko | 1e-4 (encoder: 1e-5) | Stopniowe odmrażanie |
| 3 | 31-50 | Wszystko | 1e-5 | Fine-tuning |

**Uwaga:** Warstwy s1 (3→32→32) są nowe i mogą wymagać wyższego LR na początku.

### 8.4 TODO implementacyjne

- [ ] Dataloader dla 15-klatkowych klipów (Vimeo90K septuplet → rozszerzyć?)
- [ ] Augmentacje z temporal awareness
- [ ] Training loop z gradient accumulation (jeśli batch nie mieści się)
- [ ] Checkpointing co N epok
- [ ] TensorBoard: loss, αₖ, przykładowe interpolacje

---

## 9. Ewaluacja

### 9.1 Metryki

| Metryka | Co mierzy | Target |
|---------|-----------|--------|
| PSNR | Pixel-level similarity | >30 dB |
| SSIM | Structural similarity | >0.9 |
| LPIPS | Perceptual similarity | <0.1 |
| IE (Interpolation Error) | Flow accuracy | - |

### 9.2 Benchmarki

- **Vimeo90K-septuplet** (podstawowy)
- **UCF101** (action recognition clips)
- **DAVIS** (z okluzjami)
- **SNU-FILM** (różne poziomy trudności: Easy/Medium/Hard/Extreme)

### 9.3 Ablacje do przeprowadzenia

1. **Liczba klatek kontekstu:** 3 vs 7 vs 15 klatek
2. **Pełna vs okienkowa uwaga:** czy pełna jest lepsza przy T=14?
3. **Z kontekstem vs bez:** czy F_ctx pomaga?
4. **Liczba warstw transformera:** 2 vs 3 vs 4
5. **Z s1 vs bez s1:** czy cechy pełnej rozdzielczości poprawiają detale? ← NOWE!

---

## 10. Szacowanie zasobów

### 10.1 Pamięć GPU (szacunki dla batch=4, 256×256)

| Komponent | Pamięć | Uwagi |
|-----------|--------|-------|
| 14 klatek input | ~50 MB | |
| Features s16 (14×256×16×16) | ~28 MB | |
| **Features s1 (2×32×256×256)** | **~16 MB** | **NOWE** |
| Features s4/s8 (tylko I₇, I₉) | ~10 MB | |
| Transformer activations | ~200 MB | |
| Flow estimation | ~100 MB | |
| Refinement (full-res) | ~200 MB | Większe przez s1 |
| **Łącznie (forward)** | **~650 MB** | |
| **Z gradientami (training)** | **~3-4 GB** | |

### 10.2 Porównanie wersji

```
64 klatki (oryginał):     ~8-12 GB VRAM
15 klatek (bez s1):       ~2-3 GB VRAM
15 klatek (z s1):         ~3-4 GB VRAM    ← AKTUALNA WERSJA

Wzrost przez s1: ~0.5-1 GB (akceptowalny trade-off za lepsze detale!)
```

### 10.3 Optymalizacje pamięci (jeśli potrzebne)

```python
# Gradient checkpointing dla transformera
# (przeliczyć aktywacje podczas backward zamiast je trzymać)
from torch.utils.checkpoint import checkpoint

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
```

---

## 11. Struktura kodu (proponowana)

```
lift/
├── models/
│   ├── __init__.py
│   ├── encoder.py          # STAGE 1: FeatureEncoder (z wyjściem s1!)
│   ├── transformer.py      # STAGE 2: TemporalTransformer
│   ├── flow_estimator.py   # STAGE 3: FlowEstimator
│   ├── synthesis.py        # STAGE 4: CoarseSynthesis
│   ├── refiner.py          # STAGE 5: FullResRefiner (przyjmuje s1!)
│   └── lift.py             # Full LIFT model
├── data/
│   ├── __init__.py
│   ├── vimeo_dataset.py    # Vimeo90K loader (15 frames)
│   └── augmentations.py    # Temporal-aware augmentations
├── losses/
│   ├── __init__.py
│   ├── reconstruction.py   # L1, L2
│   ├── perceptual.py       # LPIPS, VGG
│   └── flow_losses.py      # Smoothness, warping
├── utils/
│   ├── __init__.py
│   ├── warp.py             # backward_warp, grid_sample
│   ├── positional.py       # Sinusoidal PE
│   └── visualization.py    # Flow vis, attention vis
├── train.py
├── evaluate.py
└── config.yaml
```

---

## 12. Checklisty

### 12.1 Przed rozpoczęciem implementacji

- [ ] Potwierdzić dostęp do Vimeo90K dataset
- [ ] Sprawdzić dostępność GPU (min. 8GB VRAM zalecane)
- [ ] Zainstalować zależności: torch, torchvision, lpips, tensorboard
- [ ] Pobrać pretrenowane wagi RIFE

### 12.2 Milestone 1: Encoder + Transformer

- [ ] STAGE 1 działa z wyjściami s1, s4, s8, s16
- [ ] Cechy s1 tylko dla I₇, I₉ (optymalizacja pamięci)
- [ ] STAGE 2 działa, αₖ sumują się do 1
- [ ] Forward pass bez błędów pamięci

### 12.3 Milestone 2: Flow + Synthesis

- [ ] STAGE 3 produkuje sensowne flow (wizualizacja)
- [ ] STAGE 4 backward warp działa poprawnie
- [ ] I_coarse wygląda jak rozmyta interpolacja

### 12.4 Milestone 3: Full Pipeline z s1

- [ ] STAGE 5 przyjmuje F₇ˢ¹, F₉ˢ¹ bezpośrednio
- [ ] Loss spada podczas treningu
- [ ] Wizualnie wyniki są sensowne
- [ ] Krawędzie są ostrzejsze niż w wersji bez s1

### 12.5 Milestone 4: Ewaluacja

- [ ] PSNR/SSIM/LPIPS na validation set
- [ ] Porównanie z RIFE baseline
- [ ] Ablacja: z s1 vs bez s1
- [ ] Pozostałe ablacje przeprowadzone

---

## 13. Diagram przepływu danych (podsumowanie)

```
INPUT: [I₀, I₁, ..., I₇, I₉, ..., I₁₄]  (14 klatek)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Feature Extraction                                        │
│                                                                     │
│   Wszystkie 14 klatek → s16 features                               │
│   Tylko I₇, I₉ → s1, s4, s8 features                               │
│                                                                     │
│   Output:                                                           │
│   ├── F_temporal: [B, 14, 256, H/16, W/16]                         │
│   ├── F₇ˢ¹, F₉ˢ¹: [B, 32, H, W]           ← FULL RESOLUTION       │
│   ├── F₇ˢ⁴, F₉ˢ⁴: [B, 128, H/4, W/4]                               │
│   └── F₇ˢ⁸, F₉ˢ⁸: [B, 192, H/8, W/8]                               │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Temporal Transformer                                       │
│                                                                     │
│   F_temporal → Full Self-Attention (T=14) → Adaptive Aggregation   │
│                                                                     │
│   Output:                                                           │
│   ├── F_ctx: [B, 256, H/16, W/16]                                  │
│   └── αₖ: [14] (attention weights)                                 │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Flow Estimation (2-scale cascade)                         │
│                                                                     │
│   [F₇ˢ⁸, F₉ˢ⁸, F_ctx] → s8 estimation → s4 refinement             │
│                                                                     │
│   Output:                                                           │
│   ├── flow₇ˢ⁴, flow₉ˢ⁴: [B, 2, H/4, W/4]                          │
│   └── O₇ˢ⁴, O₉ˢ⁴: [B, 1, H/4, W/4]                                 │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Coarse Synthesis                                           │
│                                                                     │
│   [I₇, I₉, flows, occlusions, F_ctx] → warp + blend + context      │
│                                                                     │
│   Output:                                                           │
│   └── I₈_coarse: [B, 3, H/4, W/4]                                  │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Full-Resolution Refinement                                 │
│                                                                     │
│   [I₈_coarse↑, F₇ˢ¹, F₉ˢ¹] → RefineNet → residual                  │
│                    ↑                                                │
│          FULL-RES FEATURES (no upsampling loss!)                   │
│                                                                     │
│   Output:                                                           │
│   └── I₈_final: [B, 3, H, W]                                       │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
              OUTPUT: Î₈
              
              Loss = L1(Î₈, I₈_GT) + λ·LPIPS(Î₈, I₈_GT)
```

---
