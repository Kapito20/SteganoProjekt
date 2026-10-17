# System Znakowania Wodnego Obrazów (Digital Watermarking Framework)

Modułowe środowisko programistyczne w języku Python przeznaczone do implementacji, testowania i wizualizacji algorytmów cyfrowego znakowania wodnego. Projekt został zaprojektowany w architekturze wtyczek (pluginów) – panel główny automatycznie integruje każdy nowy algorytm spełniający określony interfejs, wymagając od programisty jedynie definicji matematycznej logiki osadzania i ekstrakcji.

---

## 1. Założenia Projektowe

*   **Format i Wymiary Danych:** System przetwarza wyłącznie obrazy o stałej rozdzielczości **512x512 pikseli** w formacie bezstratnym (np. PNG, BMP), co zapobiega niszczeniu znaków wodnych przez kompresję.
*   **Architektura Modułowa:** Rdzeń aplikacji odpowiada za potok przetwarzania (wczytywanie danych, automatyzację testów, obliczanie metryk jakościowych, generowanie wykresów). Skrypty algorytmów są odizolowane i zawierają **wyłącznie matematyczną logikę** osadzania (`encrypt`) i ekstrakcji (`decrypt`) znaku.
*   **Zestaw Algorytmów:** Projekt implementuje i porównuje ze sobą 7 algorytmów podzielonych na 3 kategorie:
    *   *Niewidoczne w dziedzinie przestrzennej:* **LSB** (Least Significant Bit).
    *   *Niewidoczne w dziedzinie częstotliwości:* **DCT** (Discrete Cosine Transform), **DWT** (Discrete Wavelet Transform), **SVD** (Singular Value Decomposition).
    *   *Niewidoczne odporne:* **Spread Spectrum** (Widmo Rozproszone).
    *   *Widoczne:* **Visible Alpha** (Miksowanie przestrzenne), **Visible Freq** (Miksowanie w pasmach częstotliwości).
*   **Automatyzacja i Ewaluacja:** Każdy algorytm po uruchomieniu automatycznie podlega standaryzowanej ocenie zniekształceń obrazu nosiciela (PSNR, SSIM) oraz poprawności odzyskania znaku wodnego (NC / BER).

---

## 2. Struktura Projektu

```text
watermarking-framework/
│
├── core/                       # Serce systemu (wspólna logika)
│   ├── __init__.py
│   ├── base_algorithm.py       # Klasa abstrakcyjna (szablon dla algorytmów)
│   ├── evaluator.py            # Moduł obliczający metryki (PSNR, SSIM)
│   └── visualizer.py           # Narzędzia do automatycznego generowania wykresów
│
├── algorithms/                 # Folder na moduły algorytmów (pluginy)
│   ├── __init__.py
│   ├── lsb.py                  # Każdy plik implementuje metody encrypt/decrypt
│   ├── dct.py
│   ├── dwt.py
│   ├── svd.py
│   ├── spread_spectrum.py
│   ├── visible_alpha.py
│   └── visible_freq.py
│
├── data/                       # Pliki wejściowe (tylko 512x512)
│   ├── cover/                  # Oryginalne obrazy testowe
│   └── watermarks/             # Znaki do ukrycia (graficzne lub binarne)
│
├── output/                     # Rezultaty działania (generowane automatycznie)
│   ├── stego_images/           # Obrazy z wstrzykniętym znakiem wodnym
│   ├── extracted/              # Odzyskane znaki wodne i zrekonstruowane obrazy
│   └── reports/                # Wykresy, histogramy i tabele z metrykami
│
├── .gitignore
├── main.py                     # Skrypt uruchomieniowy (CLI / pętla automatyzująca)
└── requirements.txt            # Spis zależności bibliotecznych
```

---

## 3. Obsługa i Rozbudowa Systemu

### Tworzenie Nowego Modułu (Rozbudowa)
Aby dodać nowy algorytm (np. `xxx.py`), wystarczy stworzyć nowy plik w katalogu `algorithms/` dziedziczący po klasie bazowej. Moduł musi implementować tylko dwie metody zgodnie z poniższym szablonem:

```python
# algorithms/xxx.py
from core.base_algorithm import BaseWatermarkAlgorithm
import numpy as np

class XxxAlgorithm(BaseWatermarkAlgorithm):
    def encrypt(self, cover_image: np.ndarray, watermark: np.ndarray, **kwargs) -> np.ndarray:
        """
        Przyjmuje obrazek 512x512 oraz znak wodny jako macierze NumPy.
        Zwraca przetworzony obraz (stego_image) o wymiarach 512x512.
        """
        stego_image = cover_image.copy()
        # [MIEJSCE NA LOGIKĘ ZNAKOWANIA]
        return stego_image

    def decrypt(self, stego_image: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Przyjmuje oznakowany obrazek 512x512.
        Zwraca krotkę: (odzyskany_znak_wodny, oczyszczony_obraz_oryginalny).
        """
        # [MIEJSCE NA LOGIKĘ EKSTRAKCJI I ODZYSKIWANIA]
        extracted_watermark = np.zeros_like(stego_image) 
        cleaned_cover = stego_image.copy()                 
        return extracted_watermark, cleaned_cover
```

### Uruchomienie Potoku i Automatyzacji
Framework sterowany jest automatycznie za pomocą skryptu głównego `main.py`.

1. **Test pojedynczego algorytmu:**
   ```bash
   python main.py --algo lsb --cover lena.png --wm logo.png
   ```
   *System załaduje pliki, wywoła `encrypt`, następnie `decrypt`, obliczy zniekształcenia, zapisze pliki wynikowe do `output/` i wygeneruje raport wizualny dla modułu LSB.*

2. **Automatyczny test porównawczy (Batch mode):**
   ```bash
   python main.py --run-all
   ```
   *Skrypt automatycznie przeskanuje folder `algorithms/`, dynamicznie załaduje wszystkie znajdujące się tam klasy algorytmów, wykona dla każdego proces szyfrowania/deszyfrowania na tej samej parze obrazów, a na koniec wypluje zbiorczy wykres porównawczy i tabelę z metrykami w folderze `output/reports/`.*
