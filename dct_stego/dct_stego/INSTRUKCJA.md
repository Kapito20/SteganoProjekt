# dct_stego – instrukcja obsługi

Odwracalny (bezstratny) znak wodny w obrazach JPEG według artykułu:
C.-C. Chang, C.-C. Lin, C.-S. Tseng, W.-L. Tai, *Reversible hiding in DCT-based compressed images*,
Information Sciences 177 (2007) 2768–2786.

Program działa w dwie strony:

1. **Osadzanie** – ukrywa tekst lub dowolny plik w skwantowanych współczynnikach DCT obrazu JPEG.
2. **Odczyt + odtworzenie** – wyciąga ukryte dane i przywraca oryginalne współczynniki DCT
   bit w bit. Odtworzony obraz jest identyczny z obrazem nośnym przed osadzeniem.

---

## 1. Instalacja

Wymagany jest Python 3.9 lub nowszy.

```bash
pip install -r requirements.txt      # jpeglib, numpy, pillow
python test_dct_stego.py             # opcjonalnie: testy (ok. 2 s)
```

`jpeglib` daje bezpośredni dostęp do skwantowanych współczynników DCT. Dzięki temu zapis pliku
nie dokłada kolejnej stratnej kompresji, a znak wodny daje się w pełni cofnąć.

---

## 2. Szybki start

```bash
# a) osadzenie tekstu (z kluczem)
python dct_stego.py embed obraz.jpg stego.jpg -m "© Jan Kowalski 2026" -k mojeHaslo

# b) odczyt
python dct_stego.py extract stego.jpg -k mojeHaslo

# c) odczyt + usunięcie znaku wodnego (odtworzenie oryginału)
python dct_stego.py extract stego.jpg -k mojeHaslo -r oryginal.jpg

# d) kontrola: czy odtworzony obraz jest identyczny z nośnikiem?
python dct_stego.py verify obraz.jpg oryginal.jpg      # -> IDENTYCZNE współczynniki DCT
```

---

## 3. Polecenia

### `embed` – osadzanie

```
python dct_stego.py embed NOŚNIK WYNIK.jpg (-m TEKST | -f PLIK) [opcje]
```

| Opcja | Znaczenie |
|---|---|
| `-m, --message TEKST` | tekst do ukrycia (UTF-8) |
| `-f, --file PLIK` | dowolny plik do ukrycia (np. logo, podpis, .zip) |
| `-k, --key KLUCZ` | tajny klucz: miesza kolejność bloków i szyfruje dane (XOR). Bez klucza dane odczyta każdy, kto ma ten program. |
| `-L N` | liczba zbiorów R1…RN używanych w bloku 8×8 (1–9, domyślnie 9). Mniejsze L daje lepszą jakość i mniejszą pojemność. **Przy odczycie trzeba podać to samo L.** |
| `--seed N` | ziarno losowania ±1 (tylko dla powtarzalności wyników) |
| `--table modified\|standard` | tablica kwantyzacji użyta, gdy nośnik **nie jest** JPEG-iem (domyślnie `modified`, czyli rys. 9 z artykułu) |
| `--quality Q` | współczynnik jakości 1–100 przy takiej kompresji (50 = tablica bez skalowania) |
| `--gray` | kompresuj nośnik do skali szarości |

Jeśli podasz PNG, BMP lub TIFF, program najpierw utworzy z niego JPEG `WYNIK_cover.jpg`.
To jedyny stratny krok, taki sam jak w eksperymencie z artykułu. Plik `_cover.jpg` jest
„oryginałem”, do którego wraca polecenie `extract -r`.

Program wypisuje pojemność, liczbę użytych bloków i PSNR między nośnikiem a obrazem stego.

### `extract` – odczyt i odtworzenie

```
python dct_stego.py extract STEGO.jpg [-k KLUCZ] [-L N] [-o PLIK] [-r ODTWORZONY.jpg]
```

* bez `-o` tekst jest wypisywany na ekran; dane binarne zapisz przez `-o dane.bin`,
* `-r PLIK.jpg` zapisuje obraz z usuniętym znakiem wodnym, czyli oryginał,
* zły klucz, złe L lub zmodyfikowany obraz kończą się komunikatem o błędzie
  (sprawdzana jest suma CRC32).

### `capacity` – pojemność obrazu

```bash
python dct_stego.py capacity obraz.jpg          # tabela dla L = 1..9
python dct_stego.py capacity obraz.jpg -L 4
```

Narzut na nagłówek wynosi 64 bity (32 bity długości i 32 bity CRC32).

### `compress` – przygotowanie nośnika JPEG

```bash
python dct_stego.py compress zdjecie.png nosnik.jpg --table modified --quality 50
```

Tworzy JPEG z tablicą standardową (rys. 2) lub zmodyfikowaną (rys. 9). Według artykułu tablica
zmodyfikowana daje wyższy PSNR przy prawie tej samej pojemności (rozdz. 3.4, tab. 3–6).

### `verify` – porównanie współczynników DCT

```bash
python dct_stego.py verify a.jpg b.jpg
```

Kod wyjścia: 0 – identyczne, 1 – różne.

---

## 4. Użycie z poziomu Pythona

```python
import dct_stego as ds

# pliki
rep = ds.embed_file("nosnik.jpg", "stego.jpg", b"moj znak wodny", L=9, key="haslo")
print(rep.capacity, rep.bits_embedded, rep.psnr)

dane = ds.extract_file("stego.jpg", L=9, key="haslo", restored="oryginal.jpg")
assert ds.coefficients_equal("nosnik.jpg", "oryginal.jpg")

# niski poziom – tablica współczynników Y (n_by, n_bx, 8, 8)
import jpeglib
Y = jpeglib.read_dct("nosnik.jpg").Y
bits = ds.pack_payload(b"abc", key=None)
Ys, _ = ds.embed_bits(Y, bits, L=9)
got, Yr = ds.extract_bits(Ys, L=9, restore=True)
assert ds.unpack_payload(got) == b"abc" and (Yr == Y).all()
```

---

## 5. Jak to działa (skrót artykułu)

* **Zbiory R1…R9** (rys. 4) to przekątne bloku 8×8 równoległe do przekątnej głównej:
  R1 to przekątna główna bez DC (7 elementów); R2, R4, R6, R8 leżą pod nią; R3, R5, R7, R9
  nad nią (długości 7, 7, 6, 6, 5, 5, 4, 4). Każdy zbiór czytany jest od najwyższej do
  najniższej częstotliwości.
* **b_i** to liczba kolejnych zer od najwyższej częstotliwości. Gdy **b_i ≥ 2**, zbiór przenosi
  1 bit: współczynnik z_{i,2} zostaje 0 (bit 0) albo dostaje ±1 (bit 1), wzór (3).
* **Sytuacje niejednoznaczne** (rozdz. 3.1.1–3.1.3):
  * A: `(0,…,0, x, 0, …)` → |x| + 1, wzór (4),
  * B: `(x, 0, …)` w zbiorze bez bitu → |x| + 1, wzór (5),
  * C: `(0, x, 0, …)` w zbiorze bez bitu → |x| + 1, wzór (6).
* **Odczyt**: reguły 1–6 z rozdz. 3.2. **Odtworzenie**: z_{i,2} := 0, potem cofnięcie zmian
  A, B, C według wzorów (7)–(9).
* Każdy zbiór zmienia się o co najwyżej 2 współczynniki, i to o ±1.
* Dane trafiają tylko do składowej luminancji Y. Chrominancja i tablice kwantyzacji
  pozostają bez zmian.

### Uzupełnienia względem artykułu

Zmiany są praktyczne i nie zmieniają algorytmu.

* **Ramka danych** `[długość 32 b][dane][CRC32]`. Dzięki niej odbiorca wie, ile bitów czytać
  i które zbiory odtworzyć. Osadzanie kończy się na zbiorze z ostatnim bitem, a reszta obrazu
  zostaje nietknięta.
* **Klucz** (opcjonalny). Działa jako „ustalona kolejność skanowania bloków” z kroku 2
  ekstrakcji: permutuje bloki i szyfruje ładunek strumieniem SHA-256. To prosta ochrona
  poufności, a nie uwierzytelnione szyfrowanie.
* W regule 6 (zbiór złożony z samych zer) pozycja z_{i,2} jest liczona tak samo jak przy
  osadzaniu. Dzięki temu odtwarzanie nie zmienia omyłkowo innych współczynników.
* Poprawność sprawdzono testem wyczerpującym: każdy ciąg wartości z {−2…2} o długości 4–7,
  dla bitu 0 i 1, przechodzi osadzenie, odczyt i odtworzenie bez błędu. Sprawdzono też
  przykład z rys. 7 / tab. 1 oraz pełne wypełnienie pojemności.

---

## 6. Ważne uwagi

* **Znak wodny jest kruchy.** Ponowne zapisanie JPEG-a w edytorze, zmiana rozmiaru, kadrowanie
  lub filtry niszczą dane, tak jak w artykule. Program zgłosi wtedy błąd CRC. Dzięki temu
  metoda nadaje się do wykrywania ingerencji w obraz.
* Obraz stego zapisuj i przesyłaj **jako plik**, bez ponownej kompresji (komunikatory często
  rekompresują zdjęcia).
* Przy odczycie użyj tego samego `-L` i `-k` co przy osadzaniu.
* Pojemność zależy od treści obrazu. Obrazy z dużą ilością detali (np. „Mandrill”) mają mniej
  zer w wysokich częstotliwościach, więc pomieszczą mniej danych.
* Raportowany PSNR liczony jest względem nośnika JPEG. W artykule PSNR liczono względem
  nieskompresowanego oryginału, więc tamte wartości są niższe, bo obejmują też stratę samej
  kompresji JPEG.
* Według artykułu plik stego może być większy od nośnika (rys. 17). Przy małym ładunku
  różnica jest niewielka.
