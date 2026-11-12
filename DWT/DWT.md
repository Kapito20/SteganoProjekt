# Znakowanie wodne w dziedzinie DWT – demonstracja

Aplikacja `DWT.py` osadza binarne logo w obrazie w skali szarości, a następnie odzyskuje je bez dostępu do obrazu oryginalnego. Implementuje algorytm z artykułu:

> D. Kundur, D. Hatzinakos, *Digital watermarking using multiresolution wavelet decomposition*, ICASSP 1998.

## 1. Uruchomienie

Folder jest samodzielny: zawiera kod, obraz nośny (`host.png`) i znak wodny (`logo.png`).

```
pip install -r requirements.txt
python DWT.py
python DWT.py --jpeg 75
python DWT.py --q 2 --noise 5
```

| Argument | Domyślnie | Znaczenie |
|---|---|---|
| `--image` | `host.png` | obraz nośny (skalowany do 512x512, skala szarości) |
| `--logo` | `logo.png` | znak wodny (binaryzowany do 32x32 = 1024 bity) |
| `--out` | `output/` | katalog na obrazy wynikowe |
| `--key` | `42` | tajny klucz: ziarno permutacji pozycji |
| `--wavelet` | `db5` | falka; `db5` to 10-punktowa falka Daubechies z artykułu |
| `--level` | `3` | liczba poziomów dekompozycji L (1–4) |
| `--q` | `4` | parametr kwantyzacji Q |
| `--jpeg` | brak | atak: kompresja JPEG o podanej jakości (1–100) |
| `--noise` | brak | atak: szum gaussowski o podanym odchyleniu standardowym |

Program wypisuje PSNR, BER i współczynnik korelacji oraz zapisuje w `output/` pliki `watermarked.png`, `attacked.png`, `original_logo.png`, `extracted_all_levels.png` i `extracted_coarsest.png`.

## 2. Idea: dlaczego DWT

Dyskretna transformata falkowa rozkłada obraz na pasma o szerokości około jednej oktawy. Jeden poziom dekompozycji daje cztery podpasma o połowie rozdzielczości:

```
+---------+---------+
|   LL    |    h    |     LL – aproksymacja (zgrubna wersja obrazu)
| (cA)    |  (cH)   |     h  – szczegóły poziome
+---------+---------+     v  – szczegóły pionowe
|    v    |    d    |     d  – szczegóły diagonalne
|  (cV)   |  (cD)   |
+---------+---------+
```

Kolejny poziom powstaje przez ponowny rozkład podpasma LL. Dla obrazu 512x512 i L = 3 otrzymujemy podpasma szczegółów o rozmiarach 256x256 (poziom 1), 128x128 (poziom 2) i 64x64 (poziom 3) oraz aproksymację 64x64.

W nawiasach podano nazwy tablic zwracanych przez `pywt.wavedec2`. Aplikacja używa trybu `periodization`, dzięki czemu rozmiary podpasm są dokładnie potęgami dwójki.

Algorytm wykorzystuje dwie własności tej reprezentacji:

* **Lokalizacja w przestrzeni i w częstotliwości jednocześnie.** Każda pozycja `(m, n)` na poziomie `l` opisuje konkretny fragment obrazu w konkretnym paśmie częstotliwości. Znak rozłożony po wszystkich poziomach jest więc rozproszony zarówno w przestrzeni, jak i w widmie.
* **Trzy współczynniki szczegółów w każdej pozycji.** W pozycji `(m, n)` poziomu `l` istnieje trójka `(h, v, d)`. Bit zapisuje się we wzajemnym położeniu tych trzech wartości, a nie w wartości bezwzględnej, dlatego do odczytu nie potrzeba obrazu oryginalnego.

Aproksymacja LL pozostaje nietknięta, co ogranicza widoczność znaku.

## 3. Osadzanie

Dla każdej pozycji `(l, m, n)` wybranej przez klucz:

1. Posortuj trójkę współczynników szczegółów (równanie 1 z artykułu):

   `f_k1 ≤ f_k2 ≤ f_k3`, gdzie `k1, k2, k3 ∈ {h, v, d}`

2. Podziel przedział między najmniejszym a największym współczynnikiem na `2Q − 1` części o szerokości (równanie 2):

   `Δ = (f_k3 − f_k1) / (2Q − 1)`

3. Przesuń środkowy współczynnik `f_k2` do najbliższego węzła siatki `f_k1 + k·Δ`, którego numer `k` ma parzystość odpowiadającą bitowi: `k` nieparzyste dla bitu 1, `k` parzyste dla bitu 0.

Siatka dla Q = 4 (8 węzłów, `k = 0…7`):

```
 f_k1                                             f_k3
  :      |      :      |      :      |      :      |
 k=0    k=1    k=2    k=3    k=4    k=5    k=6    k=7
  0      1      0      1      0      1      0      1     <- kodowany bit
  <--Δ-->
```

Najmniejszy i największy współczynnik nie są zmieniane, więc dekoder odtworzy tę samą siatkę. Znak (1024 bity) jest krótszy niż liczba pozycji (65 536 + 16 384 + 4096 = 86 016 dla L = 3), dlatego każdy bit jest osadzany wielokrotnie: 64 razy na poziomie 1, 16 razy na poziomie 2 i 4 razy na poziomie 3.

**Klucz.** Na każdym poziomie generator pseudolosowy o ziarnie `--key` tworzy permutację pozycji. Pozycja o numerze `i` w tej permutacji przenosi bit `i mod N`. Bez klucza nie wiadomo, która pozycja przenosi który bit.

### Pseudokod osadzania

```
WEJŚCIE: obraz f (512x512), bity w[0..N-1], klucz, falka, L, Q
WYJŚCIE: obraz oznakowany

[LL, (h_L, v_L, d_L), ..., (h_1, v_1, d_1)] ← DWT(f, falka, L)
generator ← PRNG(klucz)

DLA poziomu l = L, L-1, ..., 1:
    kolejność ← losowa permutacja pozycji (m, n) poziomu l z generatora
    DLA i = 0, 1, ..., liczba_pozycji - 1:
        (m, n) ← kolejność[i]
        bit ← w[i mod N]

        (f_k1, f_k2, f_k3) ← posortuj rosnąco (h_l[m,n], v_l[m,n], d_l[m,n])
        Δ ← (f_k3 - f_k1) / (2Q - 1)
        JEŻELI Δ = 0: POMIŃ pozycję

        t ← (f_k2 - f_k1) / Δ
        k ← bit + 2 · zaokrąglij((t - bit) / 2)      # najbliższy węzeł o parzystości bitu
        k ← ogranicz(k, bit, 2Q - 2 + bit)           # węzeł musi leżeć w [f_k1, f_k3]
        f_k2 ← f_k1 + k · Δ                          # zapis w podpasmie, z którego pochodził f_k2

obraz ← IDWT([LL, zmodyfikowane podpasma], falka)
ZWRÓĆ obetnij(zaokrąglij(obraz), 0, 255) jako uint8
```

## 4. Ekstrakcja

Dekoder zna klucz, falkę, L i Q. Wykonuje tę samą dekompozycję obrazu badanego `r`, w każdej pozycji odtwarza siatkę z aktualnych wartości skrajnych i sprawdza, przy którym węźle leży środkowy współczynnik. Każdy bit znaku jest wynikiem głosowania większościowego po wszystkich pozycjach, które go przenoszą. Remis rozstrzyga losowanie.

Aplikacja wykonuje ekstrakcję w dwóch trybach opisanych w artykule:

* **wszystkie poziomy** – głosują pozycje z poziomów 1…L,
* **tylko najgrubszy poziom** – głosują wyłącznie pozycje poziomu L.

### Pseudokod ekstrakcji

```
WEJŚCIE: obraz r, długość znaku N, klucz, falka, L, Q, tryb
WYJŚCIE: bity ŵ[0..N-1]

[LL, (h_L, v_L, d_L), ..., (h_1, v_1, d_1)] ← DWT(r, falka, L)
generator ← PRNG(klucz)
głosy[0..N-1] ← 0

poziomy ← {L} JEŻELI tryb = "najgrubszy" W PRZECIWNYM RAZIE {L, L-1, ..., 1}
DLA poziomu l w poziomach:
    kolejność ← losowa permutacja pozycji poziomu l z generatora
    DLA i = 0, 1, ..., liczba_pozycji - 1:
        (m, n) ← kolejność[i]
        (r_k1, r_k2, r_k3) ← posortuj rosnąco (h_l[m,n], v_l[m,n], d_l[m,n])
        Δ ← (r_k3 - r_k1) / (2Q - 1)
        JEŻELI Δ = 0: POMIŃ pozycję

        k ← zaokrąglij((r_k2 - r_k1) / Δ)
        JEŻELI k nieparzyste: głosy[i mod N] ← głosy[i mod N] + 1
        W PRZECIWNYM RAZIE:    głosy[i mod N] ← głosy[i mod N] - 1

DLA j = 0, ..., N-1:
    ŵ[j] ← 1 gdy głosy[j] > 0;  0 gdy głosy[j] < 0;  los gdy głosy[j] = 0
ZWRÓĆ ŵ
```

## 5. Miary jakości

* **PSNR** – jakość obrazu oznakowanego względem nośnika: `10·log10(255² / MSE)`. Powyżej około 38 dB zmiany są zwykle niewidoczne.
* **BER** – odsetek błędnie odczytanych bitów. Wartość 0,5 oznacza odczyt losowy.
* **Współczynnik korelacji** (równanie 4 z artykułu) dla bitów odwzorowanych na `{−1, 1}`:

  `ρ(w, ŵ) = Σ w(n)·ŵ(n) / (√Σ w²(n) · √Σ ŵ²(n)) = 1 − 2·BER`

  Znak uznaje się za wykryty, gdy `ρ ≥ T`. Dla obrazu bez znaku `ρ ≈ 0`.

## 6. Wpływ parametrów

* **Q** – większe Q daje gęstszą siatkę: mniejsze zmiany współczynników (wyższy PSNR), ale mniejszy margines błędu `Δ/2`, więc słabszą odporność.
* **L** – więcej poziomów oznacza osadzanie także w niższych częstotliwościach, które lepiej znoszą kompresję i filtrację, kosztem widoczności.
* **Falka** – jest elementem tajemnicy: bez znajomości falki nie da się odtworzyć współczynników.
* **Klucz** – ustala, która pozycja przenosi który bit.

## 7. Wyniki (`host.png`, `logo.png`, falka `db5`, klucz 42)

| Ustawienia | Atak | PSNR znakowania | BER, wszystkie poziomy | BER, najgrubszy poziom |
|---|---|---|---|---|
| L=3, Q=4 | brak | 46,06 dB | 0,0000 | 0,0127 |
| L=3, Q=4 | JPEG 75 | 46,06 dB | 0,3613 | 0,1738 |
| L=3, Q=4 | szum σ=5 | 46,06 dB | 0,4268 | 0,3330 |
| L=3, Q=2 | brak | 38,63 dB | 0,0000 | 0,0000 |
| L=3, Q=2 | JPEG 75 | 38,63 dB | 0,2559 | 0,0537 |
| L=3, Q=2 | szum σ=5 | 38,63 dB | 0,2910 | 0,1641 |
| L=4, Q=2 | JPEG 75 | 35,60 dB | 0,2129 | 0,0283 |

Wnioski:

* Bez ataku znak odczytywany jest bezbłędnie ze wszystkich poziomów.
* Po ataku odczyt z samego najgrubszego poziomu jest wyraźnie lepszy niż głosowanie po wszystkich poziomach. Potwierdza to obserwację z artykułu: zniekształcenia niszczą głównie wysokie częstotliwości, a liczne, ale zniszczone głosy z poziomu 1 przegłosowują poprawne głosy z poziomu L.
* Zmniejszenie Q z 4 do 2 obniża PSNR o około 7,5 dB, ale kilkukrotnie zmniejsza BER po kompresji JPEG.
* Metoda jest wrażliwa na zniekształcenia, bo margines błędu `Δ/2` zależy od lokalnego kontrastu: w gładkich obszarach obrazu trzy współczynniki są niemal równe i siatka jest bardzo gęsta.

## 8. Uwagi implementacyjne i odstępstwa od artykułu

* **Zaokrąglenie do 8 bitów.** Obraz oznakowany jest zapisywany jako `uint8`, co wprowadza do współczynników błąd o odchyleniu około 0,29. W gładkich obszarach (małe Δ) to wystarcza, by zmienić odczytany bit. Stąd BER = 0,0127 w trybie najgrubszego poziomu przy Q = 4 bez ataku: każdy bit ma tam tylko 4 powtórzenia. Przy głosowaniu po wszystkich poziomach (84 powtórzenia) błąd znika.
* **Klucz.** Artykuł opisuje klucz jako losową maskę binarną wybierającą pozycje. Tutaj oznaczane są wszystkie pozycje, a klucz jest permutacją określającą przypisanie bitów do pozycji. Permutacja jest losowana osobno na każdym poziomie, aby każdy poziom zawierał komplet bitów znaku, co umożliwia ekstrakcję z samego najgrubszego poziomu.
* **Reprezentacja bitów.** Artykuł używa wartości `{−1, 1}`, aplikacja `{0, 1}` (0 odpowiada −1).
* **Binaryzacja logo.** Logo jest skalowane do 32x32 i progowane metodą Otsu.
* **Pozycje z Δ = 0** (trzy równe współczynniki) nie przenoszą bitu i nie biorą udziału w głosowaniu.
