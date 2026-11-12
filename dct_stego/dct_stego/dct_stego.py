#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odwracalne ukrywanie danych (znak wodny) w obrazach JPEG – implementacja schematu:

    C.-C. Chang, C.-C. Lin, C.-S. Tseng, W.-L. Tai,
    "Reversible hiding in DCT-based compressed images",
    Information Sciences 177 (2007) 2768-2786.

Działa "w dwie strony":
  * embed   – osadza wiadomość/plik w skwantowanych współczynnikach DCT (składowa Y),
  * extract – odczytuje wiadomość ORAZ (opcjonalnie) odtwarza bit w bit oryginalne
              współczynniki DCT obrazu nośnego (bezstratne odwrócenie znaku wodnego).

Zasada działania (rozdz. 3 artykułu):
  * W każdym bloku 8x8 zdefiniowano 9 zbiorów R1..R9 – przekątne równoległe do
    głównej przekątnej (R1 = przekątna główna bez DC, R2/R4/R6/R8 – pod nią,
    R3/R5/R7/R9 – nad nią), czytane od najwyższej do najniższej częstotliwości.
  * b_i = liczba kolejnych zer licząc od najwyższej częstotliwości.
    Jeśli b_i >= 2, zbiór przenosi 1 bit: współczynnik z_{i,2} (zero tuż "za"
    najniższym zerem z_{i,1}) ustawia się na 0 (bit 0) lub ±1 (bit 1) – r. (3).
  * Sytuacje niejednoznaczne A, B, C są usuwane przez zwiększenie modułu jednego
    współczynnika o 1 (r. (4), (5), (6)); odtwarzanie cofa te zmiany (r. (7)-(9)).
  * Opcjonalnie stosuje się zmodyfikowaną tablicę kwantyzacji (rys. 9, rozdz. 3.4).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import jpeglib
except ImportError:  # pragma: no cover
    sys.exit("Brak biblioteki 'jpeglib'. Zainstaluj: pip install jpeglib numpy pillow")

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

#: Standardowa tablica kwantyzacji luminancji JPEG (rys. 2 artykułu; w artykule
#: wydrukowana jest "do góry nogami" – tu w normalnej orientacji, [0,0] = DC).
STANDARD_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int32)

#: Zmodyfikowana tablica kwantyzacji z rys. 9 artykułu (współczynniki średnich i
#: wysokich częstotliwości w zbiorach R_i przemnożone przez 0.7).
MODIFIED_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 41, 60, 55],
    [14, 13, 16, 17, 28, 40, 48, 56],
    [14, 17, 22, 20, 36, 61, 56, 43],
    [18, 22, 26, 39, 48, 76, 72, 54],
    [24, 25, 39, 45, 57, 73, 79, 64],
    [49, 64, 55, 61, 72, 85, 84, 71],
    [72, 92, 95, 69, 78, 70, 72, 70],
], dtype=np.int32)

#: Standardowa tablica chrominancji (dla obrazów kolorowych; dane i tak trafiają do Y).
STANDARD_CHROMA_QT = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.int32)

MAX_AC = 1023          # zakres współczynników AC w JPEG baseline (8 bit)
HEADER_BITS = 32       # długość ładunku w bajtach
CRC_BITS = 32          # suma kontrolna CRC32 ładunku
OVERHEAD_BITS = HEADER_BITS + CRC_BITS


def _build_sets() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Współrzędne (wiersz, kolumna) zbiorów R1..R9, od wysokiej do niskiej częstotliwości."""
    sets = []
    # R1: przekątna główna (7,7) -> (1,1), bez składowej DC.
    sets.append([(t, t) for t in range(7, 0, -1)])
    for d in range(1, 5):
        # R2, R4, R6, R8: pod przekątną (wiersz - kolumna = d), koniec w dolnym wierszu.
        below = [(c + d, c) for c in range(7 - d, -1, -1)]
        # R3, R5, R7, R9: nad przekątną (kolumna - wiersz = d), koniec w prawej kolumnie.
        above = [(r, r + d) for r in range(7 - d, -1, -1)]
        sets.append(below)
        sets.append(above)
    return [(np.array([p[0] for p in s]), np.array([p[1] for p in s])) for s in sets]


SETS = _build_sets()   # długości: 7,7,7,6,6,5,5,4,4


def _sgn(v: int) -> int:
    return 1 if v > 0 else -1


# ---------------------------------------------------------------------------
# Rdzeń algorytmu – operacje na pojedynczym zbiorze R_i (lista od wysokiej do niskiej częst.)
# ---------------------------------------------------------------------------

def leading_zeros(seq: Sequence[int]) -> int:
    """b_i – liczba kolejnych zer od najwyższej częstotliwości."""
    b = 0
    for v in seq:
        if v != 0:
            break
        b += 1
    return b


def embed_set(seq: List[int], bit: Optional[int], rng: np.random.Generator) -> bool:
    """Modyfikuje `seq` w miejscu. Zwraca True, jeśli zbiór przeniósł bit `bit`.

    Gdy zbiór nie nadaje się do osadzania (b_i < 2), stosowane są środki zaradcze
    dla sytuacji niejednoznacznych B i C, a bit NIE jest zużywany.
    """
    k = len(seq)
    b = leading_zeros(seq)
    if b >= 2:
        # Sytuacja niejednoznaczna A (r. 4): (0,...,0, x, 0, ...) -> |x| += 1
        if b + 1 < k and seq[b + 1] == 0:
            seq[b] += _sgn(seq[b])
            if abs(seq[b]) > MAX_AC:
                raise OverflowError("Przepełnienie współczynnika DCT")
        z2 = b - 2                      # indeks (0-based) z_{i,2}
        seq[z2] = 0 if bit == 0 else int(rng.choice((-1, 1)))   # r. (3)
        return True
    # Przypadek 2: zbiór bez bitu – usuwamy sytuacje B (r. 5) i C (r. 6)
    if b == 0 and seq[1] == 0:                       # (x, 0, ...)
        seq[0] += _sgn(seq[0])
    elif b == 1 and k > 2 and seq[2] == 0:           # (0, x, 0, ...)
        seq[1] += _sgn(seq[1])
    if max(abs(seq[0]), abs(seq[1])) > MAX_AC:
        raise OverflowError("Przepełnienie współczynnika DCT")
    return False


def extract_set(seq: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    """Reguły 1-6 z rozdz. 3.2. Zwraca (bit lub None, indeks z_{i,2} lub None)."""
    k = len(seq)
    j = next((t for t, v in enumerate(seq) if v != 0), None)
    if j is None:                                   # Reguła 6: same zera -> bit 0
        return 0, k - 2
    nxt = seq[j + 1] if j + 1 < k else None         # brak elementu traktujemy jak "≠ 0"
    one = abs(seq[j]) == 1
    if one and nxt == 0:                            # Reguła 1
        return 1, j
    if j + 1 <= 2:                                  # Reguły 3 i 5: brak bitu
        return None, None
    return 0, j - 2                                 # Reguły 2 i 4 (r_{j-1}=r_{j-2}=0)


def restore_set(seq: List[int], bit: Optional[int], z2: Optional[int]) -> None:
    """Reguły odtwarzania 1-3 z rozdz. 3.3 (w miejscu)."""
    k = len(seq)
    if bit is not None:
        seq[z2] = 0
        x = z2 + 2
        if z2 + 3 < k and seq[z2 + 3] == 0:         # Reguła 1, r. (7)
            seq[x] -= _sgn(seq[x])
    else:
        if seq[0] != 0 and seq[1] == 0:             # Reguła 2, r. (8)
            seq[0] -= _sgn(seq[0])
        elif seq[0] == 0 and seq[1] != 0 and k > 2 and seq[2] == 0:   # Reguła 3, r. (9)
            seq[1] -= _sgn(seq[1])


# ---------------------------------------------------------------------------
# Pomocnicze: bity, klucz, kolejność bloków
# ---------------------------------------------------------------------------

def _bytes_to_bits(data: bytes) -> List[int]:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).tolist()


def _bits_to_bytes(bits: Sequence[int]) -> bytes:
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def _keystream(key: str, n: int) -> bytes:
    out = bytearray()
    counter = 0
    seed = hashlib.sha256(("ks|" + key).encode()).digest()
    while len(out) < n:
        out += hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:n])


def _xor(data: bytes, key: Optional[str]) -> bytes:
    if not key:
        return data
    ks = _keystream(key, len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


def _block_order(n_blocks: int, key: Optional[str]) -> np.ndarray:
    """Ustalona kolejność skanowania bloków (krok 2 ekstrakcji). Z kluczem – permutacja."""
    if not key:
        return np.arange(n_blocks)
    seed = int.from_bytes(hashlib.sha256(("perm|" + key).encode()).digest()[:8], "big")
    return np.random.default_rng(seed).permutation(n_blocks)


def _check_L(L: int) -> None:
    if not 1 <= L <= 9:
        raise ValueError("L (liczba zbiorów na blok) musi być z zakresu 1..9")


# ---------------------------------------------------------------------------
# Operacje na tablicy współczynników Y (n_by, n_bx, 8, 8)
# ---------------------------------------------------------------------------

def capacity_bits(Y: np.ndarray, L: int = 9) -> int:
    """Liczba zbiorów z b_i >= 2 (= maksymalna liczba bitów), wektorowo."""
    _check_L(L)
    blocks = Y.reshape(-1, 8, 8)
    total = 0
    for rows, cols in SETS[:L]:
        seqs = blocks[:, rows, cols] != 0
        b = np.where(seqs.any(axis=1), seqs.argmax(axis=1), seqs.shape[1])
        total += int((b >= 2).sum())
    return total


def embed_bits(Y: np.ndarray, bits: Sequence[int], L: int = 9,
               key: Optional[str] = None, seed: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Osadza bity; zwraca (nowa tablica Y, liczba zmodyfikowanych bloków)."""
    _check_L(L)
    cap = capacity_bits(Y, L)
    if len(bits) > cap:
        raise ValueError(f"Za mało miejsca: potrzeba {len(bits)} bitów, pojemność {cap} bitów "
                         f"(L={L}). Zmniejsz wiadomość"
                         + (" lub zwiększ L." if L < 9 else " albo użyj większego obrazu."))
    out = Y.astype(np.int32).copy()
    blocks = out.reshape(-1, 8, 8)
    rng = np.random.default_rng(seed)
    pos = 0
    used_blocks = 0
    if not bits:
        return out.astype(Y.dtype), 0
    for bi in _block_order(blocks.shape[0], key):
        blk = blocks[bi]
        used_blocks += 1
        for rows, cols in SETS[:L]:
            seq = blk[rows, cols].tolist()
            if embed_set(seq, bits[pos], rng):
                pos += 1
            blk[rows, cols] = seq
            if pos == len(bits):
                return out.astype(Y.dtype), used_blocks
    raise RuntimeError("Nieoczekiwany brak pojemności")   # nie powinno wystąpić


def extract_bits(Y: np.ndarray, L: int = 9, key: Optional[str] = None,
                 restore: bool = False) -> Tuple[List[int], Optional[np.ndarray]]:
    """Odczytuje ładunek (z nagłówkiem długości) i opcjonalnie odtwarza oryginał."""
    _check_L(L)
    rest = Y.astype(np.int32).copy()
    blocks = rest.reshape(-1, 8, 8)
    max_bits = blocks.shape[0] * L
    bits: List[int] = []
    need = HEADER_BITS
    header_done = False
    for bi in _block_order(blocks.shape[0], key):
        blk = blocks[bi]
        for rows, cols in SETS[:L]:
            seq = blk[rows, cols].tolist()
            bit, z2 = extract_set(seq)
            if bit is not None:
                bits.append(bit)
            if restore:
                restore_set(seq, bit, z2)
                blk[rows, cols] = seq
            if not header_done and len(bits) == HEADER_BITS:
                header_done = True
                n_bytes = int("".join(map(str, bits)), 2)
                need = HEADER_BITS + 8 * n_bytes + CRC_BITS
                if need > max_bits:
                    raise ValueError("Nie znaleziono poprawnego znaku wodnego "
                                     "(zły klucz, złe L lub obraz bez danych).")
            if header_done and len(bits) == need:
                return bits, (rest.astype(Y.dtype) if restore else None)
    raise ValueError("Nie znaleziono poprawnego znaku wodnego (dane urwane).")


def pack_payload(data: bytes, key: Optional[str] = None) -> List[int]:
    enc = _xor(data, key)
    return (_bytes_to_bits(len(enc).to_bytes(4, "big")) + _bytes_to_bits(enc)
            + _bytes_to_bits(zlib.crc32(data).to_bytes(4, "big")))


def unpack_payload(bits: Sequence[int], key: Optional[str] = None) -> bytes:
    n = int("".join(map(str, bits[:HEADER_BITS])), 2)
    enc = _bits_to_bytes(bits[HEADER_BITS:HEADER_BITS + 8 * n])
    crc = int("".join(map(str, bits[HEADER_BITS + 8 * n:])), 2)
    data = _xor(enc, key)
    if zlib.crc32(data) != crc:
        raise ValueError("Błąd sumy kontrolnej CRC32 – zły klucz/L albo obraz został zmieniony.")
    return data


# ---------------------------------------------------------------------------
# Wysoki poziom: pliki
# ---------------------------------------------------------------------------

def scale_qt(qt: np.ndarray, quality: int) -> np.ndarray:
    """Skalowanie tablicy wg współczynnika jakości IJG (50 = tablica bez zmian)."""
    if not 1 <= quality <= 100:
        raise ValueError("Jakość musi być z zakresu 1..100")
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((qt * s + 50) / 100), 1, 255).astype(np.uint16)


def is_jpeg(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\xff\xd8"


def compress(src: str, dst: str, table: str = "modified", quality: int = 50,
             grayscale: bool = False) -> None:
    """Tworzy obraz nośny JPEG (kroki 1-3 eksperymentu) ze zwykłego obrazu (PNG/BMP/...)."""
    from PIL import Image
    img = Image.open(src)
    gray = grayscale or img.mode in ("L", "1", "I", "I;16", "LA")
    luma = MODIFIED_QT if table == "modified" else STANDARD_QT
    if gray:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)[:, :, None]
        im = jpeglib.from_spatial(arr)
        qt = scale_qt(luma, quality)[None]
    else:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        im = jpeglib.from_spatial(arr)
        qt = np.stack([scale_qt(luma, quality), scale_qt(STANDARD_CHROMA_QT, quality)])
    im.write_spatial(dst, qt=qt)


@dataclass
class EmbedReport:
    bits_embedded: int
    capacity: int
    blocks_used: int
    psnr: float


def psnr(a_path: str, b_path: str) -> float:
    from PIL import Image
    a = np.asarray(Image.open(a_path).convert("L"), dtype=np.float64)
    b = np.asarray(Image.open(b_path).convert("L"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)                      # r. (11)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)   # r. (10)


def embed_file(cover: str, stego: str, data: bytes, L: int = 9, key: Optional[str] = None,
               seed: Optional[int] = None) -> EmbedReport:
    im = jpeglib.read_dct(cover)
    bits = pack_payload(data, key)
    cap = capacity_bits(im.Y, L)
    im.Y, used = embed_bits(im.Y, bits, L=L, key=key, seed=seed)
    im.write_dct(stego)
    return EmbedReport(len(bits), cap, used, psnr(cover, stego))


def extract_file(stego: str, L: int = 9, key: Optional[str] = None,
                 restored: Optional[str] = None) -> bytes:
    im = jpeglib.read_dct(stego)
    bits, Yr = extract_bits(im.Y, L=L, key=key, restore=restored is not None)
    data = unpack_payload(bits, key)                 # sprawdza CRC przed zapisem
    if restored is not None:
        im.Y = Yr
        im.write_dct(restored)
    return data


def coefficients_equal(a: str, b: str) -> bool:
    ja, jb = jpeglib.read_dct(a), jpeglib.read_dct(b)
    if not np.array_equal(ja.Y, jb.Y) or not np.array_equal(ja.qt, jb.qt):
        return False
    if ja.has_chrominance != jb.has_chrominance:
        return False
    if ja.has_chrominance:
        return np.array_equal(ja.Cb, jb.Cb) and np.array_equal(ja.Cr, jb.Cr)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_message(args) -> bytes:
    if args.message is not None:
        return args.message.encode("utf-8")
    if args.file is not None:
        with open(args.file, "rb") as f:
            return f.read()
    raise SystemExit("Podaj --message TEKST albo --file PLIK")


def _prepare_cover(path: str, args) -> str:
    """Jeśli nośnik nie jest JPEG-iem, kompresuje go do JPEG obok pliku wynikowego."""
    if is_jpeg(path):
        return path
    base = os.path.splitext(args.output)[0] + "_cover.jpg"
    compress(path, base, table=args.table, quality=args.quality, grayscale=args.gray)
    print(f"[i] Nośnik nie był JPEG – skompresowano do: {base} "
          f"(tablica: {args.table}, jakość: {args.quality})")
    return base


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="dct_stego",
        description="Odwracalny znak wodny w JPEG (Chang i in., 2007).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, with_L=True):
        if with_L:
            sp.add_argument("-L", type=int, default=9,
                            help="liczba zbiorów R1..RL używanych w bloku (1-9, domyślnie 9)")
        sp.add_argument("-k", "--key", default=None,
                        help="tajny klucz: permutuje kolejność bloków i szyfruje dane (XOR)")

    sp = sub.add_parser("embed", help="osadź znak wodny / wiadomość")
    sp.add_argument("cover", help="obraz nośny (JPEG; inny format zostanie skompresowany)")
    sp.add_argument("output", help="plik wynikowy (stego) .jpg")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("-m", "--message", help="tekst do ukrycia")
    g.add_argument("-f", "--file", help="plik do ukrycia")
    sp.add_argument("--table", choices=("modified", "standard"), default="modified",
                    help="tablica kwantyzacji przy kompresji nie-JPEG (domyślnie modified)")
    sp.add_argument("--quality", type=int, default=50, help="jakość przy kompresji (domyślnie 50)")
    sp.add_argument("--gray", action="store_true", help="kompresuj nośnik do skali szarości")
    sp.add_argument("--seed", type=int, default=None, help="ziarno losowania ±1 (powtarzalność)")
    common(sp)

    sp = sub.add_parser("extract", help="odczytaj znak wodny i (opcjonalnie) odtwórz oryginał")
    sp.add_argument("stego", help="obraz z ukrytymi danymi")
    sp.add_argument("-o", "--out", help="zapisz odczytane dane do pliku (domyślnie: wypisz tekst)")
    sp.add_argument("-r", "--restore", metavar="PLIK.jpg",
                    help="zapisz odtworzony obraz oryginalny (bez znaku wodnego)")
    common(sp)

    sp = sub.add_parser("capacity", help="pokaż pojemność obrazu")
    sp.add_argument("image")
    sp.add_argument("-L", type=int, default=None, help="tylko dla podanego L")

    sp = sub.add_parser("compress", help="utwórz obraz nośny JPEG ze zwykłego obrazu")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--table", choices=("modified", "standard"), default="modified")
    sp.add_argument("--quality", type=int, default=50)
    sp.add_argument("--gray", action="store_true")

    sp = sub.add_parser("verify", help="sprawdź, czy dwa JPEG-i mają identyczne współczynniki DCT")
    sp.add_argument("a")
    sp.add_argument("b")

    args = p.parse_args(argv)
    try:
        if args.cmd == "embed":
            data = _read_message(args)
            cover = _prepare_cover(args.cover, args)
            rep = embed_file(cover, args.output, data, L=args.L, key=args.key, seed=args.seed)
            print(f"[OK] Osadzono {len(data)} B ({rep.bits_embedded} bitów z narzutem) "
                  f"w {args.output}")
            print(f"     pojemność: {rep.capacity} bitów (~{max(rep.capacity - OVERHEAD_BITS, 0) // 8} B danych), "
                  f"użyte bloki: {rep.blocks_used}, PSNR: {rep.psnr:.2f} dB")
        elif args.cmd == "extract":
            data = extract_file(args.stego, L=args.L, key=args.key, restored=args.restore)
            if args.out:
                with open(args.out, "wb") as f:
                    f.write(data)
                print(f"[OK] Zapisano {len(data)} B do {args.out}")
            else:
                try:
                    print(data.decode("utf-8"))
                except UnicodeDecodeError:
                    print(f"[i] Dane binarne ({len(data)} B) – użyj --out, aby zapisać do pliku.")
            if args.restore:
                print(f"[OK] Odtworzony oryginał: {args.restore}", file=sys.stderr)
        elif args.cmd == "capacity":
            Y = jpeglib.read_dct(args.image).Y
            Ls = [args.L] if args.L else range(1, 10)
            print(f"Bloki 8x8: {Y.shape[0] * Y.shape[1]}")
            for L in Ls:
                c = capacity_bits(Y, L)
                print(f"  L={L}: {c:7d} bitów  (~{max(c - OVERHEAD_BITS, 0) // 8} B wiadomości)")
        elif args.cmd == "compress":
            compress(args.input, args.output, args.table, args.quality, args.gray)
            print(f"[OK] Zapisano {args.output}")
        elif args.cmd == "verify":
            same = coefficients_equal(args.a, args.b)
            print("IDENTYCZNE współczynniki DCT" if same else "RÓŻNE współczynniki DCT")
            return 0 if same else 1
    except (ValueError, OverflowError, FileNotFoundError) as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
