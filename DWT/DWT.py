"""Demonstracja ślepego znakowania wodnego w dziedzinie DWT.

Implementacja algorytmu z artykułu: D. Kundur, D. Hatzinakos, "Digital
watermarking using multiresolution wavelet decomposition", ICASSP 1998.

Podpasma szczegółów z artykułu odpowiadają tablicom PyWavelets:
h (poziome) -> cH, v (pionowe) -> cV, d (diagonalne) -> cD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pywt

EXAMPLE_DIR = Path(__file__).resolve().parent
HOST_SIZE = 512
LOGO_SIZE = 32
PADDING_MODE = "periodization"
MIN_BIN_WIDTH = 1e-9
PREVIEW_SCALE = 8


def read_grayscale(path: Path) -> np.ndarray:
    """Wczytuje obraz jako macierz 2D uint8."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Nie można zdekodować obrazu: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Zapisuje obraz bezstratnie w formacie PNG."""
    encoded = cv2.imencode(".png", image)[1]
    encoded.tofile(path)


def load_host(path: Path) -> np.ndarray:
    """Zwraca obraz nośny w skali szarości o rozmiarze 512x512."""
    return cv2.resize(read_grayscale(path), (HOST_SIZE, HOST_SIZE), interpolation=cv2.INTER_AREA)


def load_logo_bits(path: Path) -> np.ndarray:
    """Zwraca logo zbinaryzowane do wektora 1024 bitów (32x32)."""
    logo = cv2.resize(read_grayscale(path), (LOGO_SIZE, LOGO_SIZE), interpolation=cv2.INTER_AREA)
    binary = cv2.threshold(logo, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary.astype(np.uint8).ravel()


def bits_to_image(bits: np.ndarray) -> np.ndarray:
    """Zamienia wektor bitów na powiększony obraz logo."""
    logo = bits.reshape(LOGO_SIZE, LOGO_SIZE) * np.uint8(255)
    return np.kron(logo, np.ones((PREVIEW_SCALE, PREVIEW_SCALE), dtype=np.uint8))


def stack_detail_bands(bands: tuple[np.ndarray, ...]) -> np.ndarray:
    """Układa podpasma (cH, cV, cD) w tablicę trójek o kształcie (P, 3)."""
    return np.stack(bands, axis=-1).reshape(-1, 3)


def unstack_detail_bands(triples: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """Odtwarza podpasma (cH, cV, cD) z tablicy trójek."""
    return tuple(np.moveaxis(triples.reshape(*shape, 3), -1, 0))


def assign_bits_to_positions(
    position_count: int, payload_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Zwraca dla każdej pozycji poziomu indeks bitu, który ona przenosi.

    Tajna permutacja pozycji pełni rolę klucza, a cykliczne przypisanie
    realizuje wielokrotne osadzenie znaku.
    """
    assignment = np.empty(position_count, dtype=np.int64)
    assignment[rng.permutation(position_count)] = np.arange(position_count) % payload_length
    return assignment


def quantize_middle(triples: np.ndarray, bits: np.ndarray, q: int) -> np.ndarray:
    """Kwantuje środkowy współczynnik każdej trójki zgodnie z bitem znaku."""
    rows = np.arange(len(triples))
    order = np.argsort(triples, axis=1)
    low = triples[rows, order[:, 0]]
    high = triples[rows, order[:, 2]]
    middle_column = order[:, 1]

    # Równanie (2): delta = (f_k3 - f_k1) / (2Q - 1)
    delta = (high - low) / (2 * q - 1)
    usable = delta > MIN_BIN_WIDTH
    safe_delta = np.where(usable, delta, 1.0)

    # Węzły siatki low + k * delta, k = 0..2Q-1; parzystość k koduje bit.
    position = (triples[rows, middle_column] - low) / safe_delta
    node = bits + 2 * np.round((position - bits) / 2)
    node = np.clip(node, bits, 2 * q - 2 + bits)

    marked = triples.copy()
    marked[rows[usable], middle_column[usable]] = (low + node * delta)[usable]
    return marked


def read_quantized_bits(triples: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Odczytuje bity z położenia środkowych współczynników.

    Zwraca wektor bitów oraz maskę pozycji o niezerowej szerokości przedziału.
    """
    ordered = np.sort(triples, axis=1)
    low, middle, high = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    delta = (high - low) / (2 * q - 1)
    usable = delta > MIN_BIN_WIDTH
    node = np.round((middle - low) / np.where(usable, delta, 1.0))
    return node.astype(np.int64) % 2, usable


def embed(
    image: np.ndarray, payload_bits: np.ndarray, key: int, wavelet: str, level: int, q: int
) -> np.ndarray:
    """Osadza znak wodny w podpasmach szczegółów wszystkich poziomów DWT."""
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet, mode=PADDING_MODE, level=level)
    rng = np.random.default_rng(key)
    marked_coeffs: list = [coeffs[0]]
    for bands in coeffs[1:]:
        triples = stack_detail_bands(bands)
        assignment = assign_bits_to_positions(len(triples), len(payload_bits), rng)
        marked = quantize_middle(triples, payload_bits[assignment], q)
        marked_coeffs.append(unstack_detail_bands(marked, bands[0].shape))
    watermarked = pywt.waverec2(marked_coeffs, wavelet, mode=PADDING_MODE)
    return np.clip(np.round(watermarked), 0, 255).astype(np.uint8)


def extract(
    watermarked_image: np.ndarray,
    expected_length: int,
    key: int,
    wavelet: str,
    level: int,
    q: int,
    coarsest_only: bool = False,
) -> np.ndarray:
    """Odzyskuje znak wodny bez dostępu do obrazu oryginalnego.

    Każdy bit jest wynikiem głosowania większościowego po wszystkich pozycjach,
    które go przenoszą; remis rozstrzyga losowanie.
    """
    coeffs = pywt.wavedec2(
        watermarked_image.astype(np.float64), wavelet, mode=PADDING_MODE, level=level
    )
    rng = np.random.default_rng(key)
    # pywt ustawia poziomy od najgrubszego, więc klucz pierwszego poziomu
    # jest identyczny w obu trybach ekstrakcji.
    detail_levels = coeffs[1:2] if coarsest_only else coeffs[1:]
    votes = np.zeros(expected_length)
    for bands in detail_levels:
        triples = stack_detail_bands(bands)
        assignment = assign_bits_to_positions(len(triples), expected_length, rng)
        bits, usable = read_quantized_bits(triples, q)
        votes += np.bincount(
            assignment[usable], weights=2.0 * bits[usable] - 1.0, minlength=expected_length
        )
    random_guess = rng.integers(0, 2, size=expected_length)
    return np.where(votes == 0, random_guess, votes > 0).astype(np.uint8)


def jpeg_attack(image: np.ndarray, quality: int) -> np.ndarray:
    """Kompresuje i dekompresuje obraz JPEG w pamięci."""
    encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def noise_attack(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Dodaje biały szum gaussowski o odchyleniu standardowym sigma."""
    noise = np.random.default_rng(seed).normal(0.0, sigma, image.shape)
    return np.clip(np.round(image + noise), 0, 255).astype(np.uint8)


def psnr(original: np.ndarray, distorted: np.ndarray) -> float:
    """Szczytowy stosunek sygnału do szumu w dB."""
    mse = np.mean((original.astype(np.float64) - distorted.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def bit_error_rate(sent: np.ndarray, received: np.ndarray) -> float:
    """Odsetek błędnie odczytanych bitów."""
    return float(np.mean(sent != received))


def correlation(sent: np.ndarray, received: np.ndarray) -> float:
    """Współczynnik korelacji z równania (4) dla bitów odwzorowanych na {-1, 1}."""
    return float(np.mean((2.0 * sent - 1.0) * (2.0 * received - 1.0)))


def parse_arguments() -> argparse.Namespace:
    """Definiuje i odczytuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Znakowanie wodne DWT metodą Kundur-Hatzinakos (1998)."
    )
    parser.add_argument("--image", type=Path, default=EXAMPLE_DIR / "host.png", help="obraz nośny")
    parser.add_argument("--logo", type=Path, default=EXAMPLE_DIR / "logo.png", help="znak wodny")
    parser.add_argument("--out", type=Path, default=EXAMPLE_DIR / "output", help="katalog wyników")
    parser.add_argument("--key", type=int, default=42, help="tajny klucz (ziarno permutacji)")
    parser.add_argument("--wavelet", default="db5", help="falka PyWavelets (db5 = Daubechies 10-pt)")
    parser.add_argument("--level", type=int, default=3, choices=range(1, 5), help="liczba poziomów L")
    parser.add_argument("--q", type=int, default=4, help="parametr kwantyzacji Q (>= 1)")
    parser.add_argument("--jpeg", type=int, default=None, help="atak: jakość JPEG 1-100")
    parser.add_argument("--noise", type=float, default=None, help="atak: sigma szumu gaussowskiego")
    arguments = parser.parse_args()
    if arguments.q < 1:
        parser.error("--q musi być >= 1")
    if arguments.wavelet not in pywt.wavelist(kind="discrete"):
        parser.error(f"nieznana falka dyskretna: {arguments.wavelet}")
    return arguments


def apply_attacks(image: np.ndarray, arguments: argparse.Namespace) -> np.ndarray:
    """Stosuje wybrane zniekształcenia do obrazu oznakowanego."""
    attacked = image
    if arguments.jpeg is not None:
        attacked = jpeg_attack(attacked, arguments.jpeg)
    if arguments.noise is not None:
        attacked = noise_attack(attacked, arguments.noise, arguments.key + 1)
    return attacked


def print_report(
    arguments: argparse.Namespace,
    host: np.ndarray,
    watermarked: np.ndarray,
    attacked: np.ndarray,
    payload: np.ndarray,
    extracted: dict[str, np.ndarray],
) -> None:
    """Wypisuje parametry przebiegu i metryki jakości."""
    print(f"Falka: {arguments.wavelet} | poziomy L: {arguments.level} | Q: {arguments.q} "
          f"| klucz: {arguments.key} | znak: {len(payload)} bitów")
    print(f"PSNR nośnik / oznakowany:  {psnr(host, watermarked):6.2f} dB")
    if attacked is not watermarked:
        print(f"PSNR oznakowany / po ataku: {psnr(watermarked, attacked):6.2f} dB")
    print(f"{'Tryb ekstrakcji':<26}{'BER':>8}{'Korelacja':>12}")
    for mode_name, bits in extracted.items():
        print(f"{mode_name:<26}{bit_error_rate(payload, bits):>8.4f}"
              f"{correlation(payload, bits):>12.4f}")


def main() -> None:
    """Osadza znak, opcjonalnie atakuje obraz, odzyskuje znak i raportuje wyniki."""
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_arguments()
    host = load_host(arguments.image)
    payload = load_logo_bits(arguments.logo)
    settings = (arguments.key, arguments.wavelet, arguments.level, arguments.q)

    watermarked = embed(host, payload, *settings)
    attacked = apply_attacks(watermarked, arguments)
    extracted = {
        "wszystkie poziomy": extract(attacked, len(payload), *settings),
        "tylko najgrubszy poziom": extract(attacked, len(payload), *settings, coarsest_only=True),
    }

    arguments.out.mkdir(parents=True, exist_ok=True)
    write_image(arguments.out / "watermarked.png", watermarked)
    write_image(arguments.out / "attacked.png", attacked)
    write_image(arguments.out / "original_logo.png", bits_to_image(payload))
    write_image(arguments.out / "extracted_all_levels.png", bits_to_image(extracted["wszystkie poziomy"]))
    write_image(arguments.out / "extracted_coarsest.png", bits_to_image(extracted["tylko najgrubszy poziom"]))

    print_report(arguments, host, watermarked, attacked, payload, extracted)
    print(f"Wyniki zapisano w: {arguments.out}")


if __name__ == "__main__":
    main()
