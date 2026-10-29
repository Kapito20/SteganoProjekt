#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

START_ROW = 199
START_COL = 0
PIXEL_STRIDE = 5
MAX_MESSAGE_BYTES = 1023
FOURTH_LSB_MASK = 0b00001000
THIRD_LSB_MASK = 0b00000100
TARGET_MASK = FOURTH_LSB_MASK | THIRD_LSB_MASK  # 0b00001100


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as an 8-bit grayscale NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def save_grayscale(array: np.ndarray, path: str | Path) -> None:
    """Save a grayscale array, preserving the embedded pixel values."""
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def selected_coordinates(height: int, width: int):
    """
    Yield pixel coordinates in the paper's traversal order.
    Start at (199, 0), advance 5 columns each time, and when the row
    is exhausted, advance 5 rows and restart at column 0.
    """
    if height <= START_ROW or width <= START_COL:
        raise ValueError(
            f"Image must be at least {START_ROW + 1} pixels high and "
            f"{START_COL + 1} pixels wide; got {height}x{width}."
        )

    row, col = START_ROW, START_COL
    while row < height:
        if col < width:
            yield row, col
            col += PIXEL_STRIDE
        else:
            row += PIXEL_STRIDE
            col = START_COL


def bits_msb_first(data: bytes) -> list[int]:
    """Return each byte's bits in most-significant-bit-first order."""
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def set_pair(pixel: int, first_bit: int, second_bit: int) -> int:
    """Set the 4th LSB to first_bit and the 3rd LSB to second_bit."""
    pixel = int(pixel) & (~TARGET_MASK & 0xFF)
    pixel |= (int(first_bit) & 1) << 3
    pixel |= (int(second_bit) & 1) << 2
    return pixel


def get_pair(pixel: int) -> tuple[int, int]:
    """Read (4th LSB, 3rd LSB), in the same order used for embedding."""
    pixel = int(pixel)
    return (pixel >> 3) & 1, (pixel >> 2) & 1


def embed_message(input_path: str | Path, message: str, output_path: str | Path) -> None:
    """
    Embed UTF-8 message bytes in a grayscale image and save the result.

    The paper describes a character limit of 1023. This implementation
    enforces that limit in bytes so UTF-8 messages are extracted exactly.
    """
    payload = message.encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Message is {len(payload)} UTF-8 bytes; maximum is "
            f"{MAX_MESSAGE_BYTES} bytes."
        )

    image = load_grayscale(input_path)
    height, width = image.shape
    coords = selected_coordinates(height, width)

    header_bits = [(len(payload) >> shift) & 1 for shift in range(9, -1, -1)]
    payload_bits = bits_msb_first(payload)
    all_bits = header_bits + payload_bits

    needed_pixels = math.ceil(len(all_bits) / 2)
    available_coords = list(coords)
    if len(available_coords) < needed_pixels:
        raise ValueError(
            f"Image capacity is insufficient: need {needed_pixels} selected "
            f"pixels, but only {len(available_coords)} are available."
        )

    for index in range(needed_pixels):
        first = all_bits[index * 2]
        second = all_bits[index * 2 + 1] if index * 2 + 1 < len(all_bits) else 0
        row, col = available_coords[index]
        image[row, col] = set_pair(image[row, col], first, second)

    save_grayscale(image, output_path)


def extract_message(input_path: str | Path) -> str:
    """Extract a UTF-8 message from an image watermarked by this script."""
    image = load_grayscale(input_path)
    height, width = image.shape
    coords = selected_coordinates(height, width)

    coord_iter = iter(coords)

    def read_bits(count: int) -> list[int]:
        result: list[int] = []
        for _ in range(math.ceil(count / 2)):
            try:
                row, col = next(coord_iter)
            except StopIteration as exc:
                raise ValueError("Image ended before the embedded data was complete.") from exc
            first, second = get_pair(image[row, col])
            result.extend((first, second))
        return result[:count]

    header = read_bits(10)
    message_length = 0
    for bit in header:
        message_length = (message_length << 1) | bit

    if message_length > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Decoded length ({message_length}) exceeds the supported maximum. "
            "The image may not contain a watermark made by this algorithm."
        )

    payload_bits = read_bits(message_length * 8)
    payload = bytearray()
    for start in range(0, len(payload_bits), 8):
        value = 0
        for bit in payload_bits[start:start + 8]:
            value = (value << 1) | bit
        payload.append(value)

    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Extracted bytes are not valid UTF-8. The image may not contain "
            "a compatible watermark or may have been altered."
        ) from exc


def calculate_psnr(original_path: str | Path, watermarked_path: str | Path) -> float:
    """Calculate PSNR (dB) between two grayscale images of the same dimensions."""
    original = load_grayscale(original_path).astype(np.float64)
    watermarked = load_grayscale(watermarked_path).astype(np.float64)
    if original.shape != watermarked.shape:
        raise ValueError("Original and watermarked images must have the same dimensions.")
    mse = float(np.mean((original - watermarked) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed/extract a message using the paper's 3rd/4th-LSB method."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Embed a message in an image.")
    embed_parser.add_argument("--input", required=True, help="Cover image path.")
    source = embed_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="Message text to embed.")
    source.add_argument("--message-file", help="UTF-8 text file containing the message.")
    embed_parser.add_argument("--output", required=True, help="Output watermarked image path.")

    extract_parser = subparsers.add_parser("extract", help="Extract a message from an image.")
    extract_parser.add_argument("--input", required=True, help="Watermarked image path.")
    extract_parser.add_argument("--output", help="Optional text file for extracted message.")

    psnr_parser = subparsers.add_parser("psnr", help="Calculate image PSNR.")
    psnr_parser.add_argument("--original", required=True, help="Original cover image.")
    psnr_parser.add_argument("--watermarked", required=True, help="Watermarked image.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "embed":
            if args.message_file:
                message = Path(args.message_file).read_text(encoding="utf-8")
            else:
                message = args.message
            embed_message(args.input, message, args.output)
            print(f"Watermarked image saved to: {args.output}")
            print(f"Message length: {len(message.encode('utf-8'))} UTF-8 bytes")
        elif args.command == "extract":
            message = extract_message(args.input)
            if args.output:
                Path(args.output).write_text(message, encoding="utf-8")
                print(f"Extracted message saved to: {args.output}")
            else:
                print(message)
        elif args.command == "psnr":
            value = calculate_psnr(args.original, args.watermarked)
            print("PSNR: infinite (images are identical)" if math.isinf(value)
                  else f"PSNR: {value:.4f} dB")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
