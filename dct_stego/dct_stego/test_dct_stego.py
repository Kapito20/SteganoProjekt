#!/usr/bin/env python3
"""Testy: python test_dct_stego.py  (albo: python -m pytest test_dct_stego.py)"""
import itertools
import os
import tempfile

import numpy as np
from PIL import Image

import dct_stego as ds


def test_exhaustive_sets():
    """Każdy ciąg wartości z {-2..2} dla długości 4..7: osadź -> odczytaj -> odtwórz."""
    rng = np.random.default_rng(0)
    for k in range(4, 8):
        for seq in itertools.product(range(-2, 3), repeat=k):
            for bit in (0, 1):
                s = list(seq)
                carried = ds.embed_set(s, bit, rng)
                got, z2 = ds.extract_set(s)
                if carried:
                    assert got == bit, (seq, bit, s)
                else:
                    assert got is None, (seq, s)
                ds.restore_set(s, got, z2)
                assert s == list(seq), (seq, bit, s)


def test_paper_example():
    """Przykład z rys. 7 / tabeli 1 artykułu (bity 0,0,1,1)."""
    sets = {1: [0, 0, 0, 0, 2, 2, 3], 2: [0, 0, 1, 0, 0, 0, 3], 3: [0, 1, 0, 0, 2, 4, 2],
            4: [0, 0, 0, 1, 1, 2], 5: [0, 0, 0, 0, 0, 0]}
    expected = {1: [0, 0, 0, 0, 2, 2, 3], 2: [0, 0, 2, 0, 0, 0, 3], 3: [0, 2, 0, 0, 2, 4, 2],
                4: [0, 1, 0, 1, 1, 2], 5: [0, 0, 0, 0, 1, 0]}
    bits = iter([0, 0, 1, 1])
    rng = np.random.default_rng(1)
    pending = next(bits)
    for i in range(1, 6):
        s = sets[i]
        if ds.embed_set(s, pending, rng):
            pending = next(bits, None)
        assert [abs(v) for v in s] == expected[i], (i, s)


def _test_image(path, size=256, color=False):
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 60 * np.sin(x / 17.0) * np.cos(y / 23.0) + 0.3 * (x - y)
    noise = np.random.default_rng(3).normal(0, 6, (size, size))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    if color:
        img = np.stack([img, np.roll(img, 9, 0), 255 - img], axis=2)
    Image.fromarray(img).save(path)


def test_file_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        for color in (False, True):
            for table in ("standard", "modified"):
                src = os.path.join(d, "src.png")
                cover = os.path.join(d, "cover.jpg")
                stego = os.path.join(d, "stego.jpg")
                restored = os.path.join(d, "restored.jpg")
                _test_image(src, color=color)
                ds.compress(src, cover, table=table, quality=50)
                msg = "Znak wodny © 2026 – zażółć gęślą jaźń".encode()
                ds.embed_file(cover, stego, msg, L=9, key="tajny")
                assert not ds.coefficients_equal(cover, stego)
                assert ds.extract_file(stego, L=9, key="tajny", restored=restored) == msg
                assert ds.coefficients_equal(cover, restored)
                a = np.asarray(Image.open(cover)); b = np.asarray(Image.open(restored))
                assert np.array_equal(a, b)
                try:
                    ds.extract_file(stego, L=9, key="zly")
                    raise AssertionError("zły klucz powinien zawieść")
                except ValueError:
                    pass


def test_full_capacity():
    with tempfile.TemporaryDirectory() as d:
        src, cover = os.path.join(d, "s.png"), os.path.join(d, "c.jpg")
        _test_image(src)
        ds.compress(src, cover, table="standard")
        for L in (1, 4, 9):
            Y = ds.jpeglib.read_dct(cover).Y
            cap = ds.capacity_bits(Y, L)
            payload = os.urandom((cap - ds.OVERHEAD_BITS) // 8)
            bits = ds.pack_payload(payload)
            Ys, _ = ds.embed_bits(Y, bits, L=L)
            got, Yr = ds.extract_bits(Ys, L=L, restore=True)
            assert ds.unpack_payload(got) == payload
            assert np.array_equal(Yr, Y)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("Wszystkie testy zaliczone.")
