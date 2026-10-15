#!/usr/bin/env python3
"""
Steganografia LSB na 3. i 4. najmniej znaczącym bicie (implementacja algorytmu z artykułu).

Algorytm (współrzędne w artykule są liczone od 1, tak jak w MATLAB-ie):
  * start w pikselu (y=200, x=1), skok o 5 pikseli w poziomie,
  * gdy x wyjdzie poza szerokość obrazu -> y += 5, x = 1,
  * każdy piksel niesie 2 bity: pierwszy w 4. LSB (bit nr 3), drugi w 3. LSB (bit nr 2),
  * najpierw 10 bitów długości wiadomości (5 pikseli), potem 8 bitów na każdy bajt
    wiadomości (4 piksele na bajt),
  * maksymalna długość wiadomości: 1023 (bo długość ma 10 bitów).

Uwaga: wiadomość jest kodowana w UTF-8, więc limit 1023 dotyczy bajtów
(polskie litery zajmują 2 bajty). Dla czystego ASCII bajty = znaki.

Użycie:
    python stego_lsb.py                      -> GUI (tkinter; jeśli go brak - interfejs w przeglądarce)
    python stego_lsb.py web [--port 8000]    -> interfejs w przeglądarce (bez tkintera)
    python stego_lsb.py embed  in.png out.bmp -m "tajna wiadomość" [-y 200 -x 1]
    python stego_lsb.py extract out.bmp [-y 200 -x 1]
    python stego_lsb.py compare in.png out.bmp

Wymagania: pip install pillow numpy   (tkinter jest opcjonalny)
Obraz wynikowy zapisuj w formacie bezstratnym (BMP/PNG) - JPEG zniszczy ukryte dane.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MAX_LEN = 1023      # 10 bitów długości -> max 2^10 - 1
LEN_BITS = 10
STEP = 5            # skok między pikselami
BIT_HI = 3          # 4. LSB (pierwszy bit z pary)
BIT_LO = 2          # 3. LSB (drugi bit z pary)
CLEAR_MASK = 0xFF & ~((1 << BIT_HI) | (1 << BIT_LO))   # 0b11110011
DEFAULT_Y, DEFAULT_X = 200, 1


class StegoError(Exception):
    """Błąd użytkownika / danych (zły rozmiar, za długa wiadomość itp.)."""


# --------------------------------------------------------------------------- #
#  Rdzeń algorytmu
# --------------------------------------------------------------------------- #
def load_gray(path: str | Path) -> np.ndarray:
    """Wczytuje obraz i konwertuje do skali szarości (uint8), jak w artykule."""
    with Image.open(path) as im:
        return np.array(im.convert("L"), dtype=np.uint8)


def encode_image(arr: np.ndarray, fmt: str = "BMP") -> bytes:
    fmt = fmt.upper()
    if fmt not in ("BMP", "PNG"):
        raise StegoError("Obsługiwane formaty wyjściowe: BMP i PNG (bezstratne).")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def save_image(arr: np.ndarray, path: str | Path) -> None:
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        raise StegoError("Format JPEG jest stratny i zniszczy ukryte dane. Użyj BMP lub PNG.")
    Path(path).write_bytes(encode_image(arr, "PNG" if ext == ".png" else "BMP"))


def _positions(shape: tuple[int, int], y: int, x: int):
    """Generator kolejnych pikseli (0-indeksowanych) zgodnie z regułą skoku o 5."""
    m, n = shape  # m = wiersze, n = kolumny
    if not (1 <= y <= m and 1 <= x <= n):
        raise StegoError(f"Współrzędne startowe (y={y}, x={x}) leżą poza obrazem {n}x{m}.")
    y -= 1
    x -= 1
    while True:
        if x >= n:
            y += STEP
            x = 0
        if y >= m:
            raise StegoError("Za mało miejsca w obrazie na tę wiadomość.")
        yield y, x
        x += STEP


def embed(img: np.ndarray, message: str, y: int = DEFAULT_Y, x: int = DEFAULT_X) -> np.ndarray:
    """Zwraca kopię obrazu z ukrytą wiadomością."""
    data = message.encode("utf-8")
    if not data:
        raise StegoError("Wiadomość jest pusta.")
    if len(data) > MAX_LEN:
        raise StegoError(
            f"Wiadomość ma {len(data)} bajtów - maksimum to {MAX_LEN}. Skróć ją i spróbuj ponownie."
        )

    bits = format(len(data), f"0{LEN_BITS}b") + "".join(format(b, "08b") for b in data)
    out = img.copy()
    pos = _positions(img.shape, y, x)
    for k in range(0, len(bits), 2):
        r, c = next(pos)
        hi, lo = int(bits[k]), int(bits[k + 1])
        out[r, c] = (int(out[r, c]) & CLEAR_MASK) | (hi << BIT_HI) | (lo << BIT_LO)
    return out


def extract(img: np.ndarray, y: int = DEFAULT_Y, x: int = DEFAULT_X) -> str:
    """Odczytuje ukrytą wiadomość z obrazu."""
    pos = _positions(img.shape, y, x)

    def read_bits(nbits: int) -> str:
        chunks = []
        for _ in range(nbits // 2):
            r, c = next(pos)
            v = int(img[r, c])
            chunks.append(str((v >> BIT_HI) & 1))
            chunks.append(str((v >> BIT_LO) & 1))
        return "".join(chunks)

    length = int(read_bits(LEN_BITS), 2)
    if length == 0:
        raise StegoError("Nie znaleziono wiadomości (długość = 0). Sprawdź współrzędne startowe.")
    bits = read_bits(length * 8)
    data = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return data.decode("utf-8", errors="replace")


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Porównanie oryginału z obrazem ze znakiem wodnym."""
    if a.shape != b.shape:
        raise StegoError("Obrazy mają różne rozmiary.")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    return {
        "changed": int(np.count_nonzero(diff)),
        "max_diff": int(diff.max()),
        "mse": mse,
        "psnr": psnr,
        "diff_img": np.clip(diff * 20, 0, 255).astype(np.uint8),  # wzmocniona różnica
    }


# --------------------------------------------------------------------------- #
#  GUI (tkinter)
# --------------------------------------------------------------------------- #
def run_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        from PIL import ImageTk
    except ImportError:
        print("Brak tkintera (lub Pillow.ImageTk) - uruchamiam interfejs w przeglądarce.\n"
              "  (tkinter doinstalujesz: Debian/Ubuntu: sudo apt install python3-tk; "
              "Fedora: sudo dnf install python3-tkinter; macOS: brew install python-tk)")
        run_web()
        return

    IMG_TYPES = [("Obrazy", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"), ("Wszystkie pliki", "*.*")]

    class App(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Steganografia LSB - 3. i 4. bit")
            self.geometry("940x700")
            self.minsize(820, 600)

            self.cover: np.ndarray | None = None     # oryginał
            self.marked: np.ndarray | None = None    # obraz ze znakiem wodnym
            self._photos: dict[str, ImageTk.PhotoImage] = {}

            self.var_y = tk.StringVar(value=str(DEFAULT_Y))
            self.var_x = tk.StringVar(value=str(DEFAULT_X))
            self.status = tk.StringVar(value="Gotowy.")

            self._build_header()
            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True, padx=8, pady=4)
            self._build_embed_tab(nb)
            self._build_extract_tab(nb)
            self._build_compare_tab(nb)
            ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(
                fill="x", side="bottom")

        # ---------------- wspólne ---------------- #
        def _build_header(self) -> None:
            f = ttk.LabelFrame(self, text="Współrzędne startowe (klucz - muszą być takie same przy odczycie)")
            f.pack(fill="x", padx=8, pady=6)
            ttk.Label(f, text="y (wiersz):").pack(side="left", padx=(8, 2), pady=6)
            ttk.Spinbox(f, from_=1, to=100000, width=7, textvariable=self.var_y).pack(side="left")
            ttk.Label(f, text="x (kolumna):").pack(side="left", padx=(16, 2))
            ttk.Spinbox(f, from_=1, to=100000, width=7, textvariable=self.var_x).pack(side="left")
            ttk.Label(f, text="(liczone od 1; w artykule y=200, x=1)").pack(side="left", padx=12)

        def coords(self) -> tuple[int, int]:
            try:
                return int(self.var_y.get()), int(self.var_x.get())
            except ValueError:
                raise StegoError("Współrzędne muszą być liczbami całkowitymi.")

        def show(self, key: str, label: ttk.Label, arr: np.ndarray, size=(380, 380)) -> None:
            im = Image.fromarray(arr)
            im.thumbnail(size)
            self._photos[key] = ImageTk.PhotoImage(im)
            label.configure(image=self._photos[key], text="")

        def ask_open(self) -> np.ndarray | None:
            path = filedialog.askopenfilename(filetypes=IMG_TYPES)
            if not path:
                return None
            try:
                arr = load_gray(path)
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Błąd", f"Nie można wczytać obrazu:\n{e}")
                return None
            self.status.set(f"Wczytano {Path(path).name} ({arr.shape[1]}x{arr.shape[0]}, skala szarości)")
            return arr

        # ---------------- zakładka: osadzanie ---------------- #
        def _build_embed_tab(self, nb: ttk.Notebook) -> None:
            t = ttk.Frame(nb)
            nb.add(t, text="Osadzanie")
            top = ttk.Frame(t)
            top.pack(fill="x", padx=8, pady=6)
            ttk.Button(top, text="1. Wczytaj obraz (okładka)…", command=self.on_load_cover).pack(side="left")

            ttk.Label(t, text="2. Wpisz tajną wiadomość:").pack(anchor="w", padx=8)
            self.txt_msg = tk.Text(t, height=6, wrap="word")
            self.txt_msg.pack(fill="x", padx=8)
            self.txt_msg.bind("<KeyRelease>", lambda _e: self.update_counter())
            self.lbl_count = ttk.Label(t, text=f"0 / {MAX_LEN} bajtów")
            self.lbl_count.pack(anchor="e", padx=8)

            ttk.Button(t, text="3. Osadź i zapisz jako…", command=self.on_embed).pack(anchor="w", padx=8, pady=4)

            pv = ttk.Frame(t)
            pv.pack(fill="both", expand=True, padx=8, pady=4)
            self.lbl_cover = ttk.Label(pv, text="(oryginał)", anchor="center")
            self.lbl_cover.pack(side="left", expand=True)
            self.lbl_result = ttk.Label(pv, text="(obraz ze znakiem wodnym)", anchor="center")
            self.lbl_result.pack(side="left", expand=True)

        def update_counter(self) -> None:
            n = len(self.txt_msg.get("1.0", "end-1c").encode("utf-8"))
            self.lbl_count.configure(
                text=f"{n} / {MAX_LEN} bajtów" + ("   ⚠ za długa!" if n > MAX_LEN else ""))

        def on_load_cover(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.cover, self.marked = arr, None
                self.show("cover", self.lbl_cover, arr)
                self.lbl_result.configure(image="", text="(obraz ze znakiem wodnym)")

        def on_embed(self) -> None:
            try:
                if self.cover is None:
                    raise StegoError("Najpierw wczytaj obraz.")
                y, x = self.coords()
                msg = self.txt_msg.get("1.0", "end-1c")
                result = embed(self.cover, msg, y, x)
            except StegoError as e:
                messagebox.showwarning("Uwaga", str(e))
                return
            self.marked = result
            self.show("result", self.lbl_result, result)
            stats = compare(self.cover, result)
            path = filedialog.asksaveasfilename(
                defaultextension=".bmp", initialfile="watermarked_image.bmp",
                filetypes=[("BMP", "*.bmp"), ("PNG", "*.png")])
            if path:
                try:
                    save_image(result, path)
                except Exception as e:  # noqa: BLE001
                    messagebox.showerror("Błąd zapisu", str(e))
                    return
                self.status.set(f"Zapisano {Path(path).name}  |  PSNR = {stats['psnr']:.2f} dB, "
                                f"zmienionych pikseli: {stats['changed']}")
            else:
                self.status.set("Osadzono (nie zapisano pliku). "
                                f"PSNR = {stats['psnr']:.2f} dB, zmienionych pikseli: {stats['changed']}")

        # ---------------- zakładka: wyodrębnianie ---------------- #
        def _build_extract_tab(self, nb: ttk.Notebook) -> None:
            t = ttk.Frame(nb)
            nb.add(t, text="Wyodrębnianie")
            top = ttk.Frame(t)
            top.pack(fill="x", padx=8, pady=6)
            ttk.Button(top, text="Wczytaj obraz ze znakiem wodnym…", command=self.on_load_marked).pack(side="left")
            ttk.Button(top, text="Użyj ostatnio wygenerowanego", command=self.on_use_last).pack(side="left", padx=6)
            ttk.Button(top, text="Wyodrębnij wiadomość", command=self.on_extract).pack(side="left", padx=6)

            self.lbl_marked = ttk.Label(t, text="(obraz)", anchor="center")
            self.lbl_marked.pack(side="left", expand=True, padx=8)

            right = ttk.Frame(t)
            right.pack(side="left", fill="both", expand=True, padx=8, pady=4)
            ttk.Label(right, text="Odczytana wiadomość:").pack(anchor="w")
            self.txt_out = tk.Text(right, wrap="word", state="disabled")
            self.txt_out.pack(fill="both", expand=True)

        def on_load_marked(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.marked = arr
                self.show("marked", self.lbl_marked, arr, (400, 400))

        def on_use_last(self) -> None:
            if self.marked is None:
                messagebox.showinfo("Info", "Nie ma jeszcze wygenerowanego obrazu.")
                return
            self.show("marked", self.lbl_marked, self.marked, (400, 400))

        def on_extract(self) -> None:
            try:
                if self.marked is None:
                    raise StegoError("Najpierw wczytaj obraz ze znakiem wodnym.")
                y, x = self.coords()
                text = extract(self.marked, y, x)
            except StegoError as e:
                messagebox.showwarning("Uwaga", str(e))
                return
            self.txt_out.configure(state="normal")
            self.txt_out.delete("1.0", "end")
            self.txt_out.insert("1.0", text)
            self.txt_out.configure(state="disabled")
            self.status.set(f"Odczytano {len(text.encode('utf-8'))} bajtów.")

        # ---------------- zakładka: porównanie ---------------- #
        def _build_compare_tab(self, nb: ttk.Notebook) -> None:
            t = ttk.Frame(nb)
            nb.add(t, text="Porównanie")
            top = ttk.Frame(t)
            top.pack(fill="x", padx=8, pady=6)
            ttk.Button(top, text="Wczytaj oryginał…", command=self.on_cmp_cover).pack(side="left")
            ttk.Button(top, text="Wczytaj obraz ze znakiem wodnym…", command=self.on_cmp_marked).pack(side="left", padx=6)
            ttk.Button(top, text="Porównaj (odejmij)", command=self.on_compare).pack(side="left", padx=6)
            self.lbl_stats = ttk.Label(t, text="")
            self.lbl_stats.pack(anchor="w", padx=8)
            self.lbl_diff = ttk.Label(t, text="(obraz różnicy, wzmocniony 20×)", anchor="center")
            self.lbl_diff.pack(fill="both", expand=True, padx=8, pady=4)

        def on_cmp_cover(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.cover = arr

        def on_cmp_marked(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.marked = arr

        def on_compare(self) -> None:
            try:
                if self.cover is None or self.marked is None:
                    raise StegoError("Potrzebny jest oryginał i obraz ze znakiem wodnym.")
                s = compare(self.cover, self.marked)
            except StegoError as e:
                messagebox.showwarning("Uwaga", str(e))
                return
            psnr = "∞" if s["psnr"] == float("inf") else f"{s['psnr']:.2f} dB"
            self.lbl_stats.configure(
                text=f"Zmienionych pikseli: {s['changed']}   |   max różnica: {s['max_diff']}   |   "
                     f"MSE: {s['mse']:.4f}   |   PSNR: {psnr}")
            self.show("diff", self.lbl_diff, s["diff_img"], (520, 520))

    App().mainloop()


# --------------------------------------------------------------------------- #
#  Interfejs w przeglądarce (tylko biblioteka standardowa, serwer lokalny)
# --------------------------------------------------------------------------- #
WEB_PAGE = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steganografia LSB - 3. i 4. bit</title>
<style>
  :root{--bg:#fafafa;--fg:#222;--card:#fff;--bd:#d0d0d0;--ac:#2563eb}
  @media (prefers-color-scheme:dark){:root{--bg:#1b1b1d;--fg:#e8e8e8;--card:#26262a;--bd:#444;--ac:#6b9bff}}
  body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:18px auto;padding:0 12px}
  h1{font-size:1.3rem}
  .card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;margin:10px 0}
  .tabs button{padding:8px 14px;border:1px solid var(--bd);background:var(--card);color:var(--fg);cursor:pointer;border-radius:6px 6px 0 0}
  .tabs button.on{background:var(--ac);color:#fff;border-color:var(--ac)}
  .tab{display:none}.tab.on{display:block}
  textarea{width:100%;box-sizing:border-box;height:120px;background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:6px}
  input[type=number]{width:80px}
  button.act{padding:7px 14px;background:var(--ac);color:#fff;border:0;border-radius:6px;cursor:pointer;margin:6px 6px 6px 0}
  .row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:260px;text-align:center}
  img{max-width:100%;max-height:380px;border:1px solid var(--bd);image-rendering:pixelated}
  .err{color:#d33;font-weight:600}.ok{color:#2a8a3a}.small{font-size:.85rem;opacity:.8}
  a.dl{display:inline-block;margin:6px 0;color:var(--ac)}
</style></head><body>
<h1>Steganografia LSB - 3. i 4. bit</h1>
<div class="card">Współrzędne startowe (klucz - takie same przy osadzaniu i odczycie):
  y <input id="y" type="number" min="1" value="200"> &nbsp; x <input id="x" type="number" min="1" value="1">
  <span class="small">(liczone od 1; w artykule y=200, x=1)</span></div>
<div class="tabs"><button data-t="t1" class="on">Osadzanie</button><button data-t="t2">Wyodrębnianie</button><button data-t="t3">Porównanie</button></div>
<div id="msg" class="card small">Gotowy.</div>

<div id="t1" class="tab on card">
  <p>1. Wybierz obraz (okładka): <input type="file" id="f1" accept="image/*"></p>
  <p>2. Tajna wiadomość: <span id="cnt" class="small">0 / 1023 bajtów</span></p>
  <textarea id="m"></textarea>
  <p>Format wyjściowy: <select id="fmt"><option>BMP</option><option>PNG</option></select>
  <button class="act" id="go1">3. Osadź</button></p>
  <a id="dl" class="dl" style="display:none">⬇ Pobierz obraz ze znakiem wodnym</a>
  <div class="row"><div><div>Oryginał</div><img id="i1" alt=""></div><div><div>Ze znakiem wodnym</div><img id="i2" alt=""></div></div>
</div>

<div id="t2" class="tab card">
  <p>Obraz ze znakiem wodnym: <input type="file" id="f2" accept="image/*">
  <button class="act" id="use">Użyj ostatnio wygenerowanego</button></p>
  <button class="act" id="go2">Wyodrębnij wiadomość</button>
  <textarea id="out" readonly placeholder="Tu pojawi się odczytana wiadomość"></textarea>
  <div style="text-align:center"><img id="i3" alt=""></div>
</div>

<div id="t3" class="tab card">
  <p>Oryginał: <input type="file" id="f3a" accept="image/*"></p>
  <p>Ze znakiem wodnym: <input type="file" id="f3b" accept="image/*"></p>
  <button class="act" id="go3">Porównaj (odejmij)</button>
  <p id="stats"></p>
  <div style="text-align:center"><div class="small">Różnica wzmocniona 20×</div><img id="i4" alt=""></dlbl_cover, arr)
                self.lbl_result.configure(image="", text="(obraz ze znakiem wodnym)")

        def on_embed(self) -> None:
            try:
                if self.cover is None:
                    raise StegoError("Najpierw wczytaj obraz.")
                y, x = self.coords()
                msg = self.txt_msg.get("1.0", "end-1c")
                result = embed(self.cover, msg, y, x)
            except StegoError as e:
                messagebox.showwarning("Uwaga", str(e))
                return
            self.marked = result
            self.show("result", self.lbl_result, result)
            stats = compare(self.cover, result)
            path = filedialog.asksaveasfilename(
                defaultextension=".bmp", initialfile="watermarked_image.bmp",
                filetypes=[("BMP", "*.bmp"), ("PNG", "*.png")])
            if path:
                try:
                    save_image(result, path)
                except Exception as e:  # noqa: BLE001
                    messagebox.showerror("Błąd zapisu", str(e))
                    return
                self.status.set(f"Zapisano {Path(path).name}  |  PSNR = {stats['psnr']:.2f} dB, "
                                f"zmienionych pikseli: {stats['changed']}")
            else:
                self.status.set("Osadzono (nie zapisano pliku). "
                                f"PSNR = {stats['psnr']:.2f} dB, zmienionych pikseli: {stats['changed']}")

        # ---------------- zakładka: wyodrębnianie ---------------- #
        def _build_extract_tab(self, nb: ttk.Notebook) -> None:
            t = ttk.Frame(nb)
            nb.add(t, text="Wyodrębnianie")
            top = ttk.Frame(t)
            top.pack(fill="x", padx=8, pady=6)
            ttk.Button(top, text="Wczytaj obraz ze znakiem wodnym…", command=self.on_load_marked).pack(side="left")
            ttk.Button(top, text="Użyj ostatnio wygenerowanego", command=self.on_use_last).pack(side="left", padx=6)
            ttk.Button(top, text="Wyodrębnij wiadomość", command=self.on_extract).pack(side="left", padx=6)

            self.lbl_marked = ttk.Label(t, text="(obraz)", anchor="center")
            self.lbl_marked.pack(side="left", expand=True, padx=8)

            right = ttk.Frame(t)
            right.pack(side="left", fill="both", expand=True, padx=8, pady=4)
            ttk.Label(right, text="Odczytana wiadomość:").pack(anchor="w")
            self.txt_out = tk.Text(right, wrap="word", state="disabled")
            self.txt_out.pack(fill="both", expand=True)

        def on_load_marked(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.marked = arr
                self.show("marked", self.lbl_marked, arr, (400, 400))

        def on_use_last(self) -> None:
            if self.marked is None:
                messagebox.showinfo("Info", "Nie ma jeszcze wygenerowanego obrazu.")
                return
            self.show("marked", self.lbl_marked, self.marked, (400, 400))

        def on_extract(self) -> None:
            try:
                if self.marked is None:
                    raise StegoError("Najpierw wczytaj obraz ze znakiem wodnym.")
                y, x = self.coords()
                text = extract(self.marked, y, x)
            except StegoError as e:
                messagebox.showwarning("Uwaga", str(e))
                return
            self.txt_out.configure(state="normal")
            self.txt_out.delete("1.0", "end")
            self.txt_out.insert("1.0", text)
            self.txt_out.configure(state="disabled")
            self.status.set(f"Odczytano {len(text.encode('utf-8'))} bajtów.")

        # ---------------- zakładka: porównanie ---------------- #
        def _build_compare_tab(self, nb: ttk.Notebook) -> None:
            t = ttk.Frame(nb)
            nb.add(t, text="Porównanie")
            top = ttk.Frame(t)
            top.pack(fill="x", padx=8, pady=6)
            ttk.Button(top, text="Wczytaj oryginał…", command=self.on_cmp_cover).pack(side="left")
            ttk.Button(top, text="Wczytaj obraz ze znakiem wodnym…", command=self.on_cmp_marked).pack(side="left", padx=6)
            ttk.Button(top, text="Porównaj (odejmij)", command=self.on_compare).pack(side="left", padx=6)
            self.lbl_stats = ttk.Label(t, text="")
            self.lbl_stats.pack(anchor="w", padx=8)
            self.lbl_diff = ttk.Label(t, text="(obraz różnicy, wzmocniony 20×)", anchor="center")
            self.lbl_diff.pack(fill="both", expand=True, padx=8, pady=4)

        def on_cmp_cover(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.cover = arr

        def on_cmp_marked(self) -> None:
            arr = self.ask_open()
            if arr is not None:
                self.marked = arr

        def on_compare(self) -> None:
            try:
                if self.cover is None or self.marked is None:
                    raise StegoError("Potrzebny jest oryginał i obraz ze znakiem wodnym.")
                s = compare(self.cover, self.marked)
            except StegoError as e:
                messagebox.showwarning("Uwaga", str(e))
                return
            psnr = "∞" if s["psnr"] == float("inf") else f"{s['psnr']:.2f} dB"
            self.lbl_stats.configure(
                text=f"Zmienionych pikseli: {s['changed']}   |   max różnica: {s['max_diff']}   |   "
                     f"MSE: {s['mse']:.4f}   |   PSNR: {psnr}")
            self.show("diff", self.lbl_diff, s["diff_img"], (520, 520))

    App().mainloop()


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Steganografia LSB (3. i 4. bit)")
    sub = p.add_subparsers(dest="cmd")

    def add_coords(sp):
        sp.add_argument("-y", type=int, default=DEFAULT_Y, help="wiersz startowy (od 1), domyślnie 200")
        sp.add_argument("-x", type=int, default=DEFAULT_X, help="kolumna startowa (od 1), domyślnie 1")

    e = sub.add_parser("embed", help="ukryj wiadomość")
    e.add_argument("input")
    e.add_argument("output", help="plik wyjściowy .bmp lub .png")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("-m", "--message")
    g.add_argument("-f", "--file", help="plik tekstowy UTF-8 z wiadomością")
    add_coords(e)

    x = sub.add_parser("extract", help="odczytaj wiadomość")
    x.add_argument("input")
    add_coords(x)

    c = sub.add_parser("compare", help="porównaj oryginał z obrazem ze znakiem wodnym")
    c.add_argument("original")
    c.add_argument("marked")

    sub.add_parser("gui", help="uruchom interfejs graficzny")

    args = p.parse_args(argv)
    try:
        if args.cmd in (None, "gui"):
            run_gui()
        elif args.cmd == "embed":
            msg = args.message if args.message is not None else Path(args.file).read_text(encoding="utf-8")
            out = embed(load_gray(args.input), msg, args.y, args.x)
            save_image(out, args.output)
            print(f"Zapisano {args.output}")
        elif args.cmd == "extract":
            print(extract(load_gray(args.input), args.y, args.x))
        elif args.cmd == "compare":
            s = compare(load_gray(args.original), load_gray(args.marked))
            print(f"zmienionych pikseli: {s['changed']}, max różnica: {s['max_diff']}, "
                  f"MSE: {s['mse']:.4f}, PSNR: {s['psnr']:.2f} dB")
    except StegoError as err:
        print(f"Błąd: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
