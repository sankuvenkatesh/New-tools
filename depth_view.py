"""Depth preview window (a separate Tk window).

Why a separate window: the preview in Blender's Image Editor is affected by "pixel writes / colour space"
and sometimes shows up completely black, making it hard to tell whether the model output or the display is wrong. This window directly reads the
PNG files + ``stats.json`` on disk (computed by the add-on inside Blender) and shows

    [input photo]  [depth grayscale]  [depth false colour]

side by side, with numeric statistics underneath (min/max/mean/std/percentiles, whether it is constant, whether it is nearly all black)
and a **depth direction check** (the physical criterion "the person is closer than the background" vs the direction from the reconstruction fit; if they disagree, front and back are reversed).

Colour convention (the same across the whole add-on): grayscale **whiter = farther**, false colour **near = blue, far = red**.

Usage (no Blender needed, only a Python with tkinter):

    python depth_view.py --dir <preview folder>
    python depth_view.py --dir <preview folder> --columns 3 --max-size 520

The preview folder is produced by the add-on's "Depth Preview Window" button (the Blender side runs the model and writes the PNGs).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

MANIFEST = "stats.json"
_PYTHON_CACHE: dict = {}


# --------------------------------------------------------------------------- finding an interpreter
def _can_import_tkinter(executable: str) -> bool:
    """Run a candidate interpreter and see whether it can import tkinter."""
    try:
        result = subprocess.run([executable, "-c", "import tkinter"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=20,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def candidate_pythons() -> list[str]:
    """Possible interpreters: current process > PATH > py launcher > common install locations."""
    import shutil

    out: list[str] = []
    if sys.executable and sys.executable.lower().endswith(("python.exe", "pythonw.exe")):
        out.append(sys.executable)
    for name in ("pythonw.exe", "python.exe", "python3", "python"):
        found = shutil.which(name)
        if found:
            out.append(found)
    for name in ("py.exe", "py"):
        found = shutil.which(name)
        if found:
            out.append(found)
    roots = [r"C:\Python3", os.path.join(os.environ.get("LOCALAPPDATA") or "", "Programs",
                                         "Python")]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root), reverse=True):
            folder = os.path.join(root, entry)
            if os.path.isdir(folder):
                for name in ("pythonw.exe", "python.exe"):
                    path = os.path.join(folder, name)
                    if os.path.isfile(path):
                        out.append(path)
    seen = set()
    unique = []
    for path in out:
        key = os.path.abspath(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_python(preferred: str = "") -> str | None:
    """Find an interpreter that can run tkinter; returns None if none is found (results are cached per candidate path)."""
    if preferred:
        cached = _PYTHON_CACHE.get(("pref", preferred))
        if cached is None:
            cached = bool(os.path.isfile(preferred) and _can_import_tkinter(preferred))
            _PYTHON_CACHE[("pref", preferred)] = cached
        if cached:
            return preferred
        return None
    if "auto" in _PYTHON_CACHE:
        return _PYTHON_CACHE["auto"]
    found = None
    for path in candidate_pythons():
        cached = _PYTHON_CACHE.get(path)
        if cached is None:
            cached = bool(os.path.isfile(path) and _can_import_tkinter(path))
            _PYTHON_CACHE[path] = cached
        if cached:
            found = path
            break
    _PYTHON_CACHE["auto"] = found
    return found


# --------------------------------------------------------------------------- reading data
def load_manifest(directory: str) -> dict:
    path = os.path.join(directory, MANIFEST)
    if not os.path.isfile(path):
        raise FileNotFoundError("No %s in the preview folder: %s" % (MANIFEST, path))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_stats(manifest: dict) -> str:
    """Write the statistics of every image in stats.json as multi-line text."""
    lines = [manifest.get("title") or "", ""]
    for panel in manifest.get("panels", []):
        stats = panel.get("stats") or {}
        lines.append("[%s] %s" % (panel.get("label", "?"), panel.get("caption", "")))
        if stats:
            lines.append("    min=%.4f max=%.4f mean=%.4f std=%.4f"
                         % (stats.get("min", 0.0), stats.get("max", 0.0),
                            stats.get("mean", 0.0), stats.get("std", 0.0)))
            if stats.get("percentiles"):
                lines.append("    Percentiles 1/2/50/98/99 = %s"
                             % ", ".join("%.4f" % v for v in stats["percentiles"]))
            if stats.get("mean255") is not None:
                lines.append("    Display channel mean (0-255) = %.2f" % stats["mean255"])
        for note in panel.get("notes") or []:
            lines.append("    * %s" % note)
        lines.append("")

    direction = manifest.get("direction") or {}
    if direction.get("text"):
        lines.append("Direction check: %s" % direction["text"])
        if direction.get("fit") is None:
            lines.append("    (\"Apply to Armature\" has not been clicked yet, so only the physical criterion is available)")
        elif not direction.get("agree"):
            lines.append("    x The fitted direction is the opposite of \"the person is closer than the background\": front and back of the whole pose may be reversed, "
                         "tick \"Flip Direction\" and solve again")
        lines.append("")

    for note in manifest.get("notes") or []:
        lines.append("!! %s" % note)
    return "\n".join(lines)


# --------------------------------------------------------------------------- user interface
def _make_photo(tk_module, path: str, max_size: tuple[int, int]):
    """Return (PhotoImage, display size). Smooth scaling with PIL if available, otherwise integer scaling."""
    try:
        from PIL import Image, ImageTk

        image = Image.open(path)
        image.thumbnail(max_size)
        return ImageTk.PhotoImage(image), image.size
    except Exception:  # noqa: BLE001
        photo = tk_module.PhotoImage(file=path)
        factor = max(1, int(math.ceil(photo.width() / float(max_size[0]))),
                     int(math.ceil(photo.height() / float(max_size[1]))))
        if factor > 1:
            photo = photo.subsample(factor, factor)
        return photo, (photo.width(), photo.height())


def _open_folder(directory: str) -> None:
    try:
        if hasattr(os, "startfile"):
            os.startfile(directory)  # noqa: S606  (Windows)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", directory])
        else:
            subprocess.Popen(["xdg-open", directory])
    except Exception as exc:  # noqa: BLE001
        print("Cannot open the folder: %s" % exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sanku AI Pose Studio depth preview window")
    parser.add_argument("--dir", default="", help="preview folder (containing stats.json and the PNGs)")
    parser.add_argument("--max-size", type=int, default=520, help="maximum display edge length of a single image")
    parser.add_argument("--title", default="Sanku AI Pose Studio - Depth Preview")
    parser.add_argument("--check", action="store_true",
                        help="build the UI once to verify the data can be shown, without entering the main loop (self-test)")
    args, _unknown = parser.parse_known_args(argv)
    directory = args.dir or os.path.dirname(os.path.abspath(__file__))

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # noqa: BLE001
        print("This Python has no tkinter, cannot open the preview window: %s" % exc)
        return 1

    root = tk.Tk()
    root.title("%s - %s" % (args.title, directory))
    root.minsize(720, 520)

    header = tk.Label(root, text=directory, anchor="w", font=("Segoe UI", 9))
    header.pack(fill="x", padx=8, pady=(6, 0))

    body = tk.Frame(root)
    body.pack(fill="both", expand=True, padx=8, pady=6)
    info = tk.Text(root, height=15, wrap="none", font=("Consolas", 9))
    info.pack(fill="both", expand=False, padx=8, pady=(0, 6))

    state: dict = {"photos": [], "canvases": []}

    def render() -> None:
        manifest = load_manifest(directory)
        root.title("%s - %s" % (manifest.get("title") or args.title,
                                os.path.basename(directory)))
        for widget in body.winfo_children():
            widget.destroy()
        state["photos"].clear()
        max_size = (args.max_size, args.max_size)
        for column, panel in enumerate(manifest.get("panels", [])):
            cell = tk.Frame(body)
            cell.grid(row=0, column=column, padx=6, sticky="n")
            tk.Label(cell, text=panel.get("label", "?"),
                     font=("Segoe UI", 10, "bold")).pack()
            caption = panel.get("caption") or ""
            canvas = tk.Canvas(cell, width=max_size[0], height=max_size[1],
                               background="#101010", highlightthickness=1,
                               highlightbackground="#404040")
            canvas.pack()
            path = panel.get("path") or ""
            if path and os.path.isfile(path):
                try:
                    photo, size = _make_photo(tk, path, max_size)
                except Exception as exc:  # noqa: BLE001
                    canvas.create_text(10, 10, anchor="nw", fill="#ff8888",
                                       text="Failed to read image: %s" % exc)
                    photo, size = None, (0, 0)
                if photo is not None:
                    state["photos"].append(photo)
                    canvas.create_image(size[0] // 2, size[1] // 2, image=photo)
            else:
                canvas.create_text(10, 10, anchor="nw", fill="#888888",
                                   text="Missing file: %s" % path)
            tk.Label(cell, text=caption, wraplength=max_size[0],
                     justify="left", font=("Segoe UI", 8),
                     foreground="#bbbbbb").pack()
        info.delete("1.0", "end")
        info.insert("1.0", format_stats(manifest))

    def reload_all() -> None:
        try:
            render()
        except Exception as exc:  # noqa: BLE001
            info.delete("1.0", "end")
            info.insert("1.0", "Read failed: %s" % exc)

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(buttons, text="Reload", command=reload_all).pack(side="left")
    ttk.Button(buttons, text="Open Folder", command=lambda: _open_folder(directory)).pack(
        side="left", padx=6)
    ttk.Button(buttons, text="Close", command=root.destroy).pack(side="right")
    root.bind("<Escape>", lambda _event: root.destroy())

    reload_all()
    if args.check:
        root.update_idletasks()
        print("UI self-test: %d panels, %d images loaded successfully"
              % (len(body.winfo_children()), len(state["photos"])))
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
