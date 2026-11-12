"""Common utilities: paths, logging, Blender image reading, dependency lookup.

This module does not depend on onnxruntime / cv2, so it can always be imported safely.
It also deliberately avoids a hard dependency on bpy: standalone tools
(pose_to_controlnet) reuse this module directly, and in that case there is no
Blender and bpy is None.
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    import bpy
except ImportError:  # standalone tool environment (no Blender)
    bpy = None  # type: ignore[assignment]

ADDON_ID = "pose_capture"
LOG_PREFIX = "[PoseCapture]"


def has_blender() -> bool:
    """Whether we are running inside Blender (False in standalone tools)."""
    return bpy is not None


def require_blender(feature: str) -> None:
    if bpy is None:
        raise RuntimeError("%s needs to run inside Blender (this is a standalone environment)" % feature)


# --------------------------------------------------------------------------- logging
def log(*args) -> None:
    print(LOG_PREFIX, *args)


def warn(*args) -> None:
    print(LOG_PREFIX, "WARNING:", *args)


def error(*args) -> None:
    print(LOG_PREFIX, "ERROR:", *args)


# --------------------------------------------------------------------------- paths
def user_data_dir(*subdirs: str) -> str:
    """Return the add-on data directory (kept in Blender's user data directory so the extension folder stays clean).

    Without Blender, falls back to the POSE_CAPTURE_DATA environment variable, otherwise %LOCALAPPDATA%/PoseCapture.
    """
    if bpy is None:
        root = os.environ.get("POSE_CAPTURE_DATA") or os.path.join(
            os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "PoseCapture")
        path = os.path.join(root, *subdirs) if subdirs else root
        os.makedirs(path, exist_ok=True)
        return path

    rel = os.path.join(ADDON_ID, *subdirs) if subdirs else ADDON_ID
    path = bpy.utils.user_resource("DATAFILES", path=rel, create=True)
    return str(path)


def models_dir() -> str:
    path = user_data_dir("models")
    os.makedirs(path, exist_ok=True)
    return path


def libs_dir() -> str:
    path = user_data_dir("libs")
    os.makedirs(path, exist_ok=True)
    return path


def ensure_libs_on_path() -> str | None:
    """Append the pip install directory to the END of sys.path (so Blender's bundled numpy keeps priority).

    In standalone tools (no Blender) this returns None instead of raising: the dependency
    directory is managed by the tool itself (pose_to_controlnet/pose_to_cn/paths.py), and
    callers here (e.g. when dwpose creates a backend) must not fail just because Blender is missing.
    """
    if bpy is None:
        return None
    path = libs_dir()
    if path not in sys.path:
        sys.path.append(path)
    return path


def python_executable() -> str | None:
    """Locate the Python interpreter bundled with Blender, used to run pip.

    In a standalone environment this simply returns the current interpreter (sys.executable).
    """
    if bpy is None:
        return sys.executable

    # Since Blender 2.91 sys.executable is Blender's own bundled Python interpreter - the most reliable answer.
    exe = sys.executable or ""
    if exe and os.path.isfile(exe) and os.path.basename(exe).lower().startswith("python"):
        return exe

    candidates: list[str] = []

    binary = getattr(bpy.app, "binary_path", "") or ""
    if binary:
        blender_dir = os.path.dirname(os.path.abspath(binary))
        version = "%d.%d" % (bpy.app.version[0], bpy.app.version[1])
        for name in ("python.exe", "python"):
            candidates.append(os.path.join(blender_dir, version, "python", "bin", name))
        # Linux / macOS layout
        candidates.append(os.path.join(blender_dir, version, "python", "bin", "python3"))
        for name in ("python3.13", "python3.12", "python3.11"):
            candidates.append(os.path.join(blender_dir, version, "python", "bin", name))
        # macOS app bundle: Blender.app/Contents/MacOS/Blender -> Contents/Resources/<version>/python/bin
        resources = os.path.join(os.path.dirname(blender_dir), "Resources")
        for name in ("python3.13", "python3.12", "python3.11", "python3"):
            candidates.append(os.path.join(resources, version, "python", "bin", name))

    legacy = getattr(bpy.app, "binary_path_python", None)
    if legacy:
        candidates.append(legacy)

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


# --------------------------------------------------------------------------- images
def image_datablock_to_rgb8(image: "bpy.types.Image") -> np.ndarray:
    """Read a Blender image as an (H, W, 3) uint8 array (row 0 = top of the image).

    The colour space is set to Non-Color before reading so we get the raw sRGB values from the file,
    matching OpenCV's imread result (which is what the models were trained / exported with).
    """
    require_blender("Reading a Blender image (image_datablock_to_rgb8)")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError("Image size is 0, cannot read it: %s" % image.name)

    changed_space = False
    original_space = "sRGB"
    try:
        if image.colorspace_settings.name != "Non-Color":
            original_space = image.colorspace_settings.name   # restore the user's own colour space afterwards
            image.colorspace_settings.name = "Non-Color"
            changed_space = True
    except Exception:  # some generated images have no colour space settings
        pass

    width, height = int(image.size[0]), int(image.size[1])
    buf = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)

    if changed_space:
        try:
            image.colorspace_settings.name = original_space
        except Exception:
            pass

    pixels = buf.reshape(height, width, 4)[::-1]  # Blender stores rows bottom-to-top, so flip
    rgb = np.clip(pixels[:, :, :3] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def load_image_rgb8(filepath: str) -> np.ndarray:
    """Read an image from disk and return (H, W, 3) uint8 (uses Blender's own loader, no PIL needed)."""
    require_blender("Reading an image file (load_image_rgb8)")

    if not os.path.isfile(filepath):
        raise FileNotFoundError("Image file not found: %s" % filepath)

    image = bpy.data.images.load(filepath, check_existing=False)
    try:
        return image_datablock_to_rgb8(image)
    finally:
        try:
            bpy.data.images.remove(image)
        except Exception:
            pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
