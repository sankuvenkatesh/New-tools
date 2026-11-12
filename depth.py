"""Depth model (Depth Anything V2 Small, ONNX): adds a real depth to DWPose's 2D keypoints.

Why it is needed: in a single picture, "the elbow is in front of the body" and "the elbow is behind the body" project
to exactly the same image, so solving from bone lengths can only guess (the old version used heuristic switches such as
elbow bends backwards / knee bends forwards / flip the left arm front-to-back). With a depth map, the depth of every
joint is read directly from the pixels and the front/back direction no longer has to be guessed.

Preprocessing follows HuggingFace's DPTImageProcessor:
  * aspect-preserving resize to fit within 518x518, with both sides a multiple of 14 (no padding)
  * ImageNet mean/std normalisation, NCHW float32
The output has the same resolution as the input and holds **relative depth** (affine invariant: true depth = alpha*d + beta).
alpha/beta are obtained by reconstruction.py with one global least-squares fit over all bone segments of the whole image,
not by solving bone lengths joint by joint, so the "elbow forward or backward" ambiguity does not exist.

This module only does "model -> depth map -> per-pixel sampling"; it knows nothing about the skeleton and can be tested on its own.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import dwpose, imops, utils

# --------------------------------------------------------------------------- model
INPUT_SIZE = 518          # the longer side is scaled to fit within this size
SIZE_MULTIPLE = 14        # ViT patch size: both sides must be a multiple of it
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_REMOTE_REPO = "onnx-community/depth-anything-v2-small"
_REMOTE_BASE = "https://huggingface.co/%s/resolve/main/" % _REMOTE_REPO
# huggingface.co cannot be reached on some networks (e.g. in China), so a mirror is listed after it as a fallback
_MIRROR_BASE = "https://hf-mirror.com/%s/resolve/main/" % _REMOTE_REPO

MODEL_VARIANTS = {
    "fp32": {"file": "depth_anything_v2_small.onnx",
             "remote": "onnx/model.onnx", "size_mb": 95,
             "label": "Standard precision (about 95 MB)"},
    "fp16": {"file": "depth_anything_v2_small_fp16.onnx",
             "remote": "onnx/model_fp16.onnx", "size_mb": 48,
             "label": "Half precision (about 48 MB)"},
    "int8": {"file": "depth_anything_v2_small_int8.onnx",
             "remote": "onnx/model_quantized.onnx", "size_mb": 27,
             "label": "Quantized (about 27 MB, fastest)"},
}
DEFAULT_VARIANT = "fp32"


def variant_ids() -> list[str]:
    return list(MODEL_VARIANTS.keys())


def variant_items() -> list[tuple[str, str, str]]:
    """For EnumProperty: (id, name, description)."""
    return [(key, key, item["label"]) for key, item in MODEL_VARIANTS.items()]


def variant_info(variant: str) -> dict:
    return MODEL_VARIANTS.get(variant) or MODEL_VARIANTS[DEFAULT_VARIANT]


def model_filename(variant: str) -> str:
    return variant_info(variant)["file"]


def model_urls(variant: str) -> list[str]:
    """Official address first, mirror second (deps.download_model tries them in order)."""
    remote = variant_info(variant)["remote"]
    return [_REMOTE_BASE + remote, _MIRROR_BASE + remote]


def looks_like_model(filename: str) -> bool:
    """Rough filter for local imports: onnx files with 'depth' in the name (Depth Anything / MiDaS both work)."""
    lower = os.path.basename(str(filename)).lower()
    return lower.endswith(".onnx") and "depth" in lower


# --------------------------------------------------------------------------- preprocessing
def target_size(width: int, height: int) -> tuple[int, int]:
    """Align with HF DPTImageProcessor: aspect-preserving resize + round up to a multiple of 14, no padding."""
    width, height = max(int(width), 1), max(int(height), 1)
    ratio = min(INPUT_SIZE / float(height), INPUT_SIZE / float(width))
    new_w = max(SIZE_MULTIPLE, int(round(width * ratio)))
    new_h = max(SIZE_MULTIPLE, int(round(height * ratio)))
    new_w = int(math.ceil(new_w / float(SIZE_MULTIPLE))) * SIZE_MULTIPLE
    new_h = int(math.ceil(new_h / float(SIZE_MULTIPLE))) * SIZE_MULTIPLE
    return new_w, new_h


def preprocess(rgb8: np.ndarray):
    """(H,W,3) uint8 -> (blob(1,3,h,w) float32, model input size (w,h))."""
    height, width = rgb8.shape[:2]
    new_w, new_h = target_size(width, height)
    resized = imops.resize(rgb8, new_w, new_h).astype(np.float32) / 255.0
    normalized = (resized - IMAGE_MEAN) / IMAGE_STD
    blob = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None],
                                dtype=np.float32)
    return blob, (new_w, new_h)


class DepthEstimator:
    """Load once, call many times; inference re-uses dwpose.create_backend (onnxruntime preferred)."""

    def __init__(self, path: str, backend: str = "auto", providers=None):
        if not os.path.isfile(path):
            raise dwpose.ModelNotFound("Depth model not found: %s" % path)
        self.path = path
        self.session = dwpose.create_backend(path, backend, providers)
        name, _shape, dtype = self.session.input_spec()
        self.input_name = name
        self.input_dtype = dtype
        utils.log("Depth model: %s (backend %s, device %s)"
                  % (os.path.basename(path), self.session.name,
                     getattr(self.session, "active_provider", "?")))

    def predict(self, rgb8: np.ndarray):
        """Return (depth map float32 (h,w), model input size (w,h))."""
        blob, size = preprocess(rgb8)
        if self.input_dtype != np.float32:
            blob = blob.astype(self.input_dtype)
        outputs = self.session.run({self.input_name: blob})
        values = np.asarray(outputs[0], dtype=np.float32)
        while values.ndim > 2:
            values = values[0]
        return np.ascontiguousarray(values), size


# --------------------------------------------------------------------------- sampling
def _sample_one(values: np.ndarray, x: float, y: float) -> float:
    sampled = imops._bilinear(values, np.array([x], dtype=np.float32),
                              np.array([y], dtype=np.float32))
    return float(np.asarray(sampled).reshape(-1)[0])


def sample_many(values: np.ndarray, model_size, image_size, points,
                window: int = 1) -> np.ndarray:
    """Sample the depth at original-image pixel coordinates; None is allowed in points (nan is returned for it).

    A 3x3 median is taken around each joint: at limb edges the depth map mixes in background depth, and the median is steadier than a single point.
    """
    scale_x = float(model_size[0]) / max(float(image_size[0]), 1e-6)
    scale_y = float(model_size[1]) / max(float(image_size[1]), 1e-6)
    out = np.full(len(points), np.nan, dtype=np.float64)
    for index, point in enumerate(points):
        if point is None:
            continue
        u = float(point[0]) * scale_x
        v = float(point[1]) * scale_y
        if window and window > 0:
            samples = [_sample_one(values, u + dx, v + dy)
                       for dx in (-window, 0, window) for dy in (-window, 0, window)]
            out[index] = float(np.median(samples))
        else:
            out[index] = _sample_one(values, u, v)
    return out


def nearness(values: np.ndarray, near_is_large: bool = True) -> np.ndarray:
    """Depth values -> normalised array of "how near" (0 = farthest, 1 = nearest).

    ``near_is_large=True`` means a large depth value = close to the camera (the Depth Anything convention).
    Both the false-colour and the grayscale previews take their direction from this one function, so the colours never contradict each other.
    """
    array = np.asarray(values, dtype=np.float32)
    low, high = float(np.min(array)), float(np.max(array))
    span = max(high - low, 1e-6)
    large = np.clip((array - low) / span, 0.0, 1.0)     # 1 = large value
    return large if near_is_large else 1.0 - large


def colorize(values: np.ndarray, near_is_large: bool = True) -> np.ndarray:
    """Depth map -> false colour: **near = blue, far = red** (white in between).

    near_is_large=False means "large depth value = far"; the whole colour ramp is then reversed so that the blue end is always
    the pixel closest to the camera and the red end always the farthest.
    """
    farness = 1.0 - nearness(values, near_is_large)     # 1 = farthest
    red = np.clip(2.0 * farness, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * farness - 1.0), 0.0, 1.0)
    blue = np.clip(2.0 - 2.0 * farness, 0.0, 1.0)
    return ((np.stack([red, green, blue], axis=-1)) * 255.0 + 0.5).astype(np.uint8)


def preview_image(rgb8: np.ndarray, values: np.ndarray,
                  near_is_large: bool = True) -> np.ndarray:
    """Scale the false-colour depth image back to the original size, so it is easy to compare with the photo."""
    height, width = rgb8.shape[:2]
    return imops.resize(colorize(values, near_is_large), width, height)