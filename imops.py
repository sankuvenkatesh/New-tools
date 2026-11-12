"""Pure-numpy image processing (no OpenCV dependency).

Contains: bilinear sampling, resize, letterbox, affine warp (the equivalent of cv2.warpAffine), NMS, skeleton preview drawing.
"""

from __future__ import annotations

import numpy as np

from . import coco


# --------------------------------------------------------------------------- sampling / resizing
def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sampling; x/y are float pixel coordinates (may be out of range, in which case they are clamped to the border)."""
    height, width = img.shape[:2]
    src = img.astype(np.float32, copy=False)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)[..., None]
    fy = (y - y0).astype(np.float32)[..., None]

    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)

    a = src[y0c, x0c]
    b = src[y0c, x1c]
    c = src[y1c, x0c]
    d = src[y1c, x1c]

    top = a + (b - a) * fx
    bottom = c + (d - c) * fx
    return top + (bottom - top) * fy


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize (equivalent to cv2.resize(..., INTER_LINEAR))."""
    height = int(height)
    width = int(width)
    src_h, src_w = img.shape[:2]

    scale_x = src_w / float(width)
    scale_y = src_h / float(height)
    xs = (np.arange(width, dtype=np.float32) + 0.5) * scale_x - 0.5
    ys = (np.arange(height, dtype=np.float32) + 0.5) * scale_y - 0.5
    grid_x = np.tile(xs, (height, 1))
    grid_y = np.tile(ys[:, None], (1, width))

    out = _bilinear(img, grid_x, grid_y)
    if img.dtype == np.uint8:
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out.astype(img.dtype, copy=False)


def letterbox(img: np.ndarray, width: int, height: int, pad_value: int = 114):
    """Aspect-preserving resize + padding on the right/bottom (YOLOX preprocessing). Returns (result image, scale ratio)."""
    height = int(height)
    width = int(width)
    src_h, src_w = img.shape[:2]

    padded = np.full((height, width, img.shape[2]), pad_value, dtype=img.dtype)
    ratio = min(height / src_h, width / src_w)
    new_w = max(1, int(src_w * ratio))
    new_h = max(1, int(src_h * ratio))
    resized = resize(img, new_w, new_h)
    padded[:new_h, :new_w] = resized
    return padded, ratio


# --------------------------------------------------------------------------- affine
def invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix (the result is also 2x3)."""
    a = matrix[:, :2]
    t = matrix[:, 2:3]
    inv_a = np.linalg.inv(a)
    return np.concatenate([inv_a, -inv_a @ t], axis=1)


def warp_affine(img: np.ndarray, matrix: np.ndarray, width: int, height: int,
                border_value: float = 0.0) -> np.ndarray:
    """Warp img to (height, width) using a forward affine matrix.

    Behaves like cv2.warpAffine: sampling is done backwards internally and out-of-range pixels are filled with border_value.
    """
    height = int(height)
    width = int(width)
    inv = invert_affine(np.asarray(matrix, dtype=np.float64))

    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    src_x = inv[0, 0] * grid_x + inv[0, 1] * grid_y + inv[0, 2]
    src_y = inv[1, 0] * grid_x + inv[1, 1] * grid_y + inv[1, 2]

    in_bounds = ((src_x >= 0.0) & (src_x <= img.shape[1] - 1)
                 & (src_y >= 0.0) & (src_y <= img.shape[0] - 1))

    sampled = _bilinear(img, np.clip(src_x, 0, img.shape[1] - 1),
                        np.clip(src_y, 0, img.shape[0] - 1))

    if img.ndim == 2 or img.shape[2] == 1:
        out = np.where(in_bounds, sampled[..., 0], border_value)
        return out[..., None].astype(np.float32)
    out = np.where(in_bounds[..., None], sampled, border_value).astype(np.float32)
    return out


# --------------------------------------------------------------------------- NMS
def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Single-class NMS (input is xyxy)."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    return keep


# --------------------------------------------------------------------------- preview drawing
def _draw_disk(canvas: np.ndarray, cx: int, cy: int, radius: int, color) -> None:
    height, width = canvas.shape[:2]
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    canvas[y0:y1, x0:x1][mask] = color


def _draw_line(canvas: np.ndarray, p0, p1, radius: int, color) -> None:
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    steps = int(max(2, length))
    for t in np.linspace(0.0, 1.0, steps):
        x = int(round(p0[0] + (p1[0] - p0[0]) * t))
        y = int(round(p0[1] + (p1[1] - p0[1]) * t))
        _draw_disk(canvas, x, y, radius, color)


def draw_pose_preview(rgb: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                      score_thr: float = 0.3, radius: int = 4) -> np.ndarray:
    """Draw the detected skeleton on a copy of the image, for visual inspection."""
    canvas = rgb.copy()
    if canvas.ndim == 2:
        canvas = np.stack([canvas] * 3, axis=-1)

    def ok(index: int) -> bool:
        return float(scores[index]) >= score_thr

    for i, j in coco.BODY_LIMBS:
        if ok(i) and ok(j):
            color = coco.LIMB_COLORS.get((i, j), (240, 240, 240))
            _draw_line(canvas, keypoints[i], keypoints[j], max(1, radius - 1), color)

    for index in coco.FACE_ALL:
        if ok(index):
            _draw_disk(canvas, int(keypoints[index][0]), int(keypoints[index][1]),
                       1, (120, 255, 120))

    for base in (coco.HAND_LEFT_START, coco.HAND_RIGHT_START):
        for local in range(coco.HAND_COUNT):
            if ok(base + local):
                _draw_disk(canvas, int(keypoints[base + local][0]),
                           int(keypoints[base + local][1]), 1, (255, 120, 255))

    for index in range(0, 23):
        if ok(index):
            _draw_disk(canvas, int(keypoints[index][0]), int(keypoints[index][1]),
                       radius, (255, 255, 0))
    return canvas
