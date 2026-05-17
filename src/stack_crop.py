from __future__ import annotations

import numpy as np

from .config import STACK_CROP


def crop_stack_region(
    image_bgr: np.ndarray,
    x1: float = STACK_CROP["x1"],
    y1: float = STACK_CROP["y1"],
    x2: float = STACK_CROP["x2"],
    y2: float = STACK_CROP["y2"],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    px1 = int(w * x1)
    py1 = int(h * y1)
    px2 = int(w * x2)
    py2 = int(h * y2)
    cropped = image_bgr[py1:py2, px1:px2].copy()
    return cropped, (px1, py1, px2 - px1, py2 - py1)


def map_boxes_to_full_frame(
    boxes: np.ndarray,
    offset: tuple[int, int, int, int],
) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    ox, oy, _, _ = offset
    out = boxes.copy()
    out[:, [0, 2]] += ox
    out[:, [1, 3]] += oy
    return out


def is_likely_already_cropped(image_bgr: np.ndarray, aspect_threshold: float = 0.55) -> bool:
    h, w = image_bgr.shape[:2]
    if h == 0:
        return False
    return (w / h) < aspect_threshold
