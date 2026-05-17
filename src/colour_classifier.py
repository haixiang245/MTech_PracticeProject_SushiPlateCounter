"""Rim colour from HSV masks on the detection strip."""

from __future__ import annotations

import cv2
import numpy as np

from . import config

_STAT_KEYS = ("white_ratio", "black_ratio", "gold_ratio")


def _min_ratio(colour: str) -> float:
    return {
        "white": config.WHITE_MIN,
        "black": config.BLACK_MIN,
        "gold": config.GOLD_MIN,
    }[colour]


def crop_rim_strip_from_box(
    img_bgr: np.ndarray,
    box: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw, bh = max(1, x2 - x1), max(8, y2 - y1)
    xi = config.RIM_STRIP_X_INNER

    xs = max(0, int(x1 + bw * xi))
    xe = min(w, int(x2 - bw * xi))
    ys = max(0, int(y1 + bh * config.RIM_STRIP_Y_TOP))
    ye = min(h, int(y1 + bh * config.RIM_STRIP_Y_BOTTOM))
    if xe <= xs:
        xs, xe = max(0, x1), min(w, x2)
    if ye <= ys:
        ye = min(h, ys + 8)
    return img_bgr[ys:ye, xs:xe].copy(), (xs, ys, xe, ye)


def measure_rim_colours(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> dict[str, float]:
    n = int(H.size)
    if n == 0:
        return {k: 0.0 for k in _STAT_KEYS}

    white = (V >= config.WHITE_V_MIN) & (S <= config.WHITE_S_MAX)
    black = V < config.BLACK_V_MAX
    gold = (
        (H >= config.GOLD_H[0])
        & (H <= config.GOLD_H[1])
        & (S >= config.GOLD_S_MIN)
        & (V >= config.GOLD_V_MIN)
    )

    return {
        "white_ratio": float(np.sum(white)) / n,
        "black_ratio": float(np.sum(black)) / n,
        "gold_ratio": float(np.sum(gold)) / n,
    }


def colour_evidence_from_crop(crop_bgr: np.ndarray) -> dict[str, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return {k: 0.0 for k in _STAT_KEYS}

    small = cv2.resize(crop_bgr, (220, 40), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    m = config.RIM_STRIP_X_INNER
    x1 = int(H.shape[1] * m)
    x2 = int(H.shape[1] * (1.0 - m))
    return measure_rim_colours(H[:, x1:x2], S[:, x1:x2], V[:, x1:x2])


def classify_from_evidence(ev: dict[str, float]) -> tuple[str, float, dict]:
    scores = {
        "white": ev["white_ratio"],
        "black": ev["black_ratio"],
        "gold": ev["gold_ratio"],
    }
    eligible = {
        colour: ratio
        for colour, ratio in scores.items()
        if ratio >= _min_ratio(colour)
    }
    stats = {k: round(ev[k], 3) for k in _STAT_KEYS}
    if not eligible:
        stats["rule"] = "hsv_unclear"
        return "unknown", 0.0, stats
    colour = max(eligible, key=eligible.get)
    conf = float(eligible[colour])
    stats["rule"] = f"hsv_{colour}"
    return colour, conf, stats


def classify_from_ratios(
    white: float,
    black: float,
    gold: float,
) -> tuple[str, float, dict]:
    return classify_from_evidence(
        {"white_ratio": white, "black_ratio": black, "gold_ratio": gold},
    )


def classify_plate_colour_from_rim_strip(
    crop_bgr: np.ndarray,
) -> tuple[str, float, dict]:
    return classify_from_evidence(colour_evidence_from_crop(crop_bgr))


def classify_plate_colour_from_box(
    img_bgr: np.ndarray,
    box: np.ndarray,
) -> tuple[str, float, dict, tuple[int, int, int, int]]:
    strip, box_px = crop_rim_strip_from_box(img_bgr, box)
    if strip.size == 0:
        return "unknown", 0.0, {"rule": "empty"}, (0, 0, 0, 0)
    colour, conf, stats = classify_from_evidence(colour_evidence_from_crop(strip))
    return colour, conf, stats, box_px


def detect_bottom_gold_review_candidate(
    img_bgr: np.ndarray,
    display_boxes: np.ndarray,
) -> dict | None:
    if display_boxes is None or len(display_boxes) == 0:
        return None
    h, w = img_bgr.shape[:2]
    boxes = np.asarray(display_boxes, dtype=float).reshape(-1, 4)
    boxes = boxes[np.argsort([float((b[1] + b[3]) / 2) for b in boxes])]
    x1, y1, x2, y2 = map(int, boxes[-1])
    bh = max(8, y2 - y1)
    sy1 = min(h - 1, int(y2 + bh * 0.10))
    sy2 = min(h, int(h * 0.98))
    if sy2 <= sy1 + 10:
        return None
    region = img_bgr[sy1:sy2, max(0, x1) : min(w, x2)]
    if region.size == 0:
        return None
    ev = colour_evidence_from_crop(region)
    if ev["gold_ratio"] < config.BOTTOM_GOLD_MIN:
        return None
    cy = sy1 + region.shape[0] // 2
    half = max(10, int(bh * 0.55))
    return {
        "box": np.array([x1, cy - half, x2, cy + half], dtype=float),
        "type": "possible_missed_bottom_gold_plate",
        "gold_ratio": round(ev["gold_ratio"], 3),
        "action": "Manual verification recommended; excluded from billing",
    }
