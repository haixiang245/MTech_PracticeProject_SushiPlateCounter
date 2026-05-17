"""YOLO11s front-rim detection and multi-stack-safe band selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from . import config


def load_model(weights_path: Path | None = None) -> YOLO:
    path = weights_path or config.WEIGHTS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {path}\n"
            "Install trained weights at sushi_billing_app/weights/front_rim_arc_v2.pt"
        )
    return YOLO(str(path))


def box_w(box: np.ndarray) -> float:
    return max(1.0, float(box[2] - box[0]))


def box_h(box: np.ndarray) -> float:
    return max(1.0, float(box[3] - box[1]))


def centre_x(box: np.ndarray) -> float:
    return float((box[0] + box[2]) / 2)


def centre_y(box: np.ndarray) -> float:
    return float((box[1] + box[3]) / 2)


def aspect_ratio(box: np.ndarray) -> float:
    return box_w(box) / box_h(box)


def x_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1.0, min(box_w(a), box_w(b)))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = box_w(a) * box_h(a)
    area_b = box_w(b) * box_h(b)
    return inter / (area_a + area_b - inter + 1e-9)


def candidate_quality_score(box: np.ndarray, score: float, img_w: int) -> float:
    rel_w = box_w(box) / img_w
    ar_bonus = min(aspect_ratio(box) / 25.0, 1.0)
    return float(score) + 0.12 * min(rel_w, 0.60) + 0.06 * ar_bonus


def generate_frontband_candidates(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = config.CAND_CONF,
    iou: float = config.FINAL_IOU,
    max_det: int = config.MAX_DET,
    imgsz: int = config.IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]

    r = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False,
        augment=config.PREDICT_AUGMENT,
        max_det=max_det,
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    boxes = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
    scores = r.boxes.conf.cpu().numpy().reshape(-1)

    kept_boxes: list[np.ndarray] = []
    kept_scores: list[float] = []

    for box, score in zip(boxes, scores):
        ar = aspect_ratio(box)
        rel_w = box_w(box) / w
        rel_h = box_h(box) / h
        cy = centre_y(box) / h

        if score < conf:
            continue
        if not (1.6 <= ar <= 300.0):
            continue
        if not (config.MIN_REL_WIDTH <= rel_w <= config.MAX_REL_WIDTH):
            continue
        if not (0.002 <= rel_h <= 0.22):
            continue
        if not (0.02 <= cy <= 0.98):
            continue

        kept_boxes.append(box)
        kept_scores.append(float(score))

    if not kept_boxes:
        return np.empty((0, 4)), np.empty((0,))

    return np.asarray(kept_boxes), np.asarray(kept_scores)


def _dedupe_bands(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    *,
    score_floor: float,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores

    img_h, img_w = image_shape
    keep = scores >= score_floor
    boxes = boxes[keep]
    scores = scores[keep]
    if len(boxes) == 0:
        return boxes, scores

    quality = np.array([candidate_quality_score(b, s, img_w) for b, s in zip(boxes, scores)])
    order = np.argsort(-quality)

    heights = np.array([box_h(b) for b in boxes])
    y_close_px = max(4.0, float(np.median(heights)) * config.Y_CLOSE_FACTOR)

    selected: list[int] = []
    for idx in order:
        if limit is not None and len(selected) >= limit:
            break
        box = boxes[idx]
        y = centre_y(box)
        if any(
            box_iou(box, boxes[k]) > config.DEDUP_IOU_THR
            or (
                abs(y - centre_y(boxes[k])) < y_close_px
                and x_overlap_ratio(box, boxes[k]) > config.X_OVERLAP_THR
            )
            for k in selected
        ):
            continue
        selected.append(idx)

    if limit is not None and not selected:
        selected = list(order[: min(limit, len(order))])

    selected.sort(key=lambda i: (centre_y(boxes[i]), centre_x(boxes[i])))
    return boxes[selected], scores[selected]


def select_clean_detected_bands(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    score_floor: float = config.CLEAN_SCORE_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    return _dedupe_bands(boxes, scores, image_shape, score_floor=score_floor)


def select_expected_count_bands(
    boxes: np.ndarray,
    scores: np.ndarray,
    expected_count: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if expected_count <= 0:
        return np.empty((0, 4)), np.empty((0,))
    return _dedupe_bands(
        boxes,
        scores,
        image_shape,
        score_floor=config.CLEAN_SCORE_FLOOR,
        limit=expected_count,
    )


def filter_table_false_positives(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    min_score: float = config.MIN_TABLE_BAND_SCORE,
    max_bottom_cy_frac: float = config.MAX_BOTTOM_CY_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores
    h, _ = image_shape
    mask = [
        not (centre_y(box) / max(1, h) > max_bottom_cy_frac and score < min_score)
        for box, score in zip(boxes, scores)
    ]
    mask_arr = np.array(mask, dtype=bool)
    return boxes[mask_arr], scores[mask_arr]


def canonicalise_frontband_boxes(
    selected_boxes: np.ndarray,
    image_shape: tuple[int, int],
    width_scale: float = config.CANON_WIDTH_SCALE,
    height_scale: float = config.CANON_HEIGHT_SCALE,
    min_h_px: int = config.CANON_MIN_H_PX,
) -> np.ndarray:
    boxes = np.asarray(selected_boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes

    h, w = image_shape
    clean_boxes = []
    for b in boxes:
        cx = centre_x(b)
        cy = centre_y(b)
        target_w = box_w(b) * width_scale
        target_h = max(min_h_px, box_h(b) * height_scale)
        x1 = max(0, int(cx - target_w / 2))
        x2 = min(w, int(cx + target_w / 2))
        y1 = max(0, int(cy - target_h / 2))
        y2 = min(h, int(cy + target_h / 2))
        clean_boxes.append([x1, y1, x2, y2])

    return np.asarray(clean_boxes, dtype=float)


def make_gap_review_notes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    gap_note_factor: float = config.CLEAN_GAP_NOTE_FACTOR,
) -> list[dict]:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) < 3:
        return []

    _, img_h = image_shape
    ys = np.array([centre_y(b) for b in boxes])
    gaps = np.diff(ys)
    valid_gaps = gaps[gaps > 4]
    if len(valid_gaps) == 0:
        return []

    median_gap = float(np.median(valid_gaps))
    notes: list[dict] = []
    for i, gap in enumerate(gaps):
        if gap > median_gap * gap_note_factor:
            notes.append(
                {
                    "type": "possible_missing_internal_rim",
                    "between_detected_rim": f"{i + 1} and {i + 2}",
                    "gap_px": round(float(gap), 1),
                    "median_gap_px": round(median_gap, 1),
                    "action": "Manual verification recommended",
                }
            )

    if ys[-1] + median_gap < img_h * 0.97:
        notes.append(
            {
                "type": "possible_missing_bottom_rim",
                "after_detected_rim": len(boxes),
                "estimated_next_y": round(float(ys[-1] + median_gap), 1),
                "action": "Manual verification recommended",
            }
        )

    return notes
