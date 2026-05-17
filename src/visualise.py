from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import config
from .colour_classifier import crop_rim_strip_from_box

_COLOUR_STYLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "gold": {"box": (0, 220, 255), "text": (0, 220, 255), "outline": (0, 120, 180)},
    "black": {"box": (0, 0, 0), "text": (0, 0, 0), "outline": (210, 210, 210)},
    "white": {"box": (255, 255, 255), "text": (255, 255, 255), "outline": (70, 70, 70)},
}

_REVIEW_STYLE = {
    "box": (210, 210, 210),
    "text": (210, 210, 210),
    "outline": (150, 150, 150),
}


def _style_for_row(row: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    if row.get("colour_class") in ("unknown", None):
        return _REVIEW_STYLE
    if float(row.get("colour_conf", 1.0)) < config.COLOUR_LOW_CONF_THRESHOLD:
        return _REVIEW_STYLE
    return _COLOUR_STYLES.get(row.get("colour_class", ""), _REVIEW_STYLE)


def _draw_outlined_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    outline: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.putText(
            img,
            text,
            (org[0] + dx, org[1] + dy),
            font,
            scale,
            outline,
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_annotated_result(
    image_bgr: np.ndarray,
    display_boxes: np.ndarray,
    per_plate_rows: list[dict[str, Any]],
    bottom_gold_candidate: dict[str, Any] | None = None,
) -> np.ndarray:
    drawn = image_bgr.copy()

    for row, box in zip(per_plate_rows, display_boxes):
        colour_conf = float(row.get("colour_conf", 0.0))
        _, strip_box = crop_rim_strip_from_box(image_bgr, box)
        sx1, sy1, sx2, sy2 = strip_box
        style = _style_for_row(row)

        cv2.rectangle(drawn, (sx1, sy1), (sx2, sy2), style["box"], 2, cv2.LINE_AA)

        label = f"{row['rim_no']} {row['colour_class']} ({colour_conf:.2f})"
        if style is _REVIEW_STYLE:
            label = f"{row['rim_no']} REVIEW {row['colour_class']} ({colour_conf:.2f})"
        _draw_outlined_text(
            drawn,
            label,
            (sx1, max(22, sy1 - 6)),
            style["text"],
            style["outline"],
        )

    if bottom_gold_candidate is not None:
        bx1, by1, bx2, by2 = [int(v) for v in bottom_gold_candidate["box"].tolist()]
        st = _REVIEW_STYLE
        cv2.rectangle(drawn, (bx1, by1), (bx2, by2), st["box"], 2, cv2.LINE_AA)
        _draw_outlined_text(
            drawn,
            f"REVIEW gold candidate ({bottom_gold_candidate.get('gold_ratio', 0):.2f})",
            (bx1, max(22, by1 - 6)),
            st["text"],
            st["outline"],
        )

    return drawn
