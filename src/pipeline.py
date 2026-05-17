from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from ultralytics import YOLO

from . import config
from .billing import bill_summary_markdown, summarise_bill
from .colour_classifier import (
    classify_plate_colour_from_box,
    detect_bottom_gold_review_candidate,
)
from .detector import (
    canonicalise_frontband_boxes,
    filter_table_false_positives,
    generate_frontband_candidates,
    load_model,
    make_gap_review_notes,
    select_clean_detected_bands,
    select_expected_count_bands,
)
from .stack_crop import crop_stack_region, is_likely_already_cropped, map_boxes_to_full_frame
from .visualise import draw_annotated_result

Mode = Literal["organic", "count_assisted"]


@dataclass
class PipelineResult:
    annotated_bgr: np.ndarray
    per_plate_table: list[dict[str, Any]]
    bill_summary: dict[str, Any]
    review_notes: list[dict[str, Any]] = field(default_factory=list)
    crop_offset: tuple[int, int, int, int] | None = None
    candidate_count: int = 0

    def bill_markdown(self) -> str:
        return bill_summary_markdown(self.bill_summary)

    def review_text(self) -> str:
        if not self.review_notes:
            return "No review items require attention."
        lines = []
        for n in self.review_notes:
            lines.append(f"- **{n.get('type', 'note')}**: {n.get('action', n)}")
            for k, v in n.items():
                if k not in ("type", "action"):
                    lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


class SushiBillingPipeline:
    def __init__(self, weights_path: Path | None = None):
        self.model: YOLO = load_model(weights_path)

    def process_image(
        self,
        image: np.ndarray,
        mode: Mode = "organic",
        expected_count: int | None = None,
        auto_crop: bool = True,
        draw_on_full_frame: bool = True,
    ) -> PipelineResult:
        if image is None or image.size == 0:
            raise ValueError("Empty image")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        original = image
        crop_offset: tuple[int, int, int, int] | None = None

        work = original
        if auto_crop and not is_likely_already_cropped(original):
            work, crop_offset = crop_stack_region(original)

        shape = (work.shape[0], work.shape[1])
        cand_boxes, cand_scores = generate_frontband_candidates(self.model, work)

        if mode == "count_assisted":
            if expected_count is None or expected_count < 1:
                raise ValueError("count_assisted mode requires expected_count >= 1")
            final_boxes, final_scores = select_expected_count_bands(
                cand_boxes, cand_scores, expected_count, shape
            )
        else:
            final_boxes, final_scores = select_clean_detected_bands(
                cand_boxes, cand_scores, shape
            )

        final_boxes, final_scores = filter_table_false_positives(
            final_boxes, final_scores, shape
        )

        review_notes = list(make_gap_review_notes(final_boxes, shape))
        display_boxes = canonicalise_frontband_boxes(final_boxes, shape)

        per_plate: list[dict[str, Any]] = []
        for rank, (box, det_score) in enumerate(zip(display_boxes, final_scores), start=1):
            colour, colour_conf, stats, _ = classify_plate_colour_from_box(work, box)
            per_plate.append(
                {
                    "rim_no": rank,
                    "det_conf": round(float(det_score), 3),
                    "colour_class": colour,
                    "colour_conf": round(float(colour_conf), 3),
                    "price": config.PRICE_MAP.get(colour, 0.0),
                    **stats,
                }
            )

        bottom_gold = detect_bottom_gold_review_candidate(work, display_boxes)
        if bottom_gold is not None:
            review_notes.append(bottom_gold)

        bill = summarise_bill(per_plate)

        draw_boxes = display_boxes
        draw_image = work
        if draw_on_full_frame and crop_offset is not None:
            draw_boxes = map_boxes_to_full_frame(display_boxes, crop_offset)
            draw_image = original
            if bottom_gold is not None:
                bg = bottom_gold.copy()
                bg["box"] = map_boxes_to_full_frame(
                    np.asarray([bottom_gold["box"]]), crop_offset
                )[0]
                bottom_gold = bg

        annotated = draw_annotated_result(draw_image, draw_boxes, per_plate, bottom_gold)

        return PipelineResult(
            annotated_bgr=annotated,
            per_plate_table=per_plate,
            bill_summary=bill,
            review_notes=review_notes,
            crop_offset=crop_offset,
            candidate_count=len(cand_boxes),
        )


_pipeline: SushiBillingPipeline | None = None


def get_pipeline(weights_path: Path | None = None) -> SushiBillingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SushiBillingPipeline(weights_path)
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


def process_image(
    image: np.ndarray,
    mode: Mode = "organic",
    expected_count: int | None = None,
    weights_path: Path | None = None,
    auto_crop: bool = True,
) -> PipelineResult:
    return get_pipeline(weights_path).process_image(
        image,
        mode=mode,
        expected_count=expected_count,
        auto_crop=auto_crop,
    )
