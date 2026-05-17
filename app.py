#!/usr/bin/env python3
"""Gradio web interface for the sushi plate billing pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.pipeline import get_pipeline, process_image  # noqa: E402

TABLE_COLUMNS = [
    "rim_no",
    "colour_class",
    "rule",
    "colour_conf",
    "det_conf",
    "white_ratio",
    "black_ratio",
    "gold_ratio",
    "price",
]

UI_DESCRIPTION = """
## Sushi Plate Billing

Stateless billing from a single stack photograph.

**Pipeline:** region cropping (when applicable), front-rim detection, HSV rim-strip classification, price aggregation.

**Annotation legend**

| Indicator | Interpretation |
|-----------|----------------|
| Yellow label | Gold plate |
| Black or white label | Matching rim colour |
| Grey label | Unknown or low-confidence classification |

**Model:** YOLO11s front-rim detector (`weights/front_rim_arc_v2.pt`).
"""


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("No input image provided.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def run_billing(
    image_rgb: np.ndarray,
    mode: str,
    expected_count: float | None,
) -> tuple[np.ndarray, pd.DataFrame, str, str, str]:
    bgr = _to_bgr(image_rgb)

    count: int | None = None
    if mode == "count_assisted":
        if expected_count is None or expected_count < 1:
            raise gr.Error(
                "Fixed plate count mode requires an expected plate count of at least 1."
            )
        count = int(expected_count)

    result = process_image(
        bgr,
        mode="count_assisted" if mode == "count_assisted" else "organic",
        expected_count=count,
        auto_crop=True,
    )

    annotated_rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
    table_df = pd.DataFrame(result.per_plate_table)
    table_df = table_df[[c for c in TABLE_COLUMNS if c in table_df.columns]]

    crop_status = "Applied" if result.crop_offset else "Not applied"
    meta = (
        f"| Metric | Value |\n"
        f"|--------|------:|\n"
        f"| Detected rims | {len(result.per_plate_table)} |\n"
        f"| YOLO candidates | {result.candidate_count} |\n"
        f"| Stack crop | {crop_status} |"
    )
    return annotated_rgb, table_df, result.bill_markdown(), result.review_text(), meta


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sushi Plate Billing") as demo:
        gr.Markdown(UI_DESCRIPTION)

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="numpy", label="Input image")
                mode = gr.Radio(
                    choices=[
                        ("Automatic detection", "organic"),
                        ("Fixed plate count", "count_assisted"),
                    ],
                    value="organic",
                    label="Detection mode",
                )
                expected_count = gr.Number(
                    value=4,
                    precision=0,
                    label="Expected plate count (fixed plate count mode only)",
                )
                run_btn = gr.Button("Generate bill", variant="primary")

            with gr.Column():
                image_out = gr.Image(type="numpy", label="Annotated output")
                meta_out = gr.Markdown(label="Run summary")
                table_out = gr.Dataframe(label="Per-plate results")
                bill_out = gr.Markdown(label="Bill summary")
                review_out = gr.Markdown(label="Review flags")

        run_btn.click(
            fn=run_billing,
            inputs=[image_in, mode, expected_count],
            outputs=[image_out, table_out, bill_out, review_out, meta_out],
        )

    return demo


def main() -> None:
    get_pipeline()
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
    )


if __name__ == "__main__":
    main()
