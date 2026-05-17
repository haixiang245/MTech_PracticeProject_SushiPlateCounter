"""Sushi plate stack billing: YOLO front-rim detection, HSV classification, pricing."""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import PipelineResult, SushiBillingPipeline, get_pipeline, process_image, reset_pipeline

__all__ = [
    "__version__",
    "PipelineResult",
    "SushiBillingPipeline",
    "get_pipeline",
    "process_image",
    "reset_pipeline",
]
