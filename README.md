# Sushi Plate Billing Application

## Overview

This application performs stateless billing for vertically stacked sushi plates from a single photograph. It produces an annotated image, a per-plate classification table, and a price summary.

The processing pipeline comprises:

1. Optional region-of-interest cropping for full-width conveyor frames.
2. Front-rim detection using a YOLO11s model (`sushi_plate_front_rim_arc`).
3. Plate colour classification from HSV pixel fractions on each rim strip.
4. Price aggregation from configurable unit rates.

The application does not persist images or results; all processing occurs in memory.

## Default pricing

Configure unit prices in `src/config.py` (`PRICE_MAP`).

| Plate colour | Unit price (USD) |
|--------------|-----------------:|
| White        | 1.50             |
| Black        | 2.00             |
| Gold         | 3.00             |
| Unknown      | 0.00             |

## Colour classification rule

For each detected rim, the classifier measures the fraction of pixels classified as white, black, or gold in HSV space on the inner rim strip. The billed colour is the eligible category with the highest fraction, where eligibility requires meeting `WHITE_MIN`, `BLACK_MIN`, or `GOLD_MIN` respectively. If no category meets its threshold, the plate is classified as unknown.

## System requirements

| Requirement | Specification |
|-------------|----------------|
| Python      | 3.10 or later   |
| Disk space  | Approximately 2 GB (PyTorch, Ultralytics) |
| GPU         | Optional; CPU inference is supported      |
| NumPy       | 1.x (`numpy>=1.24,<2`) for compatibility with PyTorch 2.2 on macOS |

## Installation

```bash
git clone https://github.com/haixiang245/MTech_PracticeProject_SushiPlateCounter.git
cd MTech_PracticeProject_SushiPlateCounter
pip install -r requirements.txt
python app.py
```

Open the web interface at `http://127.0.0.1:7860`.

The trained model weights (`weights/front_rim_arc_v2.pt`) are included in this repository.

## Model training

Front-rim detector training is documented in:

[`notebooks/SushiPlateCounter_RimArcDetection_ModelTraining.ipynb`](notebooks/SushiPlateCounter_RimArcDetection_ModelTraining.ipynb)

The notebook was executed in Google Colab (GPU). It covers dataset preparation from rim annotations, YOLO11s training (`rim_yolo11s_FRONTBAND_ARC_CROP_V2`), and evaluation. The exported weights correspond to `weights/front_rim_arc_v2.pt` used by this application.

## Project structure

```text
├── app.py                 # Gradio web interface
├── pyproject.toml
├── requirements.txt
├── Makefile
├── LICENSE
├── README.md
├── notebooks/
│   └── SushiPlateCounter_RimArcDetection_ModelTraining.ipynb
├── weights/
│   └── front_rim_arc_v2.pt
└── src/
    ├── config.py          # thresholds, prices, ROI
    ├── stack_crop.py
    ├── detector.py
    ├── colour_classifier.py
    ├── billing.py
    ├── visualise.py
    └── pipeline.py
```

## Web interface

| Control | Description |
|---------|-------------|
| Automatic detection | Rim count determined by the detection and deduplication pipeline |
| Fixed plate count | Select exactly the specified number of highest-confidence rims |

Stack cropping is applied automatically when the input aspect ratio indicates a full conveyor frame.

### Annotation legend

| Visual element | Meaning |
|----------------|---------|
| Cyan rectangle | Colour sample region on the rim strip |
| Coloured label | Classified plate colour |
| Grey label | Unknown classification or confidence below threshold |

## Python API

```python
import cv2
from src.pipeline import process_image

image_bgr = cv2.imread("stack.jpg")
result = process_image(image_bgr, mode="organic", auto_crop=True)

print(result.per_plate_table)
print(result.bill_summary)
print(result.bill_markdown())
cv2.imwrite("annotated.jpg", result.annotated_bgr)
```

When the plate count is known in advance:

```python
result = process_image(
    image_bgr,
    mode="count_assisted",
    expected_count=4,
    auto_crop=True,
)
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | HTTP bind address |
| `GRADIO_SERVER_PORT` | `7860` | HTTP port |
| `GRADIO_SHARE` | unset | Set to `true` to enable a public Gradio tunnel |

## Configuration reference

Edit `src/config.py` and restart the application.

| Parameter | Purpose |
|-----------|---------|
| `CAND_CONF`, `CLEAN_SCORE_FLOOR` | Detection sensitivity |
| `STACK_CROP` | Region of interest for stack cropping |
| `WHITE_V_MIN`, `BLACK_V_MAX`, `GOLD_H`, etc. | HSV classification bounds |
| `WHITE_MIN`, `BLACK_MIN`, `GOLD_MIN` | Minimum pixel fraction per colour |
| `PRICE_MAP` | Unit prices |

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| `Model weights not found` | Verify `weights/front_rim_arc_v2.pt` is present |
| `Numpy is not available` | Install `numpy>=1.24,<2` and restart the process |
| Under-detection of rims | Use fixed plate count mode or reduce `CLEAN_SCORE_FLOOR` |
| Over-detection of rims | Increase `CLEAN_SCORE_FLOOR`; verify stack cropping |
| Incorrect colours | Inspect the annotated sample region; adjust HSV parameters |

## Deployment

For Hugging Face Spaces or similar hosts:

1. Deploy this repository as a Gradio application.
2. Entry point: `app.py`; dependencies: `requirements.txt`.
3. Model weights are already under `weights/`.
