#!/usr/bin/env python3
"""
Warehouse Box Counter — Streamlit application for custom YOLO detection.

Run with:
    python detect_boxes.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.ops import nms
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Warehouse Box Counter"
APP_SUBTITLE = "AI-powered Box Detection using YOLO"
MODEL_FILENAME = "best.pt"
PERSON_MODEL_FILENAME = "yolo11n.pt"
PERSON_CLASS_ID = 0
PERSON_CONF = 0.20
PERSON_SUPPRESS_IOU = 0.08
PERSON_BOX_PADDING = 0.12
# Standard inference (small / medium images — Roboflow-aligned).
INFERENCE_CONF = 0.51
INFERENCE_IOU = 0.50
INFERENCE_IMGSZ = 1280
INFERENCE_MAX_DET = 300
# Tiled inference for large / CCTV warehouse scenes.
TILED_IMAGE_MIN_SIDE = 1000
CCTV_MIN_SIDE = 600
CCTV_ASPECT_RATIO = 1.25
TILED_TILE_SIZE = 640
TILED_SMALL_TILE_SIZE = 480
TILED_OVERLAP = 0.30
TILED_SMALL_OVERLAP = 0.35
TILED_CONF = 0.05
TILED_SMALL_CONF = 0.04
TILED_IOU = 0.45
TILED_MERGE_IOU = 0.40
TILED_IMGSZ = 640
TILED_SMALL_IMGSZ = 480
TILED_MAX_DET = 1000
# Post-filter tiled results to drop trolley rails and other low-quality hits.
TILED_MIN_SCORE = 0.07
TILED_MIN_SCORE_MODERATE = 0.10  # aspect ratio > 2.5
TILED_MIN_SCORE_ELONGATED = 0.12  # aspect ratio > 3.0
TILED_MAX_ASPECT = 3.5
TILED_MAX_AREA_FRAC = 0.02
TILED_MIN_AREA_FRAC = 0.00002
THIN_ARTIFACT_MAX_WIDTH_FRAC = 0.018
THIN_ARTIFACT_MIN_ASPECT = 3.5
BOX_COLOR = (0, 200, 0)  # BGR green
BOX_THICKNESS = 2
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_LOADING_SECONDS = 2.0
MAX_LOADING_SECONDS = 4.0

LOADING_STAGES: list[tuple[str, float, float]] = [
    ("Initializing YOLO model...", 0.18, 0.45),
    ("Detecting boxes...", 0.42, 0.55),
    ("Processing detections...", 0.68, 0.45),
    ("Counting detected boxes...", 0.86, 0.40),
    ("Finalizing results...", 1.00, 0.35),
]

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / MODEL_FILENAME
PERSON_MODEL_PATH = PROJECT_ROOT / PERSON_MODEL_FILENAME


# ---------------------------------------------------------------------------
# Model loading (cached once per server process)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """
    Load the custom YOLO weights once and reuse across sessions.

    Args:
        model_path: Absolute path to the best.pt checkpoint.

    Returns:
        Loaded Ultralytics YOLO model.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError: If the model cannot be loaded.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at '{path}'. "
            f"Place your trained '{MODEL_FILENAME}' in the project root."
        )

    try:
        return YOLO(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc


@st.cache_resource(show_spinner=False)
def load_person_model() -> YOLO:
    """Load a lightweight COCO model used only to mask out people."""
    try:
        return YOLO(str(PERSON_MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(f"Failed to load person filter model: {exc}") from exc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def validate_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    """
    Validate uploaded file extension and image content.

    Raises:
        ValueError: If the file type or content is invalid.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    try:
        Image.open(BytesIO(uploaded_file.getvalue())).verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def read_upload_as_bgr(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> np.ndarray:
    """
    Decode an uploaded image to a BGR NumPy array for YOLO inference.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Image array in OpenCV BGR format.

    Raises:
        ValueError: If decoding fails.
    """
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Unable to read the uploaded image. The file may be corrupted.")
    return image


def draw_clean_boxes(image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Draw simple bounding boxes without labels or confidence scores.

    Args:
        image_bgr: Source image in BGR format.
        boxes_xyxy: Array of shape (N, 4) with x1, y1, x2, y2 coordinates.

    Returns:
        Annotated image copy in BGR format.
    """
    annotated = image_bgr.copy()
    for box in boxes_xyxy:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            BOX_THICKNESS,
            cv2.LINE_AA,
        )
    return annotated


def needs_tiled_inference(image_bgr: np.ndarray) -> bool:
    """
    Use tiled sliding-window inference for large or widescreen CCTV scenes.

    Typical godown CCTV frames (e.g. 1280×720) are below the old 1500 px cutoff
    but still need tiling to find small distant cartons.
    """
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    shortest_side = min(height, width)
    if longest_side >= TILED_IMAGE_MIN_SIDE:
        return True
    return (
        shortest_side >= CCTV_MIN_SIDE
        and longest_side / shortest_side >= CCTV_ASPECT_RATIO
    )


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union for two xyxy boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(float(box_a[2] - box_a[0]), 0.0) * max(float(box_a[3] - box_a[1]), 0.0)
    area_b = max(float(box_b[2] - box_b[0]), 0.0) * max(float(box_b[3] - box_b[1]), 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def expand_box(box: np.ndarray, padding_fraction: float) -> np.ndarray:
    """Expand a box outward by a fraction of its width and height."""
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    return np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)


def point_inside_box(x: float, y: float, box: np.ndarray) -> bool:
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def detect_person_regions(person_model: YOLO, image_bgr: np.ndarray) -> np.ndarray:
    """Return padded person boxes from a COCO pretrained detector."""
    results = person_model.predict(
        source=image_bgr,
        classes=[PERSON_CLASS_ID],
        conf=PERSON_CONF,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    person_boxes = boxes.xyxy.cpu().numpy()
    return np.stack([expand_box(box, PERSON_BOX_PADDING) for box in person_boxes])


def suppress_person_overlaps(
    boxes_xyxy: np.ndarray,
    person_boxes: np.ndarray,
) -> np.ndarray:
    """Drop box detections that overlap people (common CCTV false positive)."""
    if len(boxes_xyxy) == 0 or len(person_boxes) == 0:
        return boxes_xyxy

    kept: list[np.ndarray] = []
    for box in boxes_xyxy:
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        overlaps_person = False
        for person_box in person_boxes:
            if box_iou(box, person_box) >= PERSON_SUPPRESS_IOU:
                overlaps_person = True
                break
            if point_inside_box(center_x, center_y, person_box):
                overlaps_person = True
                break
        if not overlaps_person:
            kept.append(box)

    if not kept:
        return np.empty((0, 4), dtype=np.float32)
    return np.stack(kept)


def suppress_thin_vertical_artifacts(
    boxes_xyxy: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Remove very thin vertical hits typical of trolley handles and rails."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy

    _, width = image_shape[:2]
    max_thin_width = width * THIN_ARTIFACT_MAX_WIDTH_FRAC
    kept: list[np.ndarray] = []

    for box in boxes_xyxy:
        box_width = max(float(box[2] - box[0]), 1.0)
        box_height = max(float(box[3] - box[1]), 1.0)
        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
        is_thin_vertical = (
            box_height > box_width
            and box_width <= max_thin_width
            and aspect_ratio >= THIN_ARTIFACT_MIN_ASPECT
        )
        if not is_thin_vertical:
            kept.append(box)

    if not kept:
        return np.empty((0, 4), dtype=np.float32)
    return np.stack(kept)


def apply_cctv_cleanup(
    boxes_xyxy: np.ndarray,
    image_bgr: np.ndarray,
    person_model: YOLO,
) -> np.ndarray:
    """Final pass to remove people and warehouse-equipment false positives."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy

    person_boxes = detect_person_regions(person_model, image_bgr)
    cleaned = suppress_person_overlaps(boxes_xyxy, person_boxes)
    return suppress_thin_vertical_artifacts(cleaned, image_bgr.shape)


def run_standard_inference(model: YOLO, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass YOLO detection tuned for typical warehouse photos."""
    results = model.predict(
        source=image_bgr,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        imgsz=INFERENCE_IMGSZ,
        max_det=INFERENCE_MAX_DET,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty(0, dtype=np.float32)
        return empty_boxes, empty_scores
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    )


def filter_plausible_boxes(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove low-confidence and implausible detections from tiled inference.

    Tall thin boxes (trolley handles) and oversized boxes (people) need a
    higher confidence than typical cardboard cartons.
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores

    height, width = image_shape[:2]
    image_area = height * width
    kept_indices: list[int] = []

    for index, (box, score) in enumerate(zip(boxes_xyxy, scores)):
        x1, y1, x2, y2 = box
        box_width = max(float(x2 - x1), 1.0)
        box_height = max(float(y2 - y1), 1.0)
        area_fraction = (box_width * box_height) / image_area
        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)

        if area_fraction < TILED_MIN_AREA_FRAC or area_fraction > TILED_MAX_AREA_FRAC:
            continue
        if aspect_ratio > TILED_MAX_ASPECT:
            continue
        if aspect_ratio > 3.0 and score < TILED_MIN_SCORE_ELONGATED:
            continue
        if aspect_ratio > 2.5 and score < TILED_MIN_SCORE_MODERATE:
            continue
        if score < TILED_MIN_SCORE:
            continue

        kept_indices.append(index)

    if not kept_indices:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    kept_indices_array = np.array(kept_indices, dtype=np.int64)
    return boxes_xyxy[kept_indices_array], scores[kept_indices_array]


def merge_detection_sets(
    primary_boxes: np.ndarray,
    secondary_boxes: np.ndarray,
    secondary_scores: np.ndarray,
) -> np.ndarray:
    """Merge high-confidence full-frame hits with filtered tiled detections."""
    if len(primary_boxes) == 0 and len(secondary_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    if len(primary_boxes) == 0:
        return secondary_boxes
    if len(secondary_boxes) == 0:
        return primary_boxes

    primary_scores = np.full(len(primary_boxes), 0.99, dtype=np.float32)
    merged_boxes = np.vstack([primary_boxes, secondary_boxes])
    merged_scores = np.concatenate([primary_scores, secondary_scores.astype(np.float32)])
    keep = nms(
        torch.tensor(merged_boxes, dtype=torch.float32),
        torch.tensor(merged_scores, dtype=torch.float32),
        TILED_MERGE_IOU,
    ).numpy()
    return merged_boxes[keep]


def _predict_tiled_pass(
    model: YOLO,
    image_bgr: np.ndarray,
    tile_size: int,
    overlap: float,
    conf: float,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one sliding-window pass at a given tile size."""
    height, width = image_bgr.shape[:2]
    stride = max(int(tile_size * (1 - overlap)), 1)
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    y_positions = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
    x_positions = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]

    for y in y_positions:
        for x in x_positions:
            y2 = min(y + tile_size, height)
            x2 = min(x + tile_size, width)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)
            crop = image_bgr[y1:y2, x1:x2]

            results = model.predict(
                source=crop,
                conf=conf,
                iou=TILED_IOU,
                imgsz=imgsz,
                max_det=TILED_MAX_DET,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            tile_boxes = boxes.xyxy.cpu().numpy()
            tile_scores = boxes.conf.cpu().numpy()
            tile_boxes[:, [0, 2]] += x1
            tile_boxes[:, [1, 3]] += y1
            all_boxes.append(tile_boxes)
            all_scores.append(tile_scores)

    if not all_boxes:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty(0, dtype=np.float32)
        return empty_boxes, empty_scores

    merged_boxes = np.vstack(all_boxes)
    merged_scores = np.concatenate(all_scores)
    keep = nms(
        torch.tensor(merged_boxes, dtype=torch.float32),
        torch.tensor(merged_scores, dtype=torch.float32),
        TILED_MERGE_IOU,
    ).numpy()
    return merged_boxes[keep], merged_scores[keep]


def run_tiled_inference(model: YOLO, image_bgr: np.ndarray) -> np.ndarray:
    """
    Sliding-window detection for large dense scenes.

    Uses two tile sizes so both mid-range and very small cartons are found in
    stacked godown CCTV footage.
    """
    primary_boxes, primary_scores = _predict_tiled_pass(
        model,
        image_bgr,
        tile_size=TILED_TILE_SIZE,
        overlap=TILED_OVERLAP,
        conf=TILED_CONF,
        imgsz=TILED_IMGSZ,
    )
    small_boxes, small_scores = _predict_tiled_pass(
        model,
        image_bgr,
        tile_size=TILED_SMALL_TILE_SIZE,
        overlap=TILED_SMALL_OVERLAP,
        conf=TILED_SMALL_CONF,
        imgsz=TILED_SMALL_IMGSZ,
    )

    if len(primary_boxes) == 0 and len(small_boxes) == 0:
        merged_boxes = np.empty((0, 4), dtype=np.float32)
        merged_scores = np.empty(0, dtype=np.float32)
    elif len(primary_boxes) == 0:
        merged_boxes, merged_scores = small_boxes, small_scores
    elif len(small_boxes) == 0:
        merged_boxes, merged_scores = primary_boxes, primary_scores
    else:
        merged_boxes = np.vstack([primary_boxes, small_boxes])
        merged_scores = np.concatenate([primary_scores, small_scores])
        keep = nms(
            torch.tensor(merged_boxes, dtype=torch.float32),
            torch.tensor(merged_scores, dtype=torch.float32),
            TILED_MERGE_IOU,
        ).numpy()
        merged_boxes = merged_boxes[keep]
        merged_scores = merged_scores[keep]

    standard_boxes, _ = run_standard_inference(model, image_bgr)
    filtered_boxes, filtered_scores = filter_plausible_boxes(
        merged_boxes, merged_scores, image_bgr.shape
    )
    if len(filtered_boxes) == 0:
        return standard_boxes

    return merge_detection_sets(standard_boxes, filtered_boxes, filtered_scores)


def run_inference(
    model: YOLO,
    image_bgr: np.ndarray,
    person_model: YOLO,
) -> tuple[int, np.ndarray]:
    """
    Run YOLO detection and return box count plus annotated image.

    Automatically selects standard or tiled inference based on image size,
    then removes people and trolley-rail false positives.

    Args:
        model: Loaded custom box detector.
        image_bgr: Input image in BGR format.
        person_model: COCO model used to suppress human false positives.

    Returns:
        Tuple of (box_count, annotated_rgb_image).
    """
    if needs_tiled_inference(image_bgr):
        coords = run_tiled_inference(model, image_bgr)
    else:
        coords, _ = run_standard_inference(model, image_bgr)

    coords = apply_cctv_cleanup(coords, image_bgr, person_model)

    count = len(coords)
    if count > 0:
        annotated_bgr = draw_clean_boxes(image_bgr, coords)
    else:
        annotated_bgr = image_bgr.copy()

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return count, annotated_rgb


def animate_loading(
    progress_bar: st.progress,
    status_text: st.empty,
    inference_fn: Callable[[], tuple[int, np.ndarray]],
) -> tuple[int, np.ndarray]:
    """
    Show staged loading progress while running inference in the background.

    Ensures the UI feels responsive for at least MIN_LOADING_SECONDS.

    Args:
        progress_bar: Streamlit progress bar widget.
        status_text: Streamlit text placeholder for status messages.
        inference_fn: Zero-argument callable returning (count, annotated_rgb).

    Returns:
        Inference results from inference_fn.
    """
    result_container: dict[str, object] = {}
    start_time = time.perf_counter()

    for index, (message, target_progress, stage_duration) in enumerate(LOADING_STAGES):
        status_text.markdown(f"**{message}**")
        steps = max(int(stage_duration / 0.05), 1)
        previous_progress = LOADING_STAGES[index - 1][1] if index > 0 else 0.0

        for step in range(1, steps + 1):
            interpolated = previous_progress + (target_progress - previous_progress) * (
                step / steps
            )
            progress_bar.progress(min(interpolated, 1.0))
            time.sleep(stage_duration / steps)

            # Run inference during the detection stage (once).
            if message.startswith("Detecting") and "result" not in result_container:
                result_container["result"] = inference_fn()

    elapsed = time.perf_counter() - start_time
    if elapsed < MIN_LOADING_SECONDS:
        status_text.markdown("**Finalizing results...**")
        remaining = MIN_LOADING_SECONDS - elapsed
        progress_bar.progress(1.0)
        time.sleep(remaining)
    else:
        progress_bar.progress(1.0)

    if "result" not in result_container:
        result_container["result"] = inference_fn()

    return result_container["result"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def apply_custom_styles() -> None:
    """Inject minimal CSS for a clean, modern layout."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
                max-width: 1100px;
            }
            .app-title {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                color: #1f2937;
            }
            .app-subtitle {
                font-size: 1rem;
                color: #6b7280;
                margin-bottom: 1.75rem;
            }
            .result-card {
                background: #f0fdf4;
                border: 1px solid #bbf7d0;
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                font-size: 1.35rem;
                font-weight: 600;
                color: #166534;
                margin: 1.25rem 0 1.75rem 0;
            }
            div[data-testid="stButton"] button {
                width: 100%;
                background-color: #2563eb;
                color: white;
                font-weight: 600;
                border-radius: 10px;
                padding: 0.65rem 1rem;
                border: none;
            }
            div[data-testid="stButton"] button:hover {
                background-color: #1d4ed8;
                color: white;
            }
            .image-panel img {
                max-height: 280px;
                width: 100%;
                object-fit: contain;
                border-radius: 8px;
            }
            .image-panel p {
                text-align: center;
                font-size: 0.9rem;
                color: #4b5563;
                margin-top: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render application title and subtitle."""
    st.markdown(f'<p class="app-title">{APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)


def show_compact_image(image: Image.Image | np.ndarray, caption: str) -> None:
    """Render an image with a fixed compact height inside a styled panel."""
    st.markdown('<div class="image-panel">', unsafe_allow_html=True)
    st.image(image, caption=caption, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def run_app() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_custom_styles()
    render_header()

    if "box_count" not in st.session_state:
        st.session_state.box_count = None
    if "annotated_rgb" not in st.session_state:
        st.session_state.annotated_rgb = None
    if "last_upload_name" not in st.session_state:
        st.session_state.last_upload_name = None

    uploaded_file = st.file_uploader(
        "Upload a warehouse image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.session_state.box_count = None
        st.session_state.annotated_rgb = None
        st.info("Upload a warehouse image to begin.")
        return

    try:
        validate_upload(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    if uploaded_file.name != st.session_state.last_upload_name:
        st.session_state.box_count = None
        st.session_state.annotated_rgb = None
        st.session_state.last_upload_name = uploaded_file.name

    preview = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")

    if st.button("Count Boxes", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            model = load_model(str(MODEL_PATH))
            person_model = load_person_model()
            image_bgr = read_upload_as_bgr(uploaded_file)

            def _infer() -> tuple[int, np.ndarray]:
                return run_inference(model, image_bgr, person_model)

            count, annotated_rgb = animate_loading(
                progress_bar=progress_bar,
                status_text=status_text,
                inference_fn=_infer,
            )
            st.session_state.box_count = count
            st.session_state.annotated_rgb = annotated_rgb
        except FileNotFoundError as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(str(exc))
            return
        except RuntimeError as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(str(exc))
            return
        except ValueError as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(str(exc))
            return
        except Exception:
            progress_bar.empty()
            status_text.empty()
            st.error("Something went wrong while processing the image. Please try again.")
            return

        progress_bar.empty()
        status_text.empty()

    if st.session_state.box_count is not None:
        st.markdown(
            f'<div class="result-card">✅ Total Boxes Detected: {st.session_state.box_count}</div>',
            unsafe_allow_html=True,
        )

    preview_col, result_col = st.columns(2)

    with preview_col:
        show_compact_image(preview, "Uploaded Image")

    with result_col:
        if st.session_state.annotated_rgb is not None:
            show_compact_image(st.session_state.annotated_rgb, "Detected Boxes")
        else:
            st.markdown(
                '<div class="image-panel" style="min-height:280px;display:flex;'
                'align-items:center;justify-content:center;background:#f9fafb;'
                'border-radius:8px;color:#9ca3af;">Result will appear here</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def is_running_in_streamlit() -> bool:
    """Return True when the script is executed inside a Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_streamlit() -> None:
    """Launch the Streamlit server and open the app in the default browser."""
    app_path = str(Path(__file__).resolve())
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(subprocess.call(command))


def main() -> None:
    """Route execution to Streamlit UI or launcher."""
    if is_running_in_streamlit():
        run_app()
    else:
        launch_streamlit()


if __name__ == "__main__":
    main()
