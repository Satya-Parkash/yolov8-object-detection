"""
utils/drawing.py — All annotation / drawing utilities.

Contains:
  • draw_detections()     — draw boxes, labels & confidence on a NumPy frame
  • build_summary_table() — return a pandas DataFrame summarising detections
  • _get_color()          — deterministic per-class colour from a fixed palette
"""

import colorsys
import numpy as np
import cv2
import pandas as pd

from utils.helper import get_class_name


# ── Pre-generate a palette of 80 visually distinct colours (one per COCO class)
_PALETTE: list[tuple[int, int, int]] = []
for _i in range(80):
    hue        = _i / 80.0
    r, g, b    = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    _PALETTE.append((int(r * 255), int(g * 255), int(b * 255)))


def _get_color(class_id: int) -> tuple[int, int, int]:
    """Return an (R, G, B) colour for the given COCO class index."""
    return _PALETTE[int(class_id) % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    results,
    confidence_threshold: float = 0.40,
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and confidence scores on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        RGB image/frame (H × W × 3, uint8).
    results : list
        Output from ``model.predict()``.
    confidence_threshold : float
        Detections below this score are skipped.

    Returns
    -------
    np.ndarray
        Annotated RGB frame (same shape as input).
    """

    if results is None or len(results) == 0:
        return frame

    boxes = results[0].boxes  # ultralytics Boxes object

    if boxes is None or len(boxes) == 0:
        return frame

    h, w = frame.shape[:2]

    for box in boxes:
        conf     = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        label    = get_class_name(class_id)
        color    = _get_color(class_id)          # (R, G, B)
        color_bgr = color[::-1]                  # OpenCV uses BGR

        # ── Bounding box coordinates (xyxy format) ──────────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # ── Draw filled rectangle (main box) ────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

        # ── Label background ─────────────────────────────────────────────────
        text     = f"{label}  {conf:.0%}"
        font     = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = max(0.45, min(0.75, (x2 - x1) / 300))
        thickness   = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Place label above box; if no space, put it inside
        label_y = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 6

        # Filled background for readability
        cv2.rectangle(
            frame,
            (x1, label_y - th - 4),
            (x1 + tw + 6, label_y + 2),
            color_bgr,
            cv2.FILLED,
        )

        # ── Draw label text (white) ──────────────────────────────────────────
        cv2.putText(
            frame, text,
            (x1 + 3, label_y - 2),
            font, font_scale,
            (255, 255, 255),   # white text
            thickness,
            cv2.LINE_AA,
        )

    # ── HUD: small detection count overlay (bottom-left) ────────────────────
    num_dets = sum(
        1 for box in boxes if float(box.conf[0]) >= confidence_threshold
    )
    hud_text = f"Objects: {num_dets}  |  Conf >= {confidence_threshold:.0%}"
    cv2.putText(
        frame, hud_text,
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 128),  # bright green
        1, cv2.LINE_AA,
    )

    return frame


def build_summary_table(results) -> pd.DataFrame:
    """
    Build a pandas DataFrame summarising the detections.

    Returns a table with columns:
        Object, Count, Avg Confidence, Max Confidence

    Parameters
    ----------
    results : list
        Output from ``model.predict()``.

    Returns
    -------
    pd.DataFrame
        Summary table, sorted by Count descending.
    """
    if results is None or len(results) == 0:
        return pd.DataFrame()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame()

    records: dict[str, list[float]] = {}

    for box in boxes:
        class_id = int(box.cls[0])
        conf     = float(box.conf[0])
        label    = get_class_name(class_id)

        records.setdefault(label, []).append(conf)

    rows = []
    for label, confs in records.items():
        rows.append({
            "Object":           label.capitalize(),
            "Count":            len(confs),
            "Avg Confidence":   f"{sum(confs)/len(confs):.1%}",
            "Max Confidence":   f"{max(confs):.1%}",
        })

    df = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
    return df
