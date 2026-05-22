"""
utils/helper.py — Shared utility functions.

Contains:
  • load_model()    — loads and caches the YOLOv8 model
  • ensure_dirs()   — creates required directories at startup
  • get_class_names() — returns the 80 COCO class names
"""

import os
import streamlit as st

# ── We import YOLO inside the cached function to avoid a heavy top-level import
# ── at every Streamlit re-run.  The @st.cache_resource decorator ensures the
# ── model is loaded only ONCE per session (not once per re-run).


@st.cache_resource(show_spinner="🤖 Loading YOLOv8 model …")
def load_model(model_path: str = "yolov8m.pt"):
    """
    Load a YOLOv8 model from disk (or download it if not found).

    The first call will download ~6 MB 'yolov8n.pt' from the Ultralytics
    CDN automatically.  Subsequent calls return the cached model instantly.

    Parameters
    ----------
    model_path : str
        Path to the .pt weights file.  Defaults to 'yolov8n.pt'
        (the nano/smallest variant — ideal for demo speed).

    Returns
    -------
    ultralytics.YOLO
        A ready-to-use YOLO model instance.
    """
    from ultralytics import YOLO  # lazy import — only loads PyTorch once

    model = YOLO(model_path)
    return model


def ensure_dirs(dirs: list[str]) -> None:
    """
    Create directories if they do not already exist.

    Parameters
    ----------
    dirs : list[str]
        List of directory paths to create.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── COCO class names (80 classes, index 0–79) ────────────────────────────────
COCO_CLASSES = [
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush",
]


def get_class_name(class_id: int) -> str:
    """
    Return the COCO class name for a given integer class ID.

    Parameters
    ----------
    class_id : int
        Integer index (0–79) from the model output.

    Returns
    -------
    str
        Human-readable class name, or 'unknown' if out of range.
    """
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return "unknown"