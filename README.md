# 🎯 Real-Time Multi-Object Detection Using YOLOv8

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **complete, production-structured college project** demonstrating real-time
object detection across images, uploaded videos, and live webcam feeds —
all inside a modern Streamlit web interface.

---

## 📸 Screenshots

| Home Page | Image Detection |
|-----------|----------------|
| *(dark home with metrics)* | *(side-by-side original vs detected)* |

| Video Processing | Webcam Live Feed |
|-----------------|-----------------|
| *(progress bar + preview)* | *(live FPS + annotated feed)* |

> **Tip:** Run the app and take your own screenshots for GitHub!

---

## 🎯 Project Overview

This application uses **YOLOv8** (You Only Look Once — version 8), a
state-of-the-art single-stage object detector, to identify and localise
80+ object categories in real time.

**Key capabilities:**
- ✅ Detect objects in **uploaded images** (JPG / PNG)
- ✅ Detect objects in **uploaded videos** (MP4 / AVI / MOV)
- ✅ Detect objects from a **live webcam feed** with FPS display
- ✅ Draw **bounding boxes**, **class labels**, and **confidence scores**
- ✅ Filter detections by a **confidence threshold** slider
- ✅ **Download** annotated images and processed videos
- ✅ Object **count** and **confidence breakdown** table
- ✅ Modern dark-theme **Streamlit UI**

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.9 + | Core language |
| **YOLOv8** (Ultralytics) | 8.x | Object detection model |
| **PyTorch** | auto via ultralytics | Deep-learning backend |
| **OpenCV** | 4.8 + | Frame I/O, drawing |
| **Streamlit** | 1.32 + | Interactive web UI |
| **Pillow (PIL)** | 10.x | Image open / save |
| **NumPy** | 1.24 + | Array operations |
| **Pandas** | 2.x | Detection summary tables |

---

## 📁 Project Structure

```
yolov8_detection/
│
├── app.py               ← Main Streamlit app (router + sidebar)
├── detect_image.py      ← Image detection page
├── detect_video.py      ← Video detection page
├── detect_webcam.py     ← Live webcam detection page
├── requirements.txt     ← All pip dependencies
├── README.md            ← This file
│
├── utils/
│   ├── __init__.py      ← Makes utils a Python package
│   ├── helper.py        ← Model loading + COCO class names
│   └── drawing.py       ← Bounding box drawing + summary table
│
├── uploads/             ← Temp storage for uploaded files
├── outputs/             ← Annotated images / videos saved here
│
├── assets/
│   └── sample_images/   ← (Optional) drop sample JPGs here
│
└── .streamlit/
    └── config.toml      ← Dark theme configuration
```

---

## 🖥️ Installation

### Prerequisites
- **Python 3.9 or higher** ([download](https://www.python.org/downloads/))
- **pip** (included with Python)
- A **webcam** (optional, for webcam mode)
- **Windows 10/11** (also works on macOS & Linux)

### Step 1 — Clone / download the project

```bash
# Option A: Clone via Git
git clone https://github.com/your-username/yolov8-detection.git
cd yolov8-detection

# Option B: Download ZIP from GitHub and extract it, then open a terminal in the folder
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads PyTorch (~200 MB) and Ultralytics.
> This is a one-time process.

### Step 4 — (Optional) Download the YOLOv8 weights manually

The app auto-downloads **yolov8n.pt** (~6 MB) on first run.
If you prefer to download it manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## ▶️ Running the App

```bash
# Make sure your virtual environment is activated, then:
streamlit run app.py
```

The app will open automatically in your default browser at:
**http://localhost:8501**

---

## 🗂️ How to Use

### 🖼️ Image Detection
1. Click **"🖼️ Image Detection"** in the sidebar
2. Upload a JPG or PNG image
3. Wait ~1 second for inference
4. View original vs annotated image side by side
5. See the object breakdown table
6. Click **"⬇️ Download Annotated Image"**

### 🎬 Video Detection
1. Click **"🎬 Video Detection"** in the sidebar
2. Upload an MP4 / AVI / MOV file
3. Watch the frame-by-frame progress bar and live preview
4. Download the fully annotated video when done

### 📷 Webcam Detection
1. Click **"📷 Webcam Detection"** in the sidebar
2. Press **▶ Start Webcam**
3. Real-time detections appear with FPS display
4. Click **📸 Take Snapshot** to download the current frame
5. Press **⏹ Stop Webcam** to release the camera

### ⚙️ Confidence Threshold
- Use the **🎚️ Confidence Threshold** slider in the sidebar
- Higher values → fewer, more certain detections
- Lower values → more detections (may include false positives)

---

## 📦 Detectable Objects (COCO 80 Classes)

```
person      bicycle     car         motorbike   aeroplane   bus
train       truck       boat        traffic light  fire hydrant  stop sign
bench       bird        cat         dog         horse       sheep
cow         elephant    bear        zebra       giraffe     backpack
umbrella    handbag     tie         suitcase    frisbee     skis
snowboard   sports ball  kite       baseball bat  bottle     wine glass
cup         fork        knife       spoon       bowl        banana
apple       sandwich    orange      broccoli    carrot      hot dog
pizza       donut       cake        chair       sofa        bed
diningtable toilet      tvmonitor   laptop      mouse       remote
keyboard    cell phone  microwave   oven        toaster     sink
refrigerator  book      clock       vase        scissors    teddy bear
hair drier  toothbrush
```

---

## 🔧 Configuration

Edit `.streamlit/config.toml` to customise the theme:

```toml
[theme]
primaryColor    = "#e94560"   # accent colour (red)
backgroundColor = "#0f0f1a"   # main background
secondaryBackgroundColor = "#16213e"  # sidebar / cards
```

---

## 🚀 Future Improvements

- [ ] **Object tracking** (SORT / DeepSORT) across video frames
- [ ] **Custom model** fine-tuned on a domain-specific dataset
- [ ] **REST API** via FastAPI for headless inference
- [ ] **GPU support** (CUDA) for 10× faster inference
- [ ] **ONNX / TFLite** export for edge/mobile deployment
- [ ] **Multiple camera** selection dropdown
- [ ] **Heatmap overlays** showing detection density over time
- [ ] **Audio alerts** when specific objects are detected

---

## 📚 Learning Outcomes

After completing / studying this project, you will understand:

1. **YOLO Architecture** — single-stage detection, anchor boxes, NMS
2. **Transfer Learning** — using pre-trained weights for inference
3. **OpenCV Pipelines** — reading video streams, drawing on frames
4. **Streamlit** — building interactive ML apps without front-end code
5. **Python Project Structure** — modules, packages, separation of concerns
6. **Real-Time Systems** — frame loops, FPS measurement, UI responsiveness
7. **File Handling** — uploads, binary downloads, temp files

---

## 📄 License

This project is released under the **MIT License** — free for personal,
educational, and commercial use.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — model & inference engine
- [Streamlit](https://streamlit.io) — web UI framework
- [OpenCV](https://opencv.org) — computer vision toolkit
- [COCO Dataset](https://cocodataset.org) — pre-training dataset (80 classes)

---

*Built as a Final-Year College Project · Python · YOLOv8 · Streamlit · OpenCV*
