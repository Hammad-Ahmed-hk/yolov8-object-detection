# 🎯 YOLOv8 Real-Time Object Detection

A web-based AI object detection application built with **YOLOv8** and **Streamlit**.  
Detect objects instantly from your **webcam**, an **uploaded image**, or an **uploaded video** — all from your browser.

---

## 📸 Features

- 📷 **Live Webcam Detection** — real-time inference via WebRTC, frame by frame
- 🖼️ **Image Upload** — upload JPG / JPEG / PNG and get annotated results instantly
- 🎬 **Video Upload** — process MP4 / MOV / AVI files with per-frame detection
- 🎚️ **Adjustable Confidence Threshold** — slider to control detection sensitivity
- 🎨 **Color-coded Bounding Boxes** — a colour per COCO class (green for people, blues for vehicles, purples for animals, and so on; legend in the sidebar)
- 🎚️ **NMS IoU slider** to control how overlapping boxes are merged
- ⚡ **Auto model download** — `yolov8n.pt` is fetched automatically on first run

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) | Object detection model |
| [Streamlit](https://streamlit.io/) | Web UI framework |
| [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) | Live webcam streaming |
| [OpenCV](https://opencv.org/) | Frame processing & bounding boxes |
| [Pillow](https://python-pillow.org/) | Image loading & format handling |
| [PyAV](https://github.com/PyAV-Org/PyAV) | Video frame decoding for WebRTC |

---

## 📁 Project Structure

```
yolov8-object-detection/
├── app.py              # Streamlit app (UI, detection, webcam/image/video modes)
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Community Cloud (libgl1, libglib2.0-0)
├── runtime.txt         # Python version hint (python-3.11)
└── .python-version     # Python version for pyenv/uv (3.11)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hammad-Ahmed-hk/yolov8-object-detection.git
cd yolov8-object-detection
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

> **Note:** On first run, `yolov8n.pt` (~6 MB) will be downloaded automatically. No manual setup needed.

---

## 📦 Dependencies

Install everything with `pip install -r requirements.txt` (Python 3.11 recommended).
`streamlit-webrtc` and `tornado` are pinned in that file because newer versions broke the webcam stream; leave the pins alone unless you retest.

---

## 🎮 How to Use

### 📷 Webcam Mode
1. Select **Webcam** from the sidebar dropdown
2. Click **START** to begin the stream
3. Allow browser camera permission when prompted
4. Adjust the **Confidence** and **NMS IoU** sliders in the sidebar

### 🖼️ Image Upload Mode
1. Select **Image Upload** from the sidebar
2. Click **Browse files** and upload a `.jpg`, `.jpeg`, `.png`, `.webp` or `.bmp`
3. The detected image with bounding boxes appears instantly

### 🎬 Video Upload Mode
1. Select **Video Upload** from the sidebar
2. Upload an `.mp4`, `.mov`, `.avi` or `.mkv` file
3. Click **▶️ Run Detection** and watch frame-by-frame detection play out

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Confidence Threshold | `0.20` | Minimum score to show a detection box (slider: 0.05 – 1.0) |
| NMS IoU Threshold | `0.45` | Higher keeps more overlapping boxes (slider: 0.10 – 0.90) |
| Model | `yolov8n.pt` | Nano model — fast & lightweight |

To use a larger / more accurate model, change this line in `app.py`:

```python
return YOLO("yolov8n.pt")   # swap to yolov8s.pt / yolov8m.pt / yolov8l.pt
```

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | ~6 MB | ⚡ Fastest | Good |
| `yolov8s.pt` | ~22 MB | Fast | Better |
| `yolov8m.pt` | ~52 MB | Medium | Great |
| `yolov8l.pt` | ~87 MB | Slow | Best |

---

## 🧠 Detection Categories

YOLOv8 can detect **80 COCO classes** including:

`person · car · truck · bus · bicycle · motorcycle · airplane · boat · traffic light · dog · cat · chair · bottle · laptop · cell phone` and many more.

---

## 📌 Known Issues & Tips

| Issue | Fix |
|-------|-----|
| Webcam not starting | Allow camera permission in browser |
| Slow on CPU | Use `yolov8n` (default) or enable GPU |
| Transparent PNG crash | Already handled — images are flattened onto a white background |
| Video plays too fast | Normal — OpenCV reads frames at max CPU speed |

---

## 🤝 Contributing

Pull requests are welcome!

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

No `LICENSE` file has been added yet. Add one (for example MIT) before others reuse the code.

---

## 👤 Author

**Hammad Ahmed** · [GitHub](https://github.com/Hammad-Ahmed-hk)

> Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/)
