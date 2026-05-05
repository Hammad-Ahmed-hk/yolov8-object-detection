import io
import os
import time
import tempfile
from collections import Counter

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase
from streamlit_webrtc import webrtc_streamer

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI — YOLOv8 Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Syne:wght@700;800&display=swap');

:root {
    --bg:      #011f4b;   /* darkest navy — main background  */
    --surface: #03396c;   /* deep blue — card surface        */
    --surface2:#005b96;   /* mid blue — hover/alt surface    */
    --border:  rgba(100, 151, 177, 0.25); /* steel blue border */
    --indigo:  #6497b1;   /* steel blue — primary accent     */
    --violet:  #b3cde0;   /* pale blue — secondary accent    */
    --teal:    #005b96;   /* ocean blue — tertiary accent    */
    --amber:   #fbbf24;   /* gold — warnings (contrast pop)  */
    --text:    #ffffff;   /* white — main text     */
    --muted:   #6497b1;   /* steel blue — muted/placeholder  */
    --r:       14px;
}

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 12%  8%, rgba(0,91,150,0.25)  0%, transparent 55%),
        radial-gradient(ellipse 55% 50% at 88% 85%, rgba(3,57,108,0.20)  0%, transparent 55%),
        radial-gradient(ellipse 45% 45% at 55% 50%, rgba(100,151,177,0.10) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

.brand {
    background: linear-gradient(120deg, #6497b1 0%, #b3cde0 50%, #005b96 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-sub { color: var(--muted); font-size: 1rem; margin-top: 0.3rem; }

.glow-hr {
    background: linear-gradient(90deg, transparent, #6497b1, #b3cde0, transparent);
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    background: linear-gradient(90deg, #005b96, #6497b1, #b3cde0);
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    color: var(--indigo); line-height: 1;
}
.stat-lbl { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.2px; margin-top:4px; }

.badge {
    display: inline-block;
    background: rgba(124,143,255,0.12);
    border: 1px solid rgba(124,143,255,0.35);
    border-radius: 30px; padding: 5px 14px;
    font-size: 0.82rem; color: var(--indigo); margin: 3px; font-weight: 600;
}

.pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 30px; padding: 5px 13px; margin: 3px;
    font-size: 0.83rem; font-weight: 600; color: var(--text);
}
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; flex-shrink:0; }

.info-box {
    background: rgba(124,143,255,0.07);
    border-left: 3px solid var(--indigo);
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.9rem; color: var(--text); margin: 0.6rem 0; line-height: 1.6;
}
.warn-box {
    background: rgba(255,187,51,0.07);
    border-left: 3px solid var(--amber);
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.9rem; color: var(--text); margin: 0.6rem 0;
}
.drop-zone {
    border: 2px dashed rgba(124,143,255,0.25);
    border-radius: 18px; padding: 4rem 2rem;
    text-align: center; color: var(--muted); margin-top: 1rem;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, rgba(0,91,150,0.25), rgba(100,151,177,0.20)) !important;
    border: 1px solid #6497b1 !important;
    color: #b3cde0 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    box-shadow: 0 0 20px rgba(100,151,177,0.30) !important;
}


[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--border); }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model() -> YOLO:
    return YOLO("yolov8n.pt")

model: YOLO = load_model()


# ── Colour map (BGR) — all 80 COCO classes ───────────────────────
COLORS: dict[str, tuple] = {
    # People
    "person":           (64,  230,  64),

    # Vehicles
    "bicycle":          (0,   200, 240),
    "car":              (80,  120, 255),
    "motorcycle":       (0,   160, 255),
    "airplane":         (60,  100, 240),
    "bus":              (80,  130, 255),
    "train":            (40,   60, 200),
    "truck":            (60,   80, 220),
    "boat":             (100, 180, 255),

    # Outdoor / street
    "traffic light":    (0,   255, 200),
    "fire hydrant":     (0,   80,  255),
    "stop sign":        (0,   0,   255),
    "parking meter":    (60,  120, 200),
    "bench":            (160, 140,  80),

    # Animals
    "bird":             (255,  0,  180),
    "cat":              (200,  0,  220),
    "dog":              (170,  0,  200),
    "horse":            (220,  80, 200),
    "sheep":            (210, 100, 190),
    "cow":              (190,  60, 180),
    "elephant":         (160,  0,  180),
    "bear":             (130,  0,  160),
    "zebra":            (240, 200, 220),
    "giraffe":          (230, 150, 200),

    # Accessories
    "backpack":         (100, 200, 200),
    "umbrella":         (80,  180, 180),
    "handbag":          (60,  160, 160),
    "tie":              (40,  140, 140),
    "suitcase":         (50,  150, 150),

    # Sports
    "frisbee":          (0,   255, 160),
    "skis":             (20,  220, 180),
    "snowboard":        (40,  200, 170),
    "sports ball":      (50,  220, 220),
    "kite":             (60,  240, 200),
    "baseball bat":     (80,  200, 180),
    "baseball glove":   (100, 180, 160),
    "skateboard":       (120, 220, 200),
    "surfboard":        (140, 200, 220),
    "tennis racket":    (160, 220, 240),

    # Kitchen
    "bottle":           (0,   200, 100),
    "wine glass":       (20,  180, 110),
    "cup":              (0,   180,  90),
    "fork":             (0,   160,  80),
    "knife":            (0,   140,  70),
    "spoon":            (0,   130,  60),
    "bowl":             (0,   120,  50),

    # Food
    "banana":           (0,   240, 220),
    "apple":            (0,   220, 200),
    "sandwich":         (20,  200, 180),
    "orange":           (0,   200, 180),
    "broccoli":         (40,  180, 60),
    "carrot":           (0,   140, 255),
    "hot dog":          (20,  160, 240),
    "pizza":            (0,   180, 160),
    "donut":            (40,  200, 200),
    "cake":             (60,  180, 200),

    # Furniture
    "chair":            (180, 120,  60),
    "couch":            (160, 100,  50),
    "potted plant":     (60,  180,  40),
    "bed":              (140,  80,  40),
    "dining table":     (120,  60,  30),
    "toilet":           (200, 160, 100),

    # Electronics
    "tv":               (0,   180, 150),
    "laptop":           (0,   200, 160),
    "mouse":            (0,   140, 120),
    "remote":           (0,   160, 140),
    "keyboard":         (0,   150, 130),
    "cell phone":       (0,   220, 180),

    # Appliances
    "microwave":        (180, 200, 100),
    "oven":             (160, 180,  80),
    "toaster":          (140, 160,  60),
    "sink":             (120, 200, 160),
    "refrigerator":     (100, 220, 180),

    # Indoor objects
    "book":             (140, 200, 100),
    "clock":            (160, 180,  80),
    "vase":             (120, 160,  70),
    "scissors":         (100, 140,  60),
    "teddy bear":       (200, 140, 180),
    "hair drier":       (220, 160, 200),
    "toothbrush":       (80,  120,  50),
}
_DEFAULT = (50, 80, 255)  # fallback for any unlisted class

def color_for(label: str) -> tuple:
    return COLORS.get(label.lower(), _DEFAULT)

# ── Detection ─────────────────────────────────────────────────────
def process_frame(img: np.ndarray, conf: float, iou: float = 0.45):
    results    = model(img, conf=conf, iou=iou, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            clr   = color_for(label)
            text  = f"{label.capitalize()}  {score:.0%}"
            detections.append(label)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cv2.rectangle(img, (x1, y1-th-10), (x1+tw+6, y1), clr, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), clr, 2)
            cv2.putText(img, text, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (15,15,15), 1, cv2.LINE_AA)
    return img, detections


# ── WebRTC ────────────────────────────────────────────────────────
class VideoTransformer(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img  = frame.to_ndarray(format="bgr24")
        conf = st.session_state.get("conf_slider", 0.20)
        iou  = st.session_state.get("iou_slider",  0.45)
        out, _ = process_frame(img, conf, iou)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ── Helpers ───────────────────────────────────────────────────────
def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    return buf.getvalue()

def safe_open(f) -> Image.Image:
    raw = Image.open(f)
    if raw.mode == "P": raw = raw.convert("RGBA")
    if raw.mode in ("RGBA","LA"):
        bg = Image.new("RGB", raw.size, (255,255,255))
        bg.paste(raw.convert("RGB"), mask=raw.split()[-1])
        return bg
    return raw.convert("RGB")

def summary(dets: list[str], ms: float | None = None):
    counts = Counter(dets)
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, len(dets),    "Objects Found"),
        (c2, len(counts),  "Unique Classes"),
        (c3, f"{ms:.0f} ms" if ms is not None else "—", "Inference Time"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="stat-num">{val}</div>'
                     f'<div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    if counts:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("".join(
            f'<span class="badge">🔹 {l.capitalize()} × {n}</span>'
            for l, n in sorted(counts.items(), key=lambda x: -x[1])
        ), unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("<hr style='border-color:rgba(124,143,255,0.2);margin:0.4rem 0 1rem'>",
                unsafe_allow_html=True)

    app_mode = st.selectbox("Source", ["📷 Webcam (Live)", "🖼️ Image Upload", "🎬 Video Upload"],
                             label_visibility="collapsed")

    st.markdown("**Confidence Threshold**")
    conf_threshold = st.slider("conf", 0.05, 1.0, 0.20, 0.05, key="conf_slider",
                                label_visibility="collapsed",
                                help="Lower = detects more objects. Try 10–20% for crowded scenes.")

    st.markdown("**NMS IoU Threshold**")
    iou_threshold = st.slider("iou", 0.10, 0.90, 0.45, 0.05, key="iou_slider",
                               label_visibility="collapsed",
                               help="Higher = keeps more overlapping boxes.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Color Legend**")
    st.markdown("""
        <div class='pill'><span class='dot' style='background:#40e640'></span>Person</div>
        <div class='pill'><span class='dot' style='background:#5078ff'></span>Vehicle</div>
        <div class='pill'><span class='dot' style='background:#c800dc'></span>Animal</div>
        <div class='pill'><span class='dot' style='background:#00d4b0'></span>Electronics</div>
        <div class='pill'><span class='dot' style='background:#00c864'></span>Food / Items</div>
        <div class='pill'><span class='dot' style='background:#3250ff'></span>Other</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
        🎯 Confidence: <strong>{conf_threshold:.0%}</strong><br>
        📦 Model: <strong>YOLOv8n</strong><br>
        🔢 Detectable Classes: <strong>{len(model.names)}</strong>
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="brand">🎯 VisionAI Object Detector</div>
<p class="brand-sub">Powered by YOLOv8 · {len(model.names)} COCO classes · webcam · images · video</p>
<div class="glow-hr"></div>
""", unsafe_allow_html=True)


# ── WEBCAM ────────────────────────────────────────────────────────
if app_mode == "📷 Webcam (Live)":
    st.markdown("### 📷 Live Webcam Detection")
    st.markdown("""<div class="info-box">
        🟢 Click <strong>START</strong> to begin.
        Sidebar sliders update detections in real time.
    </div>""", unsafe_allow_html=True)

    webrtc_streamer(
        key="live-detection",
        video_processor_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    st.markdown("""<div class="warn-box">
        ⚠️ Webcam requires <strong>HTTPS</strong> in production. Works on <code>localhost</code> without it.
    </div>""", unsafe_allow_html=True)


# ── IMAGE ─────────────────────────────────────────────────────────
elif app_mode == "🖼️ Image Upload":
    st.markdown("### 🖼️ Image Detection")
    uploaded = st.file_uploader("Drop an image", type=["jpg","jpeg","png","webp","bmp"])

    if uploaded:
        try:
            pil_img = safe_open(uploaded)
        except Exception as e:
            st.error(f"❌ Cannot open image: {e}"); st.stop()

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            st.markdown("**📸 Original**")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Running detection…"):
            t0 = time.perf_counter()
            out_img, dets = process_frame(cv_img.copy(), conf_threshold, iou_threshold)
            ms = (time.perf_counter() - t0) * 1000

        with c2:
            st.markdown("**🔍 Detected**")
            st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("<div class='glow-hr'></div>", unsafe_allow_html=True)
        st.markdown("#### 📊 Detection Summary")
        summary(dets, ms)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Result", to_png_bytes(out_img),
                           "visionai_result.png", "image/png")

        if not dets:
            st.markdown("""<div class="warn-box">
                🔎 Nothing detected. Try lowering confidence to 10–15%.
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="drop-zone">
            <div style="font-size:3rem">🖼️</div>
            <div style="font-size:1.05rem;margin-top:0.5rem">Upload an image to begin</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;opacity:.6">JPG · PNG · WEBP · BMP</div>
        </div>""", unsafe_allow_html=True)


# ── VIDEO ─────────────────────────────────────────────────────────
elif app_mode == "🎬 Video Upload":
    st.markdown("### 🎬 Video Detection")
    uploaded = st.file_uploader("Drop a video", type=["mp4","mov","avi","mkv"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read()); tfile.flush()

        cap      = cv2.VideoCapture(tfile.name)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur      = n_frames / fps_src if fps_src > 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("🎞️ Frames",   n_frames)
        m2.metric("⏱️ Duration", f"{dur:.1f} s")
        m3.metric("📐 FPS",      f"{fps_src:.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        run  = st.button("▶️ Run Detection")
        prog = st.progress(0, text="Press ▶️ to start.")
        fph  = st.empty()
        sph  = st.empty()
        all_dets: list[str] = []

        if run:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fc = 0; t0 = time.perf_counter()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                ann, dets = process_frame(frame, conf_threshold, iou_threshold)
                all_dets.extend(dets)
                fph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

                fc += 1
                elapsed  = time.perf_counter() - t0
                live_fps = fc / elapsed if elapsed > 0 else 0
                prog.progress(min(fc / max(n_frames, 1), 1.0),
                              text=f"Frame {fc}/{n_frames}  |  {live_fps:.1f} FPS")

                if fc % 10 == 0 and all_dets:
                    top = Counter(all_dets).most_common(6)
                    sph.markdown("**Live:** " + "  ".join(f"`{k.capitalize()}: {v}`" for k,v in top))

            cap.release()
            try: os.unlink(tfile.name)
            except OSError: pass

            prog.progress(1.0, text="✅ Complete!")
            st.markdown("<div class='glow-hr'></div>", unsafe_allow_html=True)
            st.markdown("#### 📊 Full Video Summary")
            summary(all_dets)
    else:
        st.markdown("""<div class="drop-zone">
            <div style="font-size:3rem">🎬</div>
            <div style="font-size:1.05rem;margin-top:0.5rem">Upload a video for frame-by-frame detection</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;opacity:.6">MP4 · MOV · AVI · MKV</div>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────
st.markdown("<div class='glow-hr'></div>", unsafe_allow_html=True)
st.markdown("""<p style='text-align:center;color:#2d3352;font-size:0.8rem;
    font-family:Space Grotesk,sans-serif;letter-spacing:0.5px'>
    VisionAI &nbsp;·&nbsp; YOLOv8 &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Ultralytics
</p>""", unsafe_allow_html=True)
