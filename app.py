import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tempfile
import os
import time
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF8C00, #E74C3C, #3498DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 20px 24px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 6px solid;
    }
    .metric-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        margin: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF8C00, #E74C3C);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 28px;
        font-size: 1rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #444;
        border-radius: 12px;
        padding: 10px;
    }
    .path-found {
        background: #0d2b1a;
        border: 1px solid #2ECC71;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #2ECC71;
        margin-bottom: 10px;
    }
    .path-error {
        background: #2b0d0d;
        border: 1px solid #E74C3C;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #E74C3C;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Class config ──────────────────────────────────────────────────────────────
CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']

CLASS_COLORS = {
    "Bus":        "#FF8C00",
    "Car":        "#2ECC71",
    "Motorcycle": "#E74C3C",
    "Truck":      "#3498DB",
}

CLASS_ICONS = {
    "Bus":        "🚌",
    "Car":        "🚗",
    "Motorcycle": "🏍️",
    "Truck":      "🚛",
}

CLASS_COLORS_BGR = {
    "Bus":        (50, 140, 255),
    "Car":        (50, 204, 113),
    "Motorcycle": (50, 76, 231),
    "Truck":      (219, 152, 52),
}

# ── Model path auto-discovery ─────────────────────────────────────────────────
def find_model_weights():
    """
    Searches for best.pt in common locations relative to this script.
    Returns the path string if found, else None.
    Priority order:
      1. Exact project path from notebook: runs/vehicle_cls/capstone_v1/weights/best.pt
      2. Any best.pt anywhere under runs/
      3. Any best.pt anywhere under the current working directory
    """
    # 1. Exact path used in notebook
    candidates = [
        Path("runs/vehicle_cls/capstone_v1/weights/best.pt"),
        Path("runs/vehicle_cls/capstone_v1/weights/last.pt"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # 2. Search under runs/ recursively
    runs_dir = Path("runs")
    if runs_dir.exists():
        for pt in sorted(runs_dir.rglob("best.pt")):
            return str(pt)
        for pt in sorted(runs_dir.rglob("last.pt")):
            return str(pt)

    # 3. Search entire working directory recursively (slower)
    for pt in sorted(Path(".").rglob("best.pt")):
        return str(pt)
    for pt in sorted(Path(".").rglob("last.pt")):
        return str(pt)

    return None

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(weights_path: str):
    model = YOLO(weights_path)
    return model

# ── Resolve weights path ──────────────────────────────────────────────────────
auto_path = find_model_weights()

# Allow user to override via sidebar
with st.sidebar:
    st.markdown("## ⚙️ About")
    st.markdown("""
    **AI Vehicle Recognition System**  
    Detects vehicle type from images, videos, or webcam.

    ---
    **Classes:**
    - 🚌 Bus
    - 🚗 Car  
    - 🏍️ Motorcycle
    - 🚛 Truck

    ---
    **Model:** YOLOv8s-cls  
    **Input size:** 224 × 224 px  
    **Preprocessing:** Grayscale normalisation
    """)

    st.markdown("---")
    st.markdown("### 📂 Model Weights Path")

    if auto_path:
        st.markdown(
            f"<div class='path-found'>✅ Auto-detected:<br><code>{auto_path}</code></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='path-error'>⚠️ Could not auto-detect weights.<br>Paste path manually below.</div>",
            unsafe_allow_html=True
        )

    manual_path = st.text_input(
        "Override weights path (optional)",
        value=auto_path or "runs/vehicle_cls/capstone_v1/weights/best.pt",
        help="Absolute or relative path to your best.pt file"
    )

    st.markdown("---")
    st.markdown("**Confidence Threshold**")
    conf_threshold = st.slider("Min confidence to display", 0, 100, 50, 5, format="%d%%") / 100

# ── Load model (abort with clear message if not found) ───────────────────────
weights_to_use = manual_path.strip()

if not os.path.exists(weights_to_use):
    st.markdown("<div class='main-title'>🚗 Vehicle Recognition System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>YOLOv8s · OpenCV · Streamlit — Bus · Car · Motorcycle · Truck</div>", unsafe_allow_html=True)
    st.divider()
    st.error(
        f"❌ **Model weights not found at:** `{weights_to_use}`\n\n"
        "**How to fix:**\n"
        "1. Make sure you have trained the model (run all cells in your notebook up to Section 5)\n"
        "2. After training, weights are saved at: `runs/vehicle_cls/capstone_v1/weights/best.pt`\n"
        "3. Run `streamlit run app.py` from the **same folder** that contains the `runs/` directory\n"
        "4. Or paste the correct absolute path in the sidebar under **Model Weights Path**\n\n"
        "**Find your weights file:**\n"
        "```bash\n"
        "find . -name 'best.pt'\n"
        "```"
    )
    st.stop()

model = load_model(weights_to_use)
model_classes = [model.names[i] for i in sorted(model.names.keys())]
if model_classes:
    CLASSES = model_classes

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(img_rgb: np.ndarray):
    gray3  = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    result = model.predict(gray3, imgsz=224, verbose=False)[0]
    probs  = result.probs.data.cpu().numpy()
    return probs

# ── Bar chart ─────────────────────────────────────────────────────────────────
def draw_bar_chart(probs):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors  = [CLASS_COLORS.get(c, "#888") for c in CLASSES]
    bars    = ax.barh(CLASSES, probs * 100, color=colors, edgecolor="none", height=0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color='#ccc', fontsize=9)
    ax.set_title("Class Probabilities", color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#ccc', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color='#333', linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, probs * 100):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, color='white')
    fig.tight_layout()
    return fig

# ── Result card HTML ──────────────────────────────────────────────────────────
def result_card(label, conf):
    color = CLASS_COLORS.get(label, "#888")
    icon  = CLASS_ICONS.get(label, "🚗")
    return f"""
    <div class='result-card' style='background:{color}18; border-color:{color}'>
        <div style='font-size:2.2rem'>{icon}</div>
        <div style='font-size:1.8rem; font-weight:700; color:{color}'>{label}</div>
        <div style='font-size:1rem; color:#aaa; margin-top:4px'>
            Confidence: <b style='color:white'>{conf*100:.1f}%</b>
        </div>
    </div>
    """

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='main-title'>🚗 Vehicle Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>YOLOv8s · OpenCV · Streamlit — Bus · Car · Motorcycle · Truck</div>", unsafe_allow_html=True)

# Show which weights are loaded
st.markdown(
    f"<div class='path-found' style='display:inline-block; margin-bottom:12px;'>"
    f"✅ Model loaded: <code>{weights_to_use}</code></div>",
    unsafe_allow_html=True
)
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📷  Image Upload", "🎥  Video Upload", "📸  Webcam Snapshot"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Image
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Drop a vehicle image here", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded:
        img    = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analysing..."):
                t0    = time.time()
                probs = predict(img_np)
                elapsed = time.time() - t0

            top_idx   = int(probs.argmax())
            top_label = CLASSES[top_idx]
            top_conf  = float(probs.max())

            if top_conf >= conf_threshold:
                st.markdown(result_card(top_label, top_conf), unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Top prediction `{top_label}` ({top_conf*100:.1f}%) is below your confidence threshold.")

            st.pyplot(draw_bar_chart(probs))
            st.caption(f"⏱ Inference time: {elapsed*1000:.0f} ms")

            # Top-3 table
            top3 = probs.argsort()[::-1][:3]
            st.markdown("**Top-3 Predictions**")
            for rank, idx in enumerate(top3, 1):
                icon  = CLASS_ICONS.get(CLASSES[idx], "")
                color = CLASS_COLORS.get(CLASSES[idx], "#888")
                st.markdown(
                    f"<span style='color:{color}; font-weight:600'>{rank}. {icon} {CLASSES[idx]}</span>"
                    f" — {probs[idx]*100:.1f}%",
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Video
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        cap   = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.mktemp(suffix="_out.mp4")
        out      = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        st.info(f"📹 {total} frames · {fps:.0f} fps · {w}×{h} px")
        progress = st.progress(0, text="Starting...")
        FONT     = cv2.FONT_HERSHEY_DUPLEX
        frame_n  = 0
        label_counts = {c: 0 for c in CLASSES}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            probs = predict(rgb)
            idx   = int(probs.argmax())
            conf  = float(probs.max())
            label = CLASSES[idx]
            label_counts[label] += 1

            bgr_color = CLASS_COLORS_BGR.get(label, (200, 200, 200))
            text      = f"{label}  {conf*100:.1f}%"

            (tw, th), bl = cv2.getTextSize(text, FONT, 1.1, 2)
            cv2.rectangle(frame, (14, 14), (14 + tw + 24, 14 + th + 20), bgr_color, -1)
            cv2.rectangle(frame, (14, 14), (14 + tw + 24, 14 + th + 20), (255,255,255), 2)
            cv2.putText(frame, text, (26, 14 + th + 10), FONT, 1.1, (0,0,0), 2, cv2.LINE_AA)

            out.write(frame)

            if frame_n % 15 == 0 or frame_n == total:
                progress.progress(min(frame_n / max(total, 1), 1.0),
                                  text=f"Processing frame {frame_n} / {total}  —  {label} {conf*100:.0f}%")

        cap.release()
        out.release()
        progress.progress(1.0, text="✅ Done!")
        os.unlink(tfile.name)

        st.markdown("### 📊 Detection Summary")
        cols = st.columns(len(CLASSES))
        for col, cls in zip(cols, CLASSES):
            pct = label_counts[cls] / max(frame_n, 1) * 100
            col.metric(f"{CLASS_ICONS.get(cls,'')} {cls}", f"{label_counts[cls]} frames", f"{pct:.1f}%")

        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Video", f,
                               file_name="vehicle_output.mp4", mime="video/mp4",
                               use_container_width=True)
        st.video(out_path)
        os.unlink(out_path)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Webcam Snapshot
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    > **How it works:** Click the button below to grab a single frame from your  
    > connected webcam and run the vehicle recognition model on it.  
    > For a continuous live feed, run this app **locally** — cloud servers have no camera.
    """)

    cam_index = st.number_input("Camera index (0 = default webcam)", min_value=0, max_value=5, value=0, step=1)

    if st.button("📸 Capture & Predict"):
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Check that it is connected and not in use by another app.")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                probs = predict(rgb)

                top_idx   = int(probs.argmax())
                top_label = CLASSES[top_idx]
                top_conf  = float(probs.max())
                bgr_color = CLASS_COLORS_BGR.get(top_label, (200,200,200))

                text = f"{top_label}  {top_conf*100:.1f}%"
                FONT = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(text, FONT, 1.2, 2)
                cv2.rectangle(frame, (14,14), (14+tw+24, 14+th+20), bgr_color, -1)
                cv2.rectangle(frame, (14,14), (14+tw+24, 14+th+20), (255,255,255), 2)
                cv2.putText(frame, text, (26, 14+th+10), FONT, 1.2, (0,0,0), 2, cv2.LINE_AA)

                col1, col2 = st.columns([1,1], gap="large")
                with col1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             caption="Webcam Capture", use_container_width=True)
                with col2:
                    if top_conf >= conf_threshold:
                        st.markdown(result_card(top_label, top_conf), unsafe_allow_html=True)
                    else:
                        st.warning(f"Low confidence: {top_conf*100:.1f}% (below threshold)")
                    st.pyplot(draw_bar_chart(probs))
            else:
                st.error("❌ Failed to read a frame from the webcam.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.85rem'>"
    "AI-Based Vehicle Recognition System · YOLOv8s · OpenCV · Streamlit"
    "</div>",
    unsafe_allow_html=True
)
