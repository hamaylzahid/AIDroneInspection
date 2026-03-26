import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import pandas as pd
import numpy as np
import time
import os
import gdown

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="Autonomous AI Drone Infrastructure Inspection System",
    layout="wide",
)

# ---------------------------
# DARK UI (optional styling)
# ---------------------------
st.markdown("", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🚁 Drone Mission Control")
CONFIDENCE = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.3)
SHOW_LABELS = st.sidebar.checkbox("Show Labels", True)


MODEL_PATH = "best.pt"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1REzoCDWIqbwcEjYBVno8CggxOwd43tlp"
    gdown.download(url, MODEL_PATH, quiet=False)

# ---------------------------
# HEADER
# ---------------------------
st.title("🚁 Autonomous AI Drone Infrastructure Inspection System")
st.markdown("Real-time infrastructure defect detection using AI")

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
st.success("Model Loaded Successfully")

# ---------------------------
# LOGGER
# ---------------------------
detections_log = []

def log_detection(results, frame_id):
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        for cls_id, conf in zip(class_ids, confs):
            detections_log.append({
                "Frame": frame_id,
                "Class": r.names[int(cls_id)],
                "Confidence": round(float(conf), 2),
                "Severity": "High" if conf > 0.7 else "Medium"
            })

# ---------------------------
# DRAW FUNCTION
# ---------------------------
def draw_boxes(frame, results):
    annotated = frame.copy()
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = map(int, box)

            if conf > 0.7:
                color = (0, 0, 255)      # Red
            elif conf > 0.4:
                color = (0, 165, 255)    # Orange
            else:
                color = (255, 0, 0)      # Blue

            thickness = 2  # thinner box
            font_scale = 0.5  # smaller font
            label = f"{r.names[int(cls_id)]} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if SHOW_LABELS:
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(annotated, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1
                )

    # Smaller detection counter
    cv2.putText(
        annotated,
        f"Detections: {len(r.boxes) if r.boxes is not None else 0}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # smaller than before
        (50, 0, 200),
        2
    )

    return annotated

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload Images or Videos",
    type=["jpg", "jpeg", "png", "jfif", "mp4", "avi"],
    accept_multiple_files=True
)

# ---------------------------
# PROCESSING
# ---------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        st.info(f"Processing: {uploaded_file.name}")

        # ---------------- IMAGE MODE ----------------
        if file_ext in ["jpg", "jpeg", "png", "jfif"]:
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is None:
                    st.error(f"Failed to read {uploaded_file.name}")
                    continue

                results = model(frame, conf=CONFIDENCE)
                annotated = draw_boxes(frame, results)

                log_detection(results, 0)

                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"🎯 {uploaded_file.name}"
                )

            except Exception as e:
                st.error(f"Error: {e}")
                continue

        # ---------------- VIDEO MODE ----------------
        elif file_ext in ["mp4", "avi"]:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                cap = cv2.VideoCapture(tfile.name)

                width = int(cap.get(3))
                height = int(cap.get(4))
                fps = int(cap.get(5))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_path = f"processed_{uploaded_file.name}"

                out = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                progress = st.progress(0)
                frame_id = 0

                while cap.isOpened():
                    start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=CONFIDENCE)
                    annotated = draw_boxes(frame, results)

                    fps_val = 1 / (time.time() - start)
                    cv2.putText(
                        annotated,
                        f"FPS: {int(fps_val)}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    log_detection(results, frame_id)
                    out.write(annotated)

                    frame_id += 1
                    progress.progress(frame_id / total_frames)

                cap.release()
                out.release()

                st.success("Video Processed")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button("Download Video", f, output_path)

            except Exception as e:
                st.error(f"Video processing error: {e}")

# ---------------------------
# DASHBOARD
# ---------------------------
st.markdown("## 📊 Detection Dashboard")

if detections_log:
    df = pd.DataFrame(detections_log)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(df))
    col2.metric("High Severity", len(df[df["Severity"] == "High"]))
    col3.metric("Frames", df["Frame"].nunique())

    st.bar_chart(df["Class"].value_counts())
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Report", csv, "report.csv")

else:
    st.warning("No detections yet")
