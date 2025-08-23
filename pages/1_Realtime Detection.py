import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Deep learning framework
from ultralytics import YOLO
from sample_utils.download import download_file

st.set_page_config(
    page_title="Realtime Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

# ---------------- Theme ----------------
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"], index=0, key="theme_radio_realtime")
if theme == "Light":
    page_bg = "#F8F9F9"
    primary_color = "#154360"
    text_color = "#2C3E50"
    card_bg = "#FFFFFF"
    shadow = "rgba(0,0,0,0.1)"
    divider_color = "#1F618D"
else:
    page_bg = "#1C1C1C"
    primary_color = "#EAECEE"
    text_color = "#D5DBDB"
    card_bg = "#2C3E50"
    shadow = "rgba(255,255,255,0.05)"
    divider_color = "#5DADE2"

# ---------------- CSS ----------------
st.markdown(f"""
<style>
.stApp {{ background-color: {page_bg}; }}
.section-title {{ font-size:26px; font-weight:700; color:{primary_color}; margin-bottom:12px; text-align:center; }}
.section-text {{ font-size:17px; color:{text_color}; line-height:1.6; margin-bottom:15px; text-align:center; }}
.card {{ background-color:{card_bg}; padding:25px; margin:25px auto; border-radius:14px; box-shadow:0 3px 12px {shadow}; max-width:850px; }}
.footer {{ text-align:center; font-size:16px; color:{text_color}; margin-top:40px; margin-bottom:15px; }}
.custom-divider {{ border:none; height:3px; background-color:{divider_color}; margin:30px auto; width:70%; border-radius:5px; }}
/* Streamlit widget fixes */
[data-baseweb="slider"] * , [data-baseweb="file-uploader"] * , [data-baseweb="checkbox"] * {{
    color:{text_color} !important; background-color:{card_bg} !important; -webkit-text-fill-color:{text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Divider ----------------
def custom_divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<p class="section-title">📷 Road Damage Detection - Realtime</p>', unsafe_allow_html=True)
st.markdown('<p class="section-text">Detect road damage in real-time using USB Webcam. Select the video input device and start inference.</p>', unsafe_allow_html=True)
custom_divider()

# ---------------- Model Download ----------------
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# ---------------- Load Model ----------------
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# ---------------- UI Elements ----------------
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if no damage is detected; increase if false predictions appear.")

show_table = st.checkbox("Show Predictions Table", value=False)

# ---------------- Thread-safe Queue ----------------
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# ---------------- Video Frame Callback ----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori, w_ori = image.shape[:2]
    img_resized = cv2.resize(image, (640, 640))
    
    results = net.predict(img_resized, conf=score_threshold, verbose=False)
    
    detections_list = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for _box in boxes:
            x1, y1, x2, y2 = _box.xyxy[0].astype(int)
            cls_id = int(_box.cls)
            conf = float(_box.conf)
            label = CLASSES[cls_id]
            # Scale boxes back to original image
            x_scale = w_ori / 640
            y_scale = h_ori / 640
            x1 = int(x1 * x_scale)
            x2 = int(x2 * x_scale)
            y1 = int(y1 * y_scale)
            y2 = int(y2 * y_scale)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            detections_list.append(Detection(cls_id, label, conf, np.array([x1, y1, x2, y2])))
    
    if show_table:
        if not result_queue.full():
            result_queue.put(detections_list)
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# ---------------- Start WebRTC ----------------
webrtc_ctx = webrtc_streamer(
    key="realtime-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video":{"width":{"ideal":1280}}, "audio": False},
    async_processing=True
)

# ---------------- Predictions Table ----------------
if show_table:
    labels_placeholder = st.empty()
    if webrtc_ctx.state.playing:
        while True:
            try:
                detections = result_queue.get(timeout=1.0)
                table_data = [{"Class": d.label, "Score": f"{d.score:.2f}", "Box": d.box.tolist()} for d in detections]
                labels_placeholder.table(table_data)
            except queue.Empty:
                break

custom_divider()
st.markdown(f'<p class="footer">🚀 Realtime road damage detection with adjustable confidence threshold.</p>', unsafe_allow_html=True)
