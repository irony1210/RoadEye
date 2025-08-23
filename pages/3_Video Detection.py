import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from sample_utils.download import download_file

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Road Damage Detection - Video",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Theme Toggle ----------------
theme = st.sidebar.radio(
    "🎨 Select Theme",
    ["Light", "Dark"],
    index=0,
    key="theme_radio_video"
)

# ---------------- Theme Colors ----------------
if theme == "Light":
    page_bg = "#F8F9F9"
    primary_color = "#154360"
    text_color = "#2C3E50"
    card_bg = "#FFFFFF"
    footer_color = "#1B2631"
    shadow = "rgba(0,0,0,0.1)"
    divider_color = "#1F618D"
else:
    page_bg = "#1C1C1C"
    primary_color = "#EAECEE"
    text_color = "#D5DBDB"
    card_bg = "#2C3E50"
    footer_color = "#BDC3C7"
    shadow = "rgba(255,255,255,0.05)"
    divider_color = "#5DADE2"

# ---------------- Custom CSS ----------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {page_bg};
}}
.section-title {{
    font-size: 26px;
    font-weight: 700;
    color: {primary_color};
    margin-bottom: 12px;
    text-align: center;
}}
.section-text {{
    font-size: 17px;
    color: {text_color};
    line-height: 1.6;
    margin-bottom: 15px;
    text-align: center;
}}
.card {{
    background-color: {card_bg};
    padding: 25px;
    margin: 25px auto;
    border-radius: 14px;
    box-shadow: 0 3px 12px {shadow};
    max-width: 850px;
}}
ul, li {{
    color: {text_color};
}}
.footer {{
    text-align: center;
    font-size: 16px;
    color: {footer_color};
    margin-top: 40px;
    margin-bottom: 15px;
}}
.center-img {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 10px;
    border-radius: 10px;
    width: 400px;
}}
.custom-divider {{
    border: none;
    height: 3px;
    background-color: {divider_color};
    margin: 30px auto;
    width: 70%;
    border-radius: 5px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Divider ----------------
def custom_divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<p class="section-title">📹 Road Damage Detection - Video</p>', unsafe_allow_html=True)
st.markdown('<p class="section-text">Upload a video and detect road damage using YOLOv8.</p>', unsafe_allow_html=True)
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

# ---------------- Temporary Folder ----------------
if not os.path.exists('./temp'):
    os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    write_bytesio_to_file(temp_file_input, video_file)
    videoCapture = cv2.VideoCapture(temp_file_input)

    if not videoCapture.isOpened():
        st.error('Error opening the video file')
        return

    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    duration_str = f"{int(duration // 60)}:{int(duration % 60)}"
    st.write(f"Video Duration: {duration_str}  |  Width x Height: {width}x{height}  |  FPS: {fps}")

    inferenceBarText = "Performing inference on video..."
    inferenceBar = st.progress(0, text=inferenceBarText)
    imageLocation = st.empty()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2writer = cv2.VideoWriter(temp_file_infer, fourcc, fps, (width, height))

    frame_counter = 0
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_AREA)
        results = net.predict(image_resized, conf=score_threshold)

        annotated_frame = results[0].plot()
        frame_pred = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2writer.write(cv2.cvtColor(frame_pred, cv2.COLOR_RGB2BGR))
        imageLocation.image(frame_pred)

        frame_counter += 1
        inferenceBar.progress(frame_counter / frame_count, text=inferenceBarText)

    videoCapture.release()
    cv2writer.release()
    inferenceBar.empty()

    st.success("Video Processed!")

    col1, col2 = st.columns(2)
    with col1:
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                use_container_width=True
            )
    with col2:
        if st.button('Restart App', use_container_width=True, type="primary"):
            st.experimental_rerun()

# ---------------- Upload Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

# Custom labels (no blur)
st.markdown(f'<p class="section-text">Choose Video File</p>', unsafe_allow_html=True)
video_file = st.file_uploader(
    "",
    type=".mp4",
    disabled=st.session_state.runningInference
)

st.markdown(f'<p class="section-text">Confidence Threshold</p>', unsafe_allow_html=True)
score_threshold = st.slider(
    "",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    disabled=st.session_state.runningInference
)

st.markdown(f'<p class="section-text">Video limit: 1GB. Resize or cut your video if larger.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Run Inference ----------------
if video_file is not None:
    if st.button(
        'Process Video',
        use_container_width=True,
        disabled=st.session_state.runningInference,
        type="secondary",
        key="processing_button"
    ):
        st.warning(f"Processing Video {video_file.name}")
        processVideo(video_file, score_threshold)

custom_divider()
st.markdown(f'<p class="footer">🚀 Detect road damage from videos with ease!</p>', unsafe_allow_html=True)
