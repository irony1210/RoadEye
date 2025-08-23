import os
import logging
from pathlib import Path
from typing import NamedTuple, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

from ultralytics import YOLO
from sample_utils.download import download_file

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Road Damage Detection - Image",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Theme Toggle ----------------
theme = st.sidebar.radio(
    "🎨 Select Theme",
    ["Light", "Dark"],
    index=0,
    key="theme_radio_detection"
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
    table_header_bg = "#D6EAF8"
    table_text_color = "#154360"
else:
    page_bg = "#1C1C1C"
    primary_color = "#EAECEE"
    text_color = "#D5DBDB"
    card_bg = "#2C3E50"
    footer_color = "#BDC3C7"
    shadow = "rgba(255,255,255,0.05)"
    divider_color = "#5DADE2"
    table_header_bg = "#2C3E50"
    table_text_color = "#EAECEE"

# ---------------- Custom CSS ----------------
st.markdown(f"""
<style>
.stApp {{ background-color: {page_bg}; }}
.section-title {{ font-size: 26px; font-weight: 700; color: {primary_color}; margin-bottom: 12px; text-align: center; }}
.section-text {{ font-size: 17px; color: {text_color}; line-height: 1.6; margin-bottom: 15px; text-align: center; }}
.card {{ background-color: {card_bg}; padding: 25px; margin: 25px auto; border-radius: 14px; box-shadow: 0 3px 12px {shadow}; max-width: 850px; }}
ul, li {{ color: {text_color}; }}
.footer {{ text-align: center; font-size: 16px; color: {footer_color}; margin-top: 40px; margin-bottom: 15px; }}
.center-img {{ display: block; margin-left: auto; margin-right: auto; margin-bottom: 10px; border-radius: 10px; width: 400px; }}
.custom-divider {{ border: none; height: 3px; background-color: {divider_color}; margin: 30px auto; width: 70%; border-radius: 5px; }}

/* Fix dark mode for Streamlit widgets */
[data-baseweb="file-uploader"] *,
[data-baseweb="slider"] *,
[data-baseweb="checkbox"] * {{
    color: {text_color} !important;
    background-color: {card_bg} !important;
    -webkit-text-fill-color: {text_color} !important;
}}
[data-baseweb="slider"] input[type="range"] {{
    opacity: 1 !important;
}}

/* Fix table text sharpness in dark mode */
.stDataFrame table {{
    color: {table_text_color} !important;
    background-color: {card_bg} !important;
}}
.stDataFrame thead tr th {{
    background-color: {table_header_bg} !important;
    color: {table_text_color} !important;
    font-weight: bold;
}}
.stDataFrame tbody tr td {{
    color: {table_text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Divider ----------------
def custom_divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<p class="section-title">📷 Road Damage Detection - Image</p>', unsafe_allow_html=True)
st.markdown('<p class="section-text">Upload an image and detect road damage using YOLOv8.</p>', unsafe_allow_html=True)
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

# ---------------- Upload Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Upload Image</p>', unsafe_allow_html=True)
st.markdown('<p class="section-text">Upload your road image for detection.</p>', unsafe_allow_html=True)

image_file = st.file_uploader("Choose Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.markdown('<p class="section-text">Adjust the threshold to control detection sensitivity.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Process Image ----------------
if image_file is not None:
    image = Image.open(image_file)
    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]

    # Perform inference
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    # Annotate image
    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # ---------------- Display Original and Predictions ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Original vs Prediction</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-text">Original Image</p>', unsafe_allow_html=True)
        st.image(_image, use_container_width=True)

    with col2:
        st.markdown('<p class="section-text">Predicted Image</p>', unsafe_allow_html=True)
        st.image(_image_pred, use_container_width=True)

        # Download prediction
        buffer = BytesIO()
        Image.fromarray(_image_pred).save(buffer, format="PNG")
        _img_bytes = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_img_bytes,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )

    # ---------------- Predictions Table ----------------
    if st.checkbox("Show Predictions Table"):
        predictions_list = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls)
                label = CLASSES[class_id]
                score = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                predictions_list.append({
                    "Class": label,
                    "Score": round(score, 3),
                    "Box": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                })
        if predictions_list:
            st.table(predictions_list)
        else:
            st.info("No detections found.")

    st.markdown('</div>', unsafe_allow_html=True)

custom_divider()

# ---------------- Footer ----------------
st.markdown(f'<p class="footer">🚀 Detect road damage with ease. Adjust confidence threshold and analyze your images instantly!</p>', unsafe_allow_html=True)
