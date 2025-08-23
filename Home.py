import streamlit as st
import base64

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Road Damage Detection App",
    page_icon="🛣️",
    layout="wide"
)

# ---------------- Theme Toggle ----------------
theme = st.sidebar.radio(
    "🎨 Select Theme",
    ["Light", "Dark"],
    index=0,
    key="theme_radio"  # unique key to avoid duplicate ID error
)

# Light & Dark mode styles
if theme == "Light":
    page_bg = "#F8F9F9"
    primary_color = "#154360"
    text_color = "#2C3E50"
    card_bg = "#FFFFFF"
    footer_color = "#1B2631"
    shadow = "rgba(0,0,0,0.1)"
    divider_color = "#1F618D"
else:  # Dark theme
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
/* Page background */
.stApp {{
    background-color: {page_bg};
}}

/* Headings */
.section-title {{
    font-size: 26px;
    font-weight: 700;
    color: {primary_color};
    margin-bottom: 12px;
    text-align: center;
}}

/* Paragraphs */
.section-text {{
    font-size: 17px;
    color: {text_color};
    line-height: 1.6;
    margin-bottom: 15px;
    text-align: center;
}}

/* Cards */
.card {{
    background-color: {card_bg};
    padding: 25px;
    margin: 25px auto;
    border-radius: 14px;
    box-shadow: 0 3px 12px {shadow};
    max-width: 850px;
}}

/* Lists */
ul, li {{
    color: {text_color}; /* fix blurry lists in dark mode */
}}

/* Footer */
.footer {{
    text-align: center;
    font-size: 16px;
    color: {footer_color};
    margin-top: 40px;
    margin-bottom: 15px;
}}

/* Custom Divider */
.custom-divider {{
    border: none;
    height: 3px;
    background-color: {divider_color};
    margin: 30px auto;
    width: 70%;
    border-radius: 5px;
}}

/* Centered image */
.center-img {{
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 10px;
    border-radius: 10px;
    width: 400px;  /* control image size */
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Divider ----------------
def custom_divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<p class="section-title">🚗 Road Damage Detection Application</p>', unsafe_allow_html=True)
st.markdown('<p class="section-text">Enhancing road safety with AI-powered road damage detection.</p>', unsafe_allow_html=True)
custom_divider()

# ---------------- About Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">About the App</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-text">Our Road Damage Detection App uses the <strong>YOLOv8 deep learning model</strong> trained on the CRDDC2022 dataset. It automatically detects and categorizes road damage, helping authorities maintain roads efficiently.</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Why Needed ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Why This Project is Needed</p>', unsafe_allow_html=True)
st.markdown(
    """
<p class="section-text">
Maintaining road infrastructure is challenging. Manual inspections are slow and error-prone.  
This project automates road damage detection, allowing authorities to:
<ul>
<li>Quickly identify and prioritize repairs</li>
<li>Improve road safety for drivers and pedestrians</li>
<li>Reduce maintenance costs and improve efficiency</li>
</ul>
</p>
""",
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Road Damage Types ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Road Damage Types</p>', unsafe_allow_html=True)

# Encode local image to base64 for HTML display
image_path = "./resource/all_cracks.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()
    encoded = base64.b64encode(image_bytes).decode()

# Caption color based on theme
caption_color = "#2C3E50" if theme == "Light" else "#D5DBDB"

# Display centered image with readable caption
st.markdown(f"""
<div style="text-align:center">
    <img src="data:image/png;base64,{encoded}" class="center-img">
    <p style="color:{caption_color}; margin-top:5px;">Longitudinal, Transverse, Alligator Cracks, and Potholes</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="section-text">The model detects four major road damage types: Longitudinal Crack, Transverse Crack, Alligator Crack, and Potholes.</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- How to Use ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">How to Use</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-text">Choose input from the sidebar: <strong>Realtime Webcam</strong>, <strong>Video</strong>, or <strong>Image</strong>. The model detects and classifies road damage in real time. Results can be exported for further analysis.</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Dataset & License ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Dataset & License</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-text">- Dataset: CRDDC2022<br>'
    '- YOLOv8 License: Provided by <a href="https://github.com/ultralytics/ultralytics">Ultralytics</a><br>'
    '- Framework: Built using <a href="https://streamlit.io/">Streamlit</a></p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

custom_divider()

# ---------------- Footer ----------------
st.markdown('<p class="footer">🚀 Try it now! Select your input from the sidebar and start detecting road damage instantly.</p>', unsafe_allow_html=True)
