# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from src.filters import apply_blur
from src.detection import detect_faces

st.set_page_config(page_title="InvisiCam", layout="centered")

# Sidebar and header
st.title("🏠 InvisiCam – Real Estate Privacy Filter")
st.markdown("""
InvisiCam helps **blur faces, documents, and personal details** from real estate photos. 
Upload a photo, and we'll auto-detect sensitive regions to protect privacy before listing it online.
""")

uploaded_file = st.file_uploader("📤 Upload a listing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    image_np = np.array(image.convert("RGB"))

    # Detect faces or regions
    boxes = detect_faces(image_np)

    if not boxes:
        st.warning("No faces detected! Try another image.")
    else:
        # Apply blur
        blurred_image = apply_blur(image_np, boxes)

        st.image(blurred_image, caption="🔒 Privacy Protected Image", use_column_width=True)

        # Convert and offer download
        result_bytes = cv2.imencode('.jpg', cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Download Blurred Image", data=result_bytes, file_name="invisicam_output.jpg")
