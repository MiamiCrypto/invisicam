# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from src.filters import apply_blur
from src.detection import detect_sensitive_regions

st.set_page_config(page_title="InvisiCam – Real Estate Privacy Filter", layout="wide")

st.title("🏠 InvisiCam – Real Estate Privacy Filter")
st.markdown("""
InvisiCam helps blur **faces, people, documents, and personal details** from real estate photos.  
Upload a photo, and we'll auto-detect sensitive regions to protect privacy before listing it online.
""")

# Image upload
uploaded_file = st.file_uploader("📤 Upload a listing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    st.image(image_np, caption="📷 Original Image", use_container_width=True)

    # Detect people and objects to blur
    boxes = detect_sensitive_regions(image_np)

    if not boxes:
        st.warning("No sensitive content detected. Try another image.")
    else:
        blurred_image = apply_blur(image_np, boxes)
        st.image(blurred_image, caption="🔒 Privacy-Protected Image", use_container_width=True)

        # Encode for download
        result_bytes = cv2.imencode('.jpg', cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Download Blurred Image", data=result_bytes, file_name="invisicam_output.jpg")
