# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from src.filters import apply_blur
from src.detection import (
    detect_sensitive_regions,
    detect_faces_only,
    detect_hybrid
)
from src.edges import apply_sobel_edges

# Streamlit layout settings
st.set_page_config(page_title="InvisiCam – Real Estate Privacy Filter", layout="centered")

# Sidebar controls
st.sidebar.title("🛠 Features")
detection_mode = st.sidebar.selectbox(
    "Detection Mode",
    ["Smart (YOLOv8)", "Framed Faces (DNN)", "Hybrid (YOLO + DNN)"]
)

blur_strength = st.sidebar.slider("Blur Intensity", min_value=5, max_value=51, value=15, step=2)
apply_edge_overlay = st.sidebar.checkbox("Overlay Sobel Edge Detection")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with 💡 for real estate listing privacy.")

# App title and intro
st.title("🏠 InvisiCam – Real Estate Privacy Filter")
st.markdown("""
InvisiCam helps blur **faces, people, and personal details** from real estate listing photos.  
Upload a photo and choose a detection mode — we'll protect sensitive regions before you publish it online.
""")

# Image upload interface
uploaded_file = st.file_uploader("📄 Upload a listing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    st.image(image_np, caption="📷 Original Image", use_container_width=True)

    # Choose detection logic based on selected mode
    if detection_mode == "Smart (YOLOv8)":
        boxes = detect_sensitive_regions(image_np)
    elif detection_mode == "Framed Faces (DNN)":
        boxes = detect_faces_only(image_np)
    else:
        boxes = detect_hybrid(image_np)

    if not boxes:
        st.warning("No sensitive content detected. Try another image or mode.")
    else:
        # Apply blur
        blurred_image = apply_blur(image_np, boxes, strength=blur_strength)

        # Optional: Apply Sobel edge detection overlay
        if apply_edge_overlay:
            blurred_image = apply_sobel_edges(blurred_image)

        # Display result
        st.image(blurred_image, caption="🔒 Privacy-Protected Image", use_container_width=True)

        # Prepare downloadable image
        result_bytes = cv2.imencode('.jpg', cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Download Blurred Image", data=result_bytes, file_name="invisicam_output.jpg")
