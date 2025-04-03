# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from src.filters import apply_blur_with_mask_overlay
from src.detection import detect_segmented_masks

# Streamlit layout settings
st.set_page_config(page_title="InvisiCam – Real Estate Privacy Filter", layout="centered")

# Sidebar controls
st.sidebar.title("🛠 Features")
st.sidebar.info(
    "Don't want to share personal info in listing photos?\n"
    "We blur faces, people, and frames using AI-powered segmentation."
)

blur_strength = st.sidebar.slider("Blur Intensity", min_value=5, max_value=51, value=15, step=2)
show_contours = st.sidebar.checkbox("Show Segmentation Outlines", value=False)  # Default to off

# Optional: pick a color for the outlines
outline_color = st.sidebar.color_picker("Pick Outline Color", value="#0099ff")

preview_only = st.sidebar.checkbox("Preview Regions Only (No Blur)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Created with 💡 for real estate listing privacy.")

# App title and instructions
st.title("🏠 InvisiCam – Real Estate Privacy Filter")
st.markdown("""
InvisiCam helps blur **people, faces, and wall-mounted portraits** from real estate listing photos using modern segmentation.
Upload a photo, choose your settings, and we'll apply smart privacy filters for you.
""")

# Image upload
uploaded_file = st.file_uploader("📄 Upload a listing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    st.image(image_np, caption="📷 Original Image", use_container_width=True)

    # Run segmentation
    masks = detect_segmented_masks(image_np)

    if not masks:
        st.warning("No sensitive content detected. Try another image.")
    else:
        st.success(f"Blurred {len(masks)} sensitive region(s).")

        # Fix color conversion from hex to BGR
        hex_color = outline_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

        if preview_only:
            result = image_np.copy()
            mask_overlay = np.zeros_like(result, dtype=np.uint8)
            for mask in masks:
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (result.shape[1], result.shape[0]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, bgr_color, thickness=4)
        else:
            result = apply_blur_with_mask_overlay(
                image_np,
                masks,
                strength=blur_strength,
                draw_outline=show_contours,
                outline_color=bgr_color
            )

        st.image(result, caption="🔒 Privacy-Protected Image", use_container_width=True)

        result_bytes = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📅 Download Blurred Image", data=result_bytes, file_name="invisicam_output.jpg")

