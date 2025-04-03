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
show_contours = st.sidebar.checkbox("Show Segmentation Outlines", value=False)
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

        # ✅ Accurate manual hex-to-BGR conversion
        r = int(outline_color[1:3], 16)
        g = int(outline_color[3:5], 16)
        b = int(outline_color[5:7], 16)
        bgr_color = (b, g, r)

        if preview_only:
            result = image_np.copy()
            mask_overlay = np.zeros_like(result, dtype=np.uint8)
            for mask in masks:
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (result.shape[1], result.shape[0]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_overlay, contours, -1, bgr_color, thickness=4)
            result = cv2.addWeighted(result, 1, mask_overlay, 1, 0)
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
