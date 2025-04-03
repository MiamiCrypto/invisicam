# 🏠 InvisiCam – Real Estate Privacy Filter

**InvisiCam** is an AI-powered Streamlit web app that helps real estate agents, hosts, and homeowners protect sensitive content in listing photos. It automatically detects and blurs people, portraits, and private details using state-of-the-art image segmentation.

---

## 🎯 Purpose

Real estate listings often include people, personal photos, or wall-mounted portraits that may compromise privacy. InvisiCam protects sellers and residents by smartly detecting and obscuring such content before publication.

---

## 🚀 Features

- ✅ **Smart segmentation with YOLOv8-seg** (Ultralytics)
- 🎨 **Custom outline color** for segmentation masks
- 📦 **Streamlit sidebar controls** for interactivity:
  - Blur strength adjustment
  - Toggle segmentation outlines
  - “Preview-only” mode (no blur)
- 📸 Supports **JPG, JPEG, PNG** formats
- 💾 One-click image download of privacy-protected output

---

## 🧠 How It Works

1. Upload a property photo.
2. The app runs **YOLOv8 segmentation** to detect:
   - People
   - Wall-mounted frames (pictures, posters, TVs)
3. Selected regions are:
   - Blurred (with adjustable strength)
   - Optionally outlined with your selected color
4. Download the result instantly.

---

## 🛠 Tech Stack

| Component         | Tool / Library            |
|------------------|---------------------------|
| Frontend / UI    | [Streamlit](https://streamlit.io) |
| Image Processing | [OpenCV](https://opencv.org/)     |
| Deep Learning    | [Ultralytics YOLOv8](https://docs.ultralytics.com) |
| Model Used       | `yolov8n-seg.pt` (nano segmentation) |
| Visualization    | Pillow, NumPy             |

---

## 📂 Folder Structure

