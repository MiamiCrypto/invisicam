# src/detection.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO segmentation model
seg_model = YOLO("yolov8n-seg.pt")

def detect_segmented_masks(image):
    # Convert RGB to BGR for OpenCV/YOLO compatibility
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Run YOLOv8 segmentation
    results = seg_model(image_bgr)[0]

    masks = []
    if results.masks is not None:
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = seg_model.names[cls_id]

            # Filter for relevant privacy-sensitive classes
            if class_name in ["person", "tv", "picture", "poster", "laptop", "book"]:
                mask = results.masks.data[i].cpu().numpy()
                masks.append(mask)

    return masks


