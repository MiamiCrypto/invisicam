# src/detection.py
from ultralytics import YOLO
import cv2

# Load once at top level
model = YOLO("yolov8n.pt")  # smallest/lightest model

def detect_sensitive_regions(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image_bgr, conf=0.4)[0]  # forward pass

    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        # Filter for persons and wall items
        if class_name in ['person', 'tv', 'tvmonitor', 'picture', 'poster']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes

