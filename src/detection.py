from ultralytics import YOLO
import cv2
import numpy as np

seg_model = YOLO("yolov8n-seg.pt")

def detect_segmented_masks(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = seg_model(image_bgr)[0]

    masks = []
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        class_name = seg_model.names[class_id]

        if class_name in ["person", "tv", "picture", "poster"]:
            mask = results.masks.data[i].cpu().numpy()
            masks.append(mask)

    return masks

