# src/detection.py
import cv2
import numpy as np
import os

def detect_faces(image):
    # Get absolute path to current file's directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(base_path, '..'))

    # Model paths (adjusted to Streamlit runtime)
    model_file = os.path.join(root_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    config_file = os.path.join(root_path, "deploy.prototxt")

    if not os.path.exists(model_file) or not os.path.exists(config_file):
        raise FileNotFoundError("Model files not found. Check file paths and locations.")

    # Load DNN model
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes
