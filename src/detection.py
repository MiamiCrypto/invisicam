from ultralytics import YOLO
import cv2
import numpy as np
import os

yolo_model = YOLO("yolov8n.pt")

def detect_sensitive_regions(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = yolo_model(image_bgr, conf=0.4)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = yolo_model.names[cls_id]
        if class_name in ['person', 'tv', 'tvmonitor']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def detect_faces_only(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(base_path, '..'))
    model_file = os.path.join(root_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    config_file = os.path.join(root_path, "deploy.prototxt")

    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def detect_hybrid(image):
    boxes = detect_sensitive_regions(image)
    boxes += detect_faces_only(image)
    return boxes

