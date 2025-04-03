# src/filters.py
import cv2

def apply_blur(image, boxes, ksize=(25, 25)):
    result = image.copy()
    for (x, y, w, h) in boxes:
        region = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(region, ksize, 0)
        result[y:y+h, x:x+w] = blurred
    return result
