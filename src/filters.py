# src/filters.py
import cv2

def apply_blur(image, boxes, strength=15):
    result = image.copy()
    for (x, y, w, h) in boxes:
        roi = result[y:y+h, x:x+w]

        # Adjust kernel size based on blur strength from slider
        k_w = max(3, strength) | 1  # Ensure kernel is odd
        k_h = max(3, strength) | 1

        blurred = cv2.GaussianBlur(roi, (k_w, k_h), 0)
        result[y:y+h, x:x+w] = blurred
    return result
