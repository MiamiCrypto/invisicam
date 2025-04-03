# src/filters.py
import cv2

def apply_blur(image, boxes):
    result = image.copy()
    for (x, y, w, h) in boxes:
        roi = result[y:y+h, x:x+w]

        # Calculate kernel size dynamically based on region size
        k_w = max(5, int(w * 0.1)) | 1  # ensure odd kernel size
        k_h = max(5, int(h * 0.1)) | 1

        blurred = cv2.GaussianBlur(roi, (k_w, k_h), 0)
        result[y:y+h, x:x+w] = blurred
    return result
