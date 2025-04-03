# src/filters.py
import cv2

def apply_blur_with_overlay(image, boxes, strength=15):
    result = image.copy()
    for (x, y, w, h) in boxes:
        roi = result[y:y+h, x:x+w]

        k_w = max(3, strength) | 1
        k_h = max(3, strength) | 1
        blurred = cv2.GaussianBlur(roi, (k_w, k_h), 0)

        # Apply blur
        result[y:y+h, x:x+w] = blurred

        # Draw bounding box outline (blue)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 153, 255), thickness=3)
    return result
