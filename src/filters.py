# src/filters.py
import cv2
import numpy as np

def apply_blur_with_mask_overlay(image, masks, strength=15, draw_outline=True, outline_color=(0, 153, 255)):
    result = image.copy()
    outline_overlay = np.zeros_like(image, dtype=np.uint8)

    for mask in masks:
        mask = (mask > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        mask_3ch = cv2.merge([mask, mask, mask])

        # Apply blur only to masked regions
        k = max(3, strength) | 1
        blurred = cv2.GaussianBlur(result, (k, k), 0)
        result = np.where(mask_3ch == 1, blurred, result)

        if draw_outline:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outline_overlay, contours, -1, outline_color, thickness=4)

    # Blend result with colored outline overlay
    result = cv2.addWeighted(result, 1, outline_overlay, 1, 0)
    return result


