# src/filters.py
import cv2
import numpy as np

def apply_blur_by_mask(image, masks, blur_strength=15):
    result = image.copy()

    for mask in masks:
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Create a 3-channel binary mask
        mask_3ch = cv2.merge([mask, mask, mask])

        # Blur entire image
        k = max(3, blur_strength) | 1
        blurred = cv2.GaussianBlur(result, (k, k), 0)

        # Combine: keep original where mask = 0, blurred where mask = 1
        result = np.where(mask_3ch == 255, blurred, result)

        # Optional: draw the mask outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 153, 255), 3)

    return result
