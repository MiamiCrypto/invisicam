# src/edges.py
import cv2
import numpy as np

def apply_sobel_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    edge_map = np.uint8(normalized)

    # Convert back to RGB to blend with original
    edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(image, 0.8, edge_rgb, 0.2, 0)

    return blended

