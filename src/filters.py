# src/filters.py
def apply_blur(image, boxes, strength=15):
    result = image.copy()
    for (x, y, w, h) in boxes:
        roi = result[y:y+h, x:x+w]

        # Scale blur size based on user input (strength)
        k_w = max(3, strength) | 1
        k_h = max(3, strength) | 1

        blurred = cv2.GaussianBlur(roi, (k_w, k_h), 0)
        result[y:y+h, x:x+w] = blurred
    return result
