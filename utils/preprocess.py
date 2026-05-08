import cv2
import numpy as np

def preprocess_image(image_path):
    # 1. Load image directly as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    # 2. Ensure it's white text on a black background
    # If the top-left pixel is bright, we assume it's black text on a white background and invert it.
    if img[0, 0] > 127:
        img = cv2.bitwise_not(img)

    # 3. Apply a binary threshold to remove noise and anti-aliasing (gray pixels)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 4. Find the bounding box of the letter and crop it tightly
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # 5. Make the tightly cropped image square by padding the shorter dimension
    max_side = max(img.shape)
    top = (max_side - img.shape[0]) // 2
    bottom = max_side - img.shape[0] - top
    left = (max_side - img.shape[1]) // 2
    right = max_side - img.shape[1] - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # 6. Resize to 20x20 (This matches the internal standard of EMNIST/MNIST)
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

    # 7. Pad by 4 pixels on all sides to reach the final 28x28 shape
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    # 8. Normalize and reshape for the CNN
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img