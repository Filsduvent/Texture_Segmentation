import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from filters.filter_bank import get_filters

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def generate_scales(img, num_scales=3):
    scales = [img]
    for _ in range(1, num_scales):
        img = cv2.GaussianBlur(img, (5, 5), 1)
        img = cv2.pyrDown(img)  # Downscale by 2
        scales.append(img)
    return scales

def apply_filters(scales, filters):
    filtered = []
    for i, scale in enumerate(scales):
        scale_results = {}
        for name, f in filters.items():
            result = cv2.filter2D(scale, -1, f)
            scale_results[name] = result
        filtered.append(scale_results)
    return filtered

def show_filtered_results(scales, filtered):
    for i, (scale_img, scale_result) in enumerate(zip(scales, filtered)):
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Scale {i+1}', fontsize=16)
        for j, (name, result) in enumerate(scale_result.items()):
            plt.subplot(2, 3, j+1)
            plt.imshow(result, cmap='gray')
            plt.title(name)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Replace this path with any one of your 16 urban images
    img_path = 'data/urban_images/urban_1.jpeg'  # example path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = load_image(img_path)
    scales = generate_scales(image)
    filters = get_filters()
    filtered_results = apply_filters(scales, filters)
    show_filtered_results(scales, filtered_results)
