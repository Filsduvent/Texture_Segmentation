# utils.py or visualization.py

import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_segmented_image(segmented_img, title='Segmented Image'):
    plt.figure(figsize=(6, 6))
    plt.imshow(segmented_img, cmap='tab10')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_segmented_image(segmented_img, path='data/results/segmented.png'):
    norm_img = (segmented_img / segmented_img.max() * 255).astype(np.uint8)
    cv2.imwrite(path, norm_img)
