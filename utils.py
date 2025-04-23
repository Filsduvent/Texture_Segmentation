# utils.py or visualization.py

import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_segmented_image(segmented_img, title='Segmented Image'):
    plt.figure(figsize=(6, 6))
    plt.imshow(segmented_img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_segmented_image(segmented_img, path='data/results/segmented.png'):
    norm_img = (segmented_img / segmented_img.max() * 255).astype(np.uint8)
    cv2.imwrite(path, norm_img)


def show_image_Colored_and_gray(image, seg_color_all, seg_gray_all, img_name):
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[1].imshow(seg_color_all)
    axes[1].set_title("Global Seg Color")
    axes[2].imshow(seg_gray_all, cmap='gray')
    axes[2].set_title("Global Seg Gray")
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"Résultats globaux - {img_name}", y=1.05)
    plt.tight_layout()
    plt.show()