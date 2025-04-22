import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from filters.filter_bank import get_filters
from features import extract_texture_features
from clustering import segment_with_kmeans, labels_to_image
from utils import show_segmented_image, save_segmented_image


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
    img_path = 'data/urban_images/urban_9.jpeg'  # example path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = load_image(img_path)
    scales = generate_scales(image)
    filters = get_filters()
    filtered_results = apply_filters(scales, filters)
    show_filtered_results(scales, filtered_results)

    print("Type of filtered_results:", type(filtered_results))
    print("Length:", len(filtered_results))
    print("First element type:", type(filtered_results[0]) if len(filtered_results) > 0 else "Empty list")


    feature_matrix, (H, W) = extract_texture_features(filtered_results, win_size=7)

    #feature_matrix, (H, W) = extract_texture_features(filtered_results)
print("Feature matrix shape:", feature_matrix.shape)  # Expect: (num_pixels, num_features)


# Step 1: K-means clustering
labels = segment_with_kmeans(feature_matrix, n_clusters=3)

# Step 2: Reshape to image
segmented_img = labels_to_image(labels, (H, W))

# Step 3: Visualize and Save
show_segmented_image(segmented_img)
save_segmented_image(segmented_img, path='data/results/segmented_kmeans.png')


