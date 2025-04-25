import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from filters.filter_bank import get_filters
from features import extract_texture_features
from clustering import apply_kmeans
from utils import show_segmented_image, show_image_Colored_and_gray


if not os.path.exists('data/output'):
    os.makedirs('data/output')
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
        
def save_filtered_results(scales, filtered, img_name):
    for i, (scale_img, scale_result) in enumerate(zip(scales, filtered)):
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Scale {i+1}', fontsize=16)
        for j, (name, result) in enumerate(scale_result.items()):
            plt.subplot(2, 3, j+1)
            plt.imshow(result, cmap='gray')
            plt.title(name)
            plt.axis('off')
        
        # Sauvegarder au lieu d'afficher
        base_name = os.path.splitext(img_name)[0]
        output_path = f"data/output/{base_name}_scale_{i+1}_filters.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved filtered results to {output_path}")

if __name__ == "__main__":
  
    images_dir = "data/urban_images/"

    images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpeg' or 'jpg')])

    for img_name in images:
        print(img_name)
        path = os.path.join(images_dir, img_name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


        # image = load_image(img_path)
        scales = generate_scales(image)
        filters = get_filters()
        filtered_results = apply_filters(scales, filters)
        # show_filtered_results(scales, filtered_results)
        save_filtered_results(scales, filtered_results, img_name)

        print("Type of filtered_results:", type(filtered_results))
        print("Length:", len(filtered_results))
        print("First element type:", type(filtered_results[0]) if len(filtered_results) > 0 else "Empty list")


        feature_matrix, (H, W) = extract_texture_features(filtered_results, win_size=32)

        #feature_matrix, (H, W) = extract_texture_features(filtered_results)
        print("Feature matrix shape:", feature_matrix.shape)  # Expect: (num_pixels, num_features)


        # Step 1: K-means clustering
        #labels = segment_with_kmeans(feature_matrix, n_clusters=3)

        # Step 2: Reshape to image
        #segmented_img = labels_to_image(labels, (H, W))

        segmented_image_color, segmented_image_gray = apply_kmeans(feature_matrix,image.shape,n_clusters = 3)

        # show_image_Colored_and_gray(image,segmented_image_color, segmented_image_gray, img_name)
        
        # Save img segmentation results
        base_name = os.path.splitext(img_name)[0]
        
        # original Image
        cv2.imwrite(f"data/output/{base_name}_original.png", image)
        
        # color image segmented
        cv2.imwrite(f"data/output/{base_name}_segmented_color.png", cv2.cvtColor(segmented_image_color, cv2.COLOR_RGB2BGR))
        
        # raw image segmented
        cv2.imwrite(f"data/output/{base_name}_segmented_gray.png", segmented_image_gray)
        
        print(f"Saved segmentation results for {img_name}")

print("All processing complete. Results saved in data/output/ directory.")

        # Step 3: Visualize and Save
       # show_segmented_image(segmented_img)
        #save_segmented_image(segmented_img, path='data/results/segmented_kmeans.png')

        # Build Gabor filters and apply them
        #gabor_filters = build_filters_Gabor(ksize=15)
        #filtered_results = apply_filters_to_scales(gray_image, gabor_filters, scales=3)



