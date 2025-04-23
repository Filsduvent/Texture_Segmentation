import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_filters(size=30):
    filters = {}

    # Horizontal (detects vertical edges)
    filters['horizontal'] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Vertical (detects horizontal edges)
    filters['vertical'] = filters['horizontal'].T

    # 45 degree
    filters['45'] =  np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    # 135 degree
    filters['135'] = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])

    # Circular filter (Laplacian of Gaussian)
    filters['circular'] = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Circular (Laplacian of Gaussian)
    #filters['circular'] = cv2.getGaussianKernel(size, size//4) @ cv2.getGaussianKernel(size, size//4).T
    #filters['circular'] = cv2.Laplacian(filters['circular'], cv2.CV_64F)

    return filters


    # Resize and store each filter
    for name, base in base_filters.items():
        if base is None:
            raise ValueError(f"Filter base for '{name}' is None.")
        resized = cv2.resize(base, (ksize, ksize), interpolation=cv2.INTER_LINEAR)
        filters[name] = resized

    return filters


def show_filters(filters):
    plt.figure(figsize=(12, 6))
    for i, (name, f) in enumerate(filters.items()):
        plt.subplot(2, 3, i + 1)
        plt.imshow(f, cmap='gray')
        plt.title(f'{name} filter')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    filters = get_filters()
    show_filters(filters)
