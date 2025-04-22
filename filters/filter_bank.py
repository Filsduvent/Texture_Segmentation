import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_filters(size=15):
    filters = {}

    # Horizontal (detects vertical edges)
    filters['horizontal'] = np.array([[1]*size, [-1]*size])

    # Vertical (detects horizontal edges)
    filters['vertical'] = filters['horizontal'].T

    # 45 degree
    filters['45'] = np.eye(size)

    # 135 degree
    filters['135'] = np.fliplr(np.eye(size))

    # Circular (Laplacian of Gaussian)
    filters['circular'] = cv2.getGaussianKernel(size, size//4) @ cv2.getGaussianKernel(size, size//4).T
    filters['circular'] = cv2.Laplacian(filters['circular'], cv2.CV_64F)

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
