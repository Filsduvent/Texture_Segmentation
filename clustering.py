from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def apply_kmeans(features, original_shape, n_clusters=3):
    flat_features = features.reshape(-1, features.shape[-1])
    flat_features = StandardScaler().fit_transform(flat_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(flat_features)
    
    # Segmentation en couleur
    segmented_colored = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
    for idx, label in enumerate(unique_labels):
        segmented_colored[labels.reshape(original_shape) == label] = colors[idx]
    
    # Segmentation en niveaux de gris
    segmented_gray = labels.reshape(original_shape)
    segmented_gray = (segmented_gray / segmented_gray.max()) * 255
    segmented_gray = segmented_gray.astype(np.uint8)
    
    return segmented_colored, segmented_gray