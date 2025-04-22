from sklearn.cluster import KMeans

def segment_with_kmeans(feature_matrix, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)  # shape: (H * W,)
    return labels
    

def labels_to_image(labels, shape):
    H, W = shape
    segmented_img = labels.reshape((H, W))
    return segmented_img
