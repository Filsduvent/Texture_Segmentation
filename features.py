import numpy as np
import cv2

def extract_texture_features(filtered_results, win_size=7):
    """
    Extracts texture features from a list of dictionaries where each dictionary
    holds filtered responses for a specific scale.

    Args:
        filtered_results (list of dict): Each dict contains directional filter responses.
        win_size (int): Window size for local averaging.

    Returns:
        feature_matrix (np.ndarray): (H*W, num_features)
        original_shape (tuple): (H, W)
    """
    features = []

    if not filtered_results or not isinstance(filtered_results[0], dict):
        raise ValueError("Expected a list of dictionaries with filtered responses")

    # Get target shape from first image in first dict
    first_img = list(filtered_results[0].values())[0]
    target_shape = first_img.shape

    for scale_dict in filtered_results:
        for direction, img in scale_dict.items():
            # Resize if needed
            if img.shape != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

            # Average response in a window
            avg_response = cv2.blur(img, (win_size, win_size))
            features.append(avg_response)

    # Stack features (num_features, H, W)
    feature_stack = np.stack(features, axis=0)

    # Reshape to (H*W, num_features)
    num_features, H, W = feature_stack.shape
    feature_matrix = feature_stack.reshape((num_features, -1)).T

    return feature_matrix, (H, W)
