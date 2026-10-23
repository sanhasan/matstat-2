import numpy as np

def find_theta_hat(points: np.ndarray) -> np.ndarray:
    means = points.mean(axis=1)
    return means / (1 - means)
