from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering


def _image_to_features(img: Image.Image, use_xy: bool = True):
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
    h, w, _ = arr.shape

    X_rgb = arr.reshape(-1, 3) / 255.0

    if not use_xy:
        return X_rgb, h, w

    ys, xs = np.mgrid[0:h, 0:w]
    X_xy = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
    X_xy[:, 0] /= max(w - 1, 1)
    X_xy[:, 1] /= max(h - 1, 1)

    X = np.concatenate([X_rgb, X_xy], axis=1)  # (H*W, 5)
    return X, h, w


def _labels_to_centers_rgb(img: Image.Image, labels: np.ndarray, n_clusters: int):
    """
    Calcule une couleur RGB par cluster à partir de la moyenne des pixels originaux.
    Avantage: rendu plus cohérent qu'une palette aléatoire.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
    pixels = arr.reshape(-1, 3)  # (N,3)

    centers = np.zeros((n_clusters, 3), dtype=np.float32)
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            centers[c] = pixels[mask].mean(axis=0)
        else:
            centers[c] = 0.0
    centers = np.clip(centers, 0.0, 1.0)
    return centers


def segment_agglomerative(
    img: Image.Image,
    n_clusters: int = 6,
    use_xy: bool = True,
    linkage: str = "ward",
) -> Image.Image:
    """
    Segmentation non supervisée par clustering hiérarchique (Agglomerative).
    Note: peut être plus lent que KMeans/GMM sur grandes images.
    """
    X, h, w = _image_to_features(img, use_xy=use_xy)

    agg = AgglomerativeClustering(n_clusters=int(n_clusters), linkage=linkage)
    labels = agg.fit_predict(X)

    centers_rgb = _labels_to_centers_rgb(img, labels, n_clusters=int(n_clusters))
    out = centers_rgb[labels].reshape(h, w, 3)
    out = (out * 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
