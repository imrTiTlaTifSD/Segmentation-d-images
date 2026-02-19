from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def _image_to_features(img: Image.Image, use_xy: bool = True) -> tuple[np.ndarray, int, int]:
    """
    Convertit une image en matrice de features pour le clustering.
    - RGB normalisé (0..1)
    - optionnel: coordonnées (x,y) normalisées (0..1)
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)  
    h, w, _ = arr.shape

    X_rgb = arr.reshape(-1, 3) / 255.0 

    if not use_xy:
        return X_rgb, h, w

    ys, xs = np.mgrid[0:h, 0:w]
    X_xy = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
    X_xy[:, 0] /= max(w - 1, 1)
    X_xy[:, 1] /= max(h - 1, 1)

    X = np.concatenate([X_rgb, X_xy], axis=1)  
    return X, h, w


def segment_kmeans(
    img: Image.Image,
    n_clusters: int = 6,
    use_xy: bool = True,
    random_state: int = 42,
) -> Image.Image:
    """
    Segmentation non supervisée par KMeans.
    Retourne une image où chaque pixel est coloré selon son cluster.
    """
    X, h, w = _image_to_features(img, use_xy=use_xy)

    km = KMeans(n_clusters=int(n_clusters), n_init="auto", random_state=int(random_state))
    labels = km.fit_predict(X)  


    centers = km.cluster_centers_
    centers_rgb = centers[:, :3]  
    centers_rgb = np.clip(centers_rgb, 0.0, 1.0)

    out = centers_rgb[labels].reshape(h, w, 3)
    out = (out * 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
