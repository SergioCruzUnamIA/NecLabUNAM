"""
Correlation matrices and hierarchical clustering.

One correlation function used by every view in the app. The two linkage
builders below reproduce what the views draw today (they currently use
different clustering criteria); choosing a single one is a separate step.
"""

import numpy as np
import pandas as pd

CORRELATION_METHODS = ("pearson", "kendall", "spearman")


def correlation_matrix(data, method="pearson"):
    """Pairwise correlation between the columns of `data`.

    Args:
        data: DataFrame, or 2D array of shape (samples, signals).
        method: "pearson", "kendall" or "spearman".

    Returns:
        Square DataFrame (signals x signals).
    """
    method = str(method).lower()
    if method not in CORRELATION_METHODS:
        raise ValueError(f"method must be one of {CORRELATION_METHODS}, got {method!r}")
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data))
    return df.corr(method=method)


def agglomerative_model(X):
    """Full agglomerative (Ward) tree over the rows of X (sklearn model,
    fitted with distance_threshold=0 so every merge is kept)."""
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(np.asarray(X))


def linkage_from_model(model):
    """SciPy linkage matrix from a fitted AgglomerativeClustering model
    (children, distance, number of samples under each merge)."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    return np.column_stack([model.children_, model.distances_, counts]).astype(float)


def agglomerative_linkage(X):
    """Ward linkage over the rows of X (Dendrogram tab, loaded correlation matrix)."""
    return linkage_from_model(agglomerative_model(X))


def abs_correlation_ward_linkage(corr):
    """Ward linkage on the distance 1 - |r| (cell detection window's dendrogram)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    values = corr.values if isinstance(corr, pd.DataFrame) else np.asarray(corr)
    distance = 1 - np.abs(values)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return linkage(squareform(distance, checks=False), method="ward")
