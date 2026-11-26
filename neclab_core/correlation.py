"""
Correlation and hierarchical clustering (pipeline stage 7), the single
implementation used by every view: the cell detection window's
correlation analysis, the Multiple Files correlation pane, the Dendrogram
tab and "Load Correlation Matrix".

Clustering always works on the correlation matrix, with
    distance(i, j) = 1 - |r(i, j)|
(0 for perfectly correlated or anti-correlated signals, 1 for
uncorrelated ones) and LINKAGE_METHOD ("ward") to merge clusters. Change
LINKAGE_METHOD to switch every view at once (e.g. "average", the classic
choice for correlation distances; Ward's criterion formally assumes
Euclidean distances).
"""

import numpy as np
import pandas as pd

CORRELATION_METHODS = ("pearson", "kendall", "spearman")
LINKAGE_METHOD = "ward"
LINKAGE_DESCRIPTION = "Ward linkage on 1 − |r|"


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


def correlation_distance(corr):
    """Distance matrix 1 - |r|: symmetric, zero diagonal, clipped to >= 0
    (a loaded matrix may have values slightly outside [-1, 1])."""
    values = corr.to_numpy(dtype=float) if isinstance(corr, pd.DataFrame) else np.asarray(corr, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("A correlation matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError("The correlation matrix has missing values (e.g. a constant signal); "
                         "remove those signals to cluster")
    distance = np.clip(1 - np.abs(values), 0, None)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return distance


def cluster_linkage(corr, method=None):
    """Hierarchical clustering of the signals of a correlation matrix
    (SciPy linkage matrix, for scipy.cluster.hierarchy.dendrogram)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    distance = correlation_distance(corr)
    if distance.shape[0] < 2:
        raise ValueError("At least 2 signals are needed to cluster")
    return linkage(squareform(distance, checks=False), method=method or LINKAGE_METHOD)


def dendrogram_order(link):
    """Leaf order of a linkage (signal indices, left to right in the dendrogram)."""
    from scipy.cluster.hierarchy import leaves_list
    return [int(i) for i in leaves_list(link)]


def linkage_table(link):
    """Linkage matrix as a table: one row per merge (Merge_Step, Child_1,
    Child_2, Distance, Size). Children < n are signals; n + k is merge k."""
    link = np.asarray(link)
    return pd.DataFrame({
        "Merge_Step": np.arange(len(link)),
        "Child_1": link[:, 0].astype(int),
        "Child_2": link[:, 1].astype(int),
        "Distance": link[:, 2],
        "Size": link[:, 3].astype(int),
    })
