import numpy as np


def aggregate(x: np.array, y: np.array):
    """
    Aggregate data by grouping it into bins and calculating statistics.

    If there are at least two data points per "x", we take each unique value as a bin.
    Otherwise, we use the Freedman-Diaconis rule to determine the number of bins.

    Parameters:
    x (np.array): The input data array.
    y (np.array): The weights or values associated with each data point.

    Returns:
        x_rep (np.array): The representative x-values for each bin.
        mean (np.array): The mean values for each bin.
        std (np.array): The standard deviation values for each bin.
    """

    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)

    if u.size > 0 and np.all(counts >= 2):
        nbins = u.size
        count = counts.astype(int)
        sum_y = np.bincount(inv, weights=y, minlength=nbins)
        sum_ysq = np.bincount(inv, weights=y * y, minlength=nbins)
        left = u
        right = u
        edges = np.r_[u, u[-1]]
        x_rep = u
    else:
        edges = np.histogram_bin_edges(x, bins="fd")
        nbins = len(edges) - 1
        idx = np.digitize(x, edges, right=True) - 1
        idx[idx == nbins] = nbins - 1
        valid = (idx >= 0) & (idx < nbins)
        idx = idx[valid]
        y = y[valid]
        count = np.bincount(idx, minlength=nbins)
        sum_y = np.bincount(idx, weights=y, minlength=nbins)
        sum_ysq = np.bincount(idx, weights=y * y, minlength=nbins)
        left = edges[:-1]
        right = edges[1:]
        x_rep = 0.5 * (left + right)

    mean = np.divide(sum_y, count, out=np.full(nbins, np.nan), where=count > 0)
    var_num = sum_ysq - count * mean**2
    var = np.divide(var_num, count - 1, out=np.full(nbins, np.nan), where=count > 1)
    std = np.sqrt(np.maximum(var, 0.0))

    return x_rep, mean, std
