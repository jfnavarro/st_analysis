"""
Different functions for analysis for ST data
"""

from matplotlib import colors as mpcolors
from scipy.special import loggamma
import numpy as np


def linear_conv(value, old_min, old_max, new_min, new_max):
    """
    A simple linear conversion of one value from one scale to another
    Returns:
        Value in the new scale
    """
    return ((value - old_min) / (old_max - old_min)) * ((new_max - new_min) + new_min)


def weighted_color(n_classes, probs, n_bins=100):
    """
    Compute a weighted sum probability given
    a list of probabalities, a number of classes
    and a number of bins
    Returns:
        Weighted probability value
    """
    n_classes = float(len(n_classes) - 1)
    l = 1.0 / n_bins
    h = 1 - l
    p = 0.0
    for i, prob in enumerate(probs):
        wi = linear_conv(float(i), 0.0, n_classes, h, l)
        p += abs(prob * wi)
    return p


def composite_colors(colors, probs):
    """
    Merge a list of colors (rgba)
    using a list of probabilities
    Returns:
        Merged color (rgba)
    """
    assert len(colors) == len(probs)
    merged_color = [0.0, 0.0, 0.0, 1.0]
    for prob, color in zip(probs, colors):
        new_color = mpcolors.colorConverter.to_rgba(color)
        merged_color[0] = (new_color[0] - merged_color[0]) * prob + merged_color[0]
        merged_color[1] = (new_color[1] - merged_color[1]) * prob + merged_color[1]
        merged_color[2] = (new_color[2] - merged_color[2]) * prob + merged_color[2]
    return merged_color


def coord_to_rgb(x_p, y_p, z_p=None):
    """
    Computes a list of colors (rgb) based on the coordinates
    space given as input (2D or 3D)
    """
    labels_colors = list()
    x_max = max(x_p)
    x_min = min(x_p)
    y_max = max(y_p)
    y_min = min(y_p)
    WITH_3D = z_p is not None
    z_max = max(z_p) if WITH_3D else None
    z_min = min(z_p) if WITH_3D else y_min
    z_p = z_p if WITH_3D else np.ones(len(x_p))
    for x, y, z in zip(x_p, y_p, z_p):
        r = linear_conv(x, x_min, x_max, 0.0, 1.0)
        g = linear_conv(y, y_min, y_max, 0.0, 1.0)
        b = linear_conv(z, z_min, z_max, 0.0, 1.0) if WITH_3D else 1.0
        labels_colors.append((r, g, b))
    return labels_colors


# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def log_binom(n, k):
    """
    Numerically stable binomial coefficient
    """
    n1 = loggamma(n + 1)
    d1 = loggamma(k + 1)
    d2 = loggamma(n - k + 1)
    return n1 - d1 - d2


# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def fex(target_set, query_set, full_set, alpha=0.05):
    """
    Fisher Exact test for 3 sets of genes (target, query and full)
    """
    ts = set(target_set)
    qs = set(query_set)
    fs = set(full_set)
    
    qs_and_ts = qs.intersection(ts)
    qs_not_ts = qs.difference(ts)
    ts_not_qs = fs.difference(qs).intersection(ts)
    not_ts_not_qs = fs.difference(qs).difference(ts)
    
    x = np.zeros((2, 2))
    x[0, 0] = len(qs_and_ts)
    x[0, 1] = len(qs_not_ts)
    x[1, 0] = len(ts_not_qs)
    x[1, 1] = len(not_ts_not_qs)
    
    p1 = log_binom(x[0, :].sum(), x[0, 0])
    p2 = log_binom(x[1, :].sum(), x[1, 0])
    p3 = log_binom(x.sum(), x[:, 0].sum())

    return np.exp(p1 + p2 - p3)


# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def select_set(counts, names, mass_proportion):
    """
    Select the top G genes which constitutes
    the fraction (mass_proportion) of the counts
    using a cumulative sum distribution
    """    
    sidx = np.fliplr(np.argsort(counts, axis=1)).astype(int)
    cumsum = np.cumsum(np.take_along_axis(counts, sidx, axis=1), axis=1)
    lim = np.max(cumsum, axis=1) * mass_proportion
    lim = lim.reshape(-1, 1)
    q = np.argmin(cumsum <= lim, axis=1)
    return [names[sidx[x, 0:q[x]]].tolist() for x in range(counts.shape[0])]


# Code snippet taken from Alma Andersson 
# https://github.com/almaan/STDGE/blob/master/enrich.py
def enrichment_score(counts, target_set, mass_proportion=0.90):
    """
    Computes the enrichment score for all
    spots (rows) based on a gene set (target)
    using p-values
    """
    query_all = counts.columns.values
    query_top_list = select_set(counts.values,
                                query_all,
                                mass_proportion)
    full_set = query_all.tolist() + target_set
    pvals = [fex(target_set, q, full_set) for q in query_top_list]
    return -np.log(np.array(pvals))