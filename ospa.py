import numpy as np
from hungarian import hungarian
import elliptical_shape_metrics as esm


def ospa(x, y, c = 10, p = 1, metric = esm.gw_distance):
    if (x.size == 0) and (x.size == 0):
        dist = 0
        return dist
    
    if (x.size == 0) or (x.size == 0):
        dist = c
        return dist
    
    # Calculate size of the inpout point patterns
    n = x.shape[0]
    m = y.shape[0]

    # Calculate cost matrix for pairings
    distances = calculate_distances(x, y, p, metric)

    # Compute optimal assignment and cost using the Hungarian algorithm
    _, cost = hungarian(distances)

    # Calculate final distance
    dist = np.power( 1.0 / max(m, n) * (np.power(c, p) * abs(m - n) + cost), 1.0/p )

    return dist

def calculate_distances(x, y, p, metric):
    # Get numbers of reference and estimated objects
    n = x.shape[0]
    m = y.shape[0]
    # Calculate distances
    distances = np.zeros((n, m))
    for reference_index in range(n):
        for estimation_index in range(m):
            reference_ellipse = x[reference_index, :]
            estimation_ellipse = y[estimation_index, :]
            distances[reference_index, estimation_index] = np.power(metric(reference_ellipse, estimation_ellipse), p)

    return distances
