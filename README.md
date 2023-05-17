# eot-mertrics
Metrics for Extended Objets Tracking

## Object
Metrics assumes Ellipse objects:
$
    X = \left[ x \, y \, \alpha \, l_1 \, l_2 \right]^{T}
$

## Single Object Metrics
* `lp_metric(reference_ellipse, estimated_ellipse, p=2)` - $L_p$ distance
* `kl_distance(reference_ellipse, estimated_ellipse)` - Kullback-Leibler Divergence
* `hellinger_distance(reference_ellipse, estimated_ellipse)` - Hellinger Distance
* `gw_distance(reference_ellipse, estimated_ellipse)` - Gaussian Wasserstein Distance
