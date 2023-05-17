# eot-mertrics
Metrics for Extended Objets Tracking

## Object
Metrics assumes Ellipse objects:

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.image?&space;X&space;=&space;\left[&space;x&space;\,&space;y&space;\,&space;\alpha&space;\,&space;l_1&space;\,&space;l_2&space;\right]^{T}) 


## Single Object Metrics
* `lp_metric(reference_ellipse, estimated_ellipse, p=2)` - $L_p$ distance
* `kl_distance(reference_ellipse, estimated_ellipse)` - Kullback-Leibler Divergence
* `hellinger_distance(reference_ellipse, estimated_ellipse)` - Hellinger Distance
* `gw_distance(reference_ellipse, estimated_ellipse)` - Gaussian Wasserstein Distance
