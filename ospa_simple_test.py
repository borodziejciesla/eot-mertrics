import numpy as np
from ospa import ospa
import elliptical_shape_metrics as esm

# Set Reference
reference = np.zeros((4, 5))

reference[0, :] = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
reference[1, :] = np.array([5.0, 5.0, 0.0, 1.0, 0.5])
reference[2, :] = np.array([15.0, 5.0, 0.5, 0.25, 0.5])
reference[3, :] = np.array([15.0, 5.0, 0.5, 1.0, 0.5])

# Set Estimations
estimations = np.zeros((4, 5))

estimations[0, :] = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
estimations[1, :] = np.array([5.0, 5.0, 0.0, 1.0, 0.5])
estimations[2, :] = np.array([15.0, 5.0, 0.5, 0.25, 0.5])
estimations[3, :] = np.array([15.0, 5.0, 0.5, 1.0, 0.5])

# Calculate OSPA
ospa_metric = ospa(reference, estimations, metric=esm.lp_metric)
print(ospa_metric)