import numpy as np
import math
import matplotlib.pyplot as plt
import elliptical_shape_metrics as esm

# Set Reference
reference = np.array([0.0, 0.0, 0.0, 2.0, 1.0])

###############################################################################
# Calculate metrics for rotated ellipse
lp_distance = []
kl_distance = []
h_distance = []
gw_distance = []

rotation = [(x / 100.0) * math.pi for x in range(-100, 100)]

for alpha in rotation:
    # Rotated ellipse
    estimation = np.array([0.0, 0.0, alpha, 2.0, 1.0])
    # Calculate distances
    lp_distance.append(esm.lp_metric(reference, estimation))
    kl_distance.append(esm.kl_distance(reference, estimation))
    h_distance.append(esm.hellinger_distance(reference, estimation))
    gw_distance.append(esm.gw_distance(reference, estimation))

# Plots
fig, ax = plt.subplots()
ax.grid(True)
ax.set_title("Metrics for rotated ellipse")
ax.set_xlabel("Rotation angle [rad]")
ax.set_ylabel("Metric [-]")

ax.plot(np.array(rotation), np.array(lp_distance),
    linewidth=2.0, label='Lp Distan')
ax.plot(np.array(rotation), np.array(kl_distance),
    linewidth=2.0, label=' Kullback-Leibler Divergence')
ax.plot(np.array(rotation), np.array(h_distance),
    linewidth=2.0, label='Hellinger Distance')
ax.plot(np.array(rotation), np.array(gw_distance),
    linewidth=2.0, label='Gaussian Wasserstein Distance')

ax.set(xlim=(-math.pi, math.pi), xticks=np.arange(-3, 3),
       ylim=(0, 3.5), yticks=np.arange(0, 4))
plt.legend()

plt.show()

###############################################################################
# Calculate metrics for rotated ellipse
lp_distance = []
kl_distance = []
h_distance = []
gw_distance = []

l1 = [(x / 100.0) * 2.0 for x in range(1, 100)]

for l in l1:
    # Rotated ellipse
    estimation = np.array([0.0, 0.0, 0.0, l, 1.0])
    # Calculate distances
    lp_distance.append(esm.lp_metric(reference, estimation))
    kl_distance.append(esm.kl_distance(reference, estimation))
    h_distance.append(esm.hellinger_distance(reference, estimation))
    gw_distance.append(esm.gw_distance(reference, estimation))

# Plots
fig, ax = plt.subplots()
ax.grid(True)
ax.set_title("Metrics for different l1")
ax.set_xlabel("l1 [m]")
ax.set_ylabel("Metric [-]")

ax.plot(np.array(l1), np.array(lp_distance),
    linewidth=2.0, label='Lp Distan')
ax.plot(np.array(l1), np.array(kl_distance),
    linewidth=2.0, label=' Kullback-Leibler Divergence')
ax.plot(np.array(l1), np.array(h_distance),
    linewidth=2.0, label='Hellinger Distance')
ax.plot(np.array(l1), np.array(gw_distance),
    linewidth=2.0, label='Gaussian Wasserstein Distance')

ax.set(xlim=(0, 2), xticks=np.arange(0, 2),
       ylim=(0, 6), yticks=np.arange(0, 6))
plt.legend()

plt.show()

###############################################################################
# Calculate metrics for translated metric
lp_distance = []
kl_distance = []
h_distance = []
gw_distance = []

ranges = [(x / 100.0) * 5.0 for x in range(0, 100)]

for r in ranges:
    # Rotated ellipse
    estimation = np.array([r, r, 0.0, 2.0, 1.0])
    # Calculate distances
    lp_distance.append(esm.lp_metric(reference, estimation))
    kl_distance.append(esm.kl_distance(reference, estimation))
    h_distance.append(esm.hellinger_distance(reference, estimation))
    gw_distance.append(esm.gw_distance(reference, estimation))

# Plots
fig, ax = plt.subplots()
ax.grid(True)
ax.set_title("Metrics for translated center [d, d]")
ax.set_xlabel("Distance [m]")
ax.set_ylabel("Metric [-]")

ax.plot(np.array(ranges), np.array(lp_distance),
    linewidth=2.0, label='Lp Distan')
ax.plot(np.array(ranges), np.array(kl_distance),
    linewidth=2.0, label=' Kullback-Leibler Divergence')
ax.plot(np.array(ranges), np.array(h_distance),
    linewidth=2.0, label='Hellinger Distance')
ax.plot(np.array(ranges), np.array(gw_distance),
    linewidth=2.0, label='Gaussian Wasserstein Distance')

ax.set(xlim=(0, 5), xticks=np.arange(0, 5),
       ylim=(0, 10), yticks=np.arange(0, 10))
plt.legend()

plt.show()