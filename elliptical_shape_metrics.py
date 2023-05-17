import numpy as np
from scipy.linalg import sqrtm

def lp_metric(reference_ellipse, estimated_ellipse, p=2):
    # ellipse -> [x, y, orientation, l1, l2]'
    raw_diff = reference_ellipse - estimated_ellipse
    diff_pow = np.power(raw_diff, p)
    diff_pow_sum = sum(diff_pow)
    distance = np.sqrt(diff_pow_sum, 1.0 / p)
    return distance

def decoupled_measures(reference_ellipse, estimated_ellipse):
    # ellipse -> [x, y, orientation, l1, l2]'
    reference_matrix = calculate_matrix_representation(reference_ellipse)
    estimated_matrix = calculate_matrix_representation(estimated_ellipse)

    diff = reference_matrix - estimated_matrix
    distance = np.sqrt(np.trace(diff * diff.T))
    return distance

def kl_distance(reference_ellipse, estimated_ellipse):
    # ellipse -> [x, y, orientation, l1, l2]'
    kl_1 = kl_distance_single(reference_ellipse, estimated_ellipse)
    kl_2 = kl_distance_single(estimated_ellipse, reference_ellipse)
    return kl_1 + kl_2

def kl_distance_single(ellipse_1, ellipse_2):
    # ellipse -> [x, y, orientation, l1, l2]'
    center_1 = create_center(ellipse_1)
    matrix_1 = calculate_matrix_representation(ellipse_1)

    center_2 = create_center(ellipse_2)
    matrix_2 = calculate_matrix_representation(ellipse_2)

    diff = center_2 - center_1
    matrix_2_inv = np.linalg.inv(matrix_2)
    matrix_1_det = np.linalg.det(matrix_1)
    matrix_2_det = np.linalg.det(matrix_2)

    distance = 0.5 * (np.trace(matrix_2_inv*matrix_1) + diff.T*matrix_2_inv*diff
        - 2.0 + np.log(matrix_2_det / matrix_1_det))
    return distance

def hellinger_distance(reference_ellipse, estimated_ellipse):
    # ellipse -> [x, y, orientation, l1, l2]'
    reference_center = create_center(reference_ellipse)
    reference_matrix = calculate_matrix_representation(reference_ellipse)

    estimated_center = create_center(estimated_ellipse)
    estimated_matrix = calculate_matrix_representation(estimated_ellipse)

    diff = reference_center - estimated_center
    matrix_reference_det = np.linalg.det(reference_matrix)
    matrix_estimated_det = np.linalg.det(estimated_matrix)
    matrix_sum = reference_matrix + estimated_matrix
    matrix_sum_det = np.linalg.det(matrix_sum / 2.0)
    matrix_sum_inv = np.linalg.inv(matrix_sum / 2.0)

    distance_sqr = 1.0 - (np.power(matrix_reference_det, 0.25) * np.power(matrix_estimated_det, 0.25) / np.power(matrix_sum_det, 0.5)) * np.exp(-diff.T * matrix_sum_inv * diff / 8.0)
    return np.sqrt(distance_sqr)

def gw_distance(reference_ellipse, estimated_ellipse):
    # ellipse -> [x, y, orientation, l1, l2]'
    reference_center = create_center(reference_ellipse)
    reference_matrix = calculate_matrix_representation(reference_ellipse)

    estimated_center = create_center(estimated_ellipse)
    estimated_matrix = calculate_matrix_representation(estimated_ellipse)

    diff = reference_center - estimated_center
    diff_norm = np.linalg.norm(diff)
    matrix_reference_sqrtm = sqrtm(reference_matrix)
    matrix_sum = reference_matrix + estimated_matrix

    distance_sqr = diff_norm + np.trace(matrix_sum - 2.0 * sqrtm(matrix_reference_sqrtm * estimated_matrix * matrix_reference_sqrtm))
    return np.sqrt(distance_sqr)

def calculate_matrix_representation(ellipse):
    # ellipse -> [x, y, orientation, l1, l2]'
    alpha = ellipse[2]
    l1 = ellipse[3]
    l2 = ellipse[4]

    c = np.cos(alpha)
    s = np.sin(alpha)

    R = np.array([[c, -s], [s, c]])
    l = np.array([[l1**2, 0.0], [0.0, l2**2]])

    return R * l * R.T

def create_center(ellipse):
    return ellipse[0:1]