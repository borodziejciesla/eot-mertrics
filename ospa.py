import numpy as np
from hungarian_algorithm import algorithm
import elliptical_shape_metrics as esm

def ospa(reference, estimation, c = 10, p = 1, metric = esm.gw_distance):
    # Get numbers of reference and estimated objects
    reference_objects_number = reference.shape[0]
    estimation_objects_number = estimation.shape[0]
    # Assignment
    distances = calculate_distances(reference, estimation, p, metric)
    assignment_value = make_assignment(distances)

    # Metric
    cardinality_part = np.power(c, p) * (estimation_objects_number - reference_objects_number)
    distance_part = assignment_value
    ospa_metric = np.power((distance_part + cardinality_part) / estimation_objects_number, 1.0 / p)
    
    return ospa_metric

def calculate_distances(reference, estimation, p, metric):
    # Get numbers of reference and estimated objects
    reference_objects_number = reference.shape[0]
    estimation_objects_number = estimation.shape[0]
    # Calculate distances
    distances = np.zeros((reference_objects_number, estimation_objects_number))
    for reference_index in range(reference_objects_number):
        for estimation_index in range(estimation_objects_number):
            reference_ellipse = reference[reference_index, :]
            estimation_ellipse = estimation[estimation_index, :]
            distances[reference_index, estimation_index] = np.power(metric(reference_ellipse, estimation_ellipse), p)

    return distances

def make_assignment(distances):
    # Get Dimensions
    rows_number = distances.shape[0]
    cols_number = distances.shape[1]
    # Make input graph
    input = {}
    for row_index in range(rows_number):
        row = {}
        for column_index in range(cols_number):
            row[str(column_index)] = distances[row_index, column_index]
        input[str(row_index)] = row

    assignment_value = algorithm.find_matching(input, matching_type = 'min', return_type = 'total')
    return assignment_value