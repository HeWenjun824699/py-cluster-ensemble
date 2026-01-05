import numpy as np
import math
from sklearn import metrics
from kneed import KneeLocator


def get_threshold(adj_matrix, percent_threshold):
    sorted_graph_data = np.sort(adj_matrix, axis=None)
    sorted_graph_data = np.array(sorted_graph_data[sorted_graph_data > 0])
    boundary_element_index = math.floor(sorted_graph_data.size * (100 - percent_threshold) / 100)
    if boundary_element_index >= len(sorted_graph_data):
        boundary_element_index = len(sorted_graph_data) - 1
    return sorted_graph_data[int(boundary_element_index)]


def get_elbow(mode, adj_matrix, max_modules, min_modules):
    s, _ = np.linalg.eig(adj_matrix)
    s = np.abs(s)
    N = max_modules + 1
    ind = np.arange(min_modules, N, 1)
    y_values = s[min_modules:N]

    if len(y_values) != len(ind):
        return int((max_modules + min_modules) / 2)

    kn = KneeLocator(ind, y_values, S=1.0, curve='convex', direction='decreasing', online=True)
    if kn.knee is None:
        return int((max_modules + min_modules) / 2)
    return kn.knee


def adjust_modular_partition(individual_all_modules, group_consensus_labels, individual_orig_labels,
                             improvement_threshold, base_consensus_cost, modularisation_method, max_modules,
                             min_modules):
    modify = False
    best_consensus_cost = base_consensus_cost
    best_new_labels = individual_orig_labels

    if modularisation_method == 'Spectral':
        for num_module in range(min_modules, max_modules, 1):
            if num_module not in individual_all_modules:
                continue
            labels = individual_all_modules[num_module]
            new_consensus_cost = metrics.adjusted_mutual_info_score(group_consensus_labels, labels)

            if new_consensus_cost > best_consensus_cost and (
                    new_consensus_cost - base_consensus_cost) > improvement_threshold:
                best_consensus_cost = new_consensus_cost
                modify = True
                best_new_labels = labels
    return best_new_labels, best_consensus_cost, modify
