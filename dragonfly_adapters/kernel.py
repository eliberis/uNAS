import numpy as np
from dragonfly.nn.otmann import OTMANNDistanceComputer

from .neural_network import label_info, layer_group_types, all_possible_labels


def build_label_mismatch_cost_matrix(all_labels, non_assignment_penalty):
    num_labels = len(all_labels)
    label_penalties = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(i, num_labels):
            labi = label_info(all_labels[i])
            labj = label_info(all_labels[j])

            if all_labels[i] == all_labels[j]:
                cost = 0.0
            elif labi["group"] == labj["group"] == "conv":
                cost = np.sqrt(2 * abs(labi["kernel_size"] - labj["kernel_size"])) / 10
                cost += 0.1 if labi["batch_norm"] != labj["batch_norm"] else 0.0
                cost += 0.1 if labi["activation"] != labj["activation"] else 0.0
                cost += 0.1 if labi["is_dw"] != labj["is_dw"] else 0.0
            elif labi["group"] == labj["group"] == "pool":
                cost = 0.5 if labi["type"] != labj["type"] else 0.0
            elif labi["group"] == labj["group"] == "dense":
                cost = 0.1 if labi["batch_norm"] != labj["batch_norm"] else 0.0
                cost += 0.1 if labi["activation"] != labj["activation"] else 0.0
            else:
                cost = np.inf
            label_penalties[i, j] = cost * non_assignment_penalty
            label_penalties[j, i] = cost * non_assignment_penalty
    return label_penalties


def get_distance_computer():
    all_labels = all_possible_labels()
    non_assignment_penalty = 1.0

    struct_penalty_groups = ["all"] + layer_group_types()

    mislabel_coefs = {
        "all": 1.0,
        "conv": 1.0,
        "pool": 1.0,
        "aggr": 1.0,
        "dense": 1.0,
    }

    struct_coefs = {
        "all": 0.1,
        "conv": 0.25,
        "pool": 0.61,
        "aggr": 0.2,
        "dense": 1.5,
    }

    return OTMANNDistanceComputer(
        all_layer_labels=all_possible_labels(),
        label_mismatch_penalty=build_label_mismatch_cost_matrix(all_labels, non_assignment_penalty),
        non_assignment_penalty=non_assignment_penalty,
        structural_penalty_groups=struct_penalty_groups,
        path_length_types=["shortest", "longest", "rw"],
        dflt_mislabel_coeffs=[mislabel_coefs[g] for g in struct_penalty_groups],
        dflt_struct_coeffs=[struct_coefs[g] for g in struct_penalty_groups],
        dflt_dist_type="lp-emd",
        connectivity_diff_cost_function="linear")
