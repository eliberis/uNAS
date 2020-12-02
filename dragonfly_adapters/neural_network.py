import json
import numpy as np

from dragonfly.nn.neural_network import NeuralNetwork as DFNeuralNetwork
from dragonfly.utils.general_utils import reorder_list_or_array
from scipy.sparse import dok_matrix
from itertools import product
from copy import deepcopy

from architecture import Architecture
from resource_models.graph import OperatorDesc
from resource_models.models import peak_memory_usage, model_size, inference_latency, macs
from resource_models.ops import DWConv2D, Conv2D, Pool, Dense
from search_space import SearchSpace


def layer_group_types():
    # Within Dragonfly, each layer belongs to a group describing its structural role
    return ["conv", "pool", "aggr", "dense"]


def op_to_label(op: OperatorDesc):
    """ Creates a label which loosely describes an operator. """
    base = op.__class__.__name__
    info = None
    if isinstance(op, (DWConv2D, Conv2D)):
        # Including kernel size here helps, even though it's accounted for in layer mass
        info = {"kernel_size": op.kernel_size, "batch_norm": op.batch_norm, "activation": op.activation}
    if isinstance(op, Pool):
        info = {"type": op.type}
    if isinstance(op, Dense):
        info = {"batch_norm": op.batch_norm, "activation": op.activation}

    return base if info is None else f"{base}|{json.dumps(info, sort_keys=True)}"


def layer_group_from_label(label: str):
    """ Maps layer/op label to a layer group. """
    label = label.split("|")[0]
    if "Conv" in label:
        return "conv"
    if "Pool" in label:
        return "pool"
    if "Add" in label:
        return "aggr"
    if "Dense" in label:
        return "dense"
    return None


def label_info(label: str):
    """ Parses the label into a structured informative dictionary. """
    parts = label.split("|")
    info = {"group": layer_group_from_label(parts[0])}
    if info["group"] == "conv":
        info["is_dw"] = label.startswith("DW")
    if len(parts) > 1:
        info.update(json.loads(parts[1]))
    return info


def all_possible_labels(conv_kernel_bounds=None, pool_types=None, activation_types=None):
    conv_kernel_bounds = conv_kernel_bounds or (1, 9, 1)
    pool_types = pool_types or ["avg", "max"]
    activation_types = activation_types or ["relu", "softmax", None]

    k_min, k_max, k_incr = conv_kernel_bounds
    conv_kernel_types = range(k_min, k_max + 1, k_incr)
    batch_norm_types = [True, False]

    conv_layer_types = [json.dumps({"kernel_size": k, "batch_norm": bn, "activation": act}, sort_keys=True)
                        for k, bn, act in product(conv_kernel_types, batch_norm_types, activation_types)]
    pool_layer_types = [json.dumps({"type": t}, sort_keys=True)
                        for t in pool_types]
    dense_layer_types = [json.dumps({"batch_norm": bn, "activation": act}, sort_keys=True)
                         for bn, act in product(batch_norm_types, activation_types)]

    return ["ip", "op"] + \
           ["Add"] + \
           ["Dense|" + info for info in dense_layer_types] + \
           ["Pool|" + info for info in pool_layer_types] + \
           ["Conv2D|" + info for info in conv_layer_types] + \
           ["DWConv2D|" + info for info in conv_layer_types]


class NeuralNetworkWrapper(DFNeuralNetwork):
    def __init__(self, wrapped_arch: Architecture, space: SearchSpace, input_shape, num_classes):
        self.arch = wrapped_arch
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.val_error = None  # Filled out by the experiment caller
        self.test_error = None

        rg = space.to_resource_graph(self.arch, input_shape, num_classes)
        self.resource_features = self.compute_resource_features(rg)

        # Here we convert the architecture into a representation compatible with Dragonfly's kernel,
        # which is:
        # - `layer_labels`: a string describing the kind of a layer ("ip" and "op" for input and
        #    output layers, or e.g. "DWConv2D-3" for a 3x3 DW convolution);
        # - `edge_list`: a list of tuples representing connections between layers (indices are
        #    determined by the position of the latter in the `layer_labels` list);
        # - `units`: number of units in a layer, which Dragonfly uses to compute "layer mass".
        # The conversion can be done for any `Architecture` from its resource graph representation

        layer_labels = []
        edge_list = []
        units = []
        mass = []  # -1 marks non-processing layers, mass will be filled out later

        layers = [t for t in rg.tensors.values() if not t.is_constant]  # input and layer output tensors
        tensor_name_to_id = {t.name: i for i, t in enumerate(layers)}

        # Add input tensors as "input layers"
        for tensor in rg.inputs:
            layer_labels.append("ip")
            units.append(tensor.shape[-1])
            mass.append(-1)

        # Add each operator as its own layer and connect its input layers to itself
        for op in rg.operators.values():
            layer_labels.append(op_to_label(op))
            units.append(op.output.shape[-1])
            inputs = [i for i in op.inputs if i.name in tensor_name_to_id]
            edge_list.extend((tensor_name_to_id[i.name], tensor_name_to_id[op.output.name]) for i in inputs)
            mass.append(macs(op))

        # Invent a dummy layer for each output tensor
        for tensor in rg.outputs:
            i = len(layer_labels)
            layer_labels.append("op")
            units.append(tensor.shape[-1])
            edge_list.append((tensor_name_to_id[tensor.name], i))
            mass.append(-1)

        num_layers = len(layer_labels)
        conn_mat = dok_matrix((num_layers, num_layers))
        for (a, b) in edge_list:
            conn_mat[a, b] = 1

        non_proc_layer_mass = max(0.1 * sum(f for f in mass if f != -1), 100)  # according to Dragonfly
        self.layer_masses = np.array([(non_proc_layer_mass if f == -1 else f) for f in mass], dtype=np.float) / 1000

        super().__init__("unas-net",
                         layer_labels, conn_mat,
                         num_units_in_each_layer=units,
                         all_layer_label_classes=all_possible_labels(),
                         layer_label_similarities=None)

    def to_keras_model(self, space: SearchSpace, *args, **kwargs):
        return space.to_keras_model(self.arch, self.input_shape, self.num_classes, *args, **kwargs)

    def to_resource_graph(self, space: SearchSpace, *args, **kwargs):
        return space.to_resource_graph(self.arch, self.input_shape, self.num_classes, *args, **kwargs)

    @staticmethod
    def compute_resource_features(rg, sparse=False):
        return [peak_memory_usage(rg), model_size(rg, sparse=sparse), macs(rg)]

    def estimate_resource_features(self, sparsity=None):
        rfs = deepcopy(self.resource_features)  # We have them cached from __init__
        if sparsity:
            # Update model size, other resource features are not affected
            # TODO: Add a bitmask?
            rfs[1] = round((1.0 - sparsity) * rfs[1])
        return rfs

    def _child_compute_layer_masses(self):
        # Disable built-in layer computation
        pass

    # Required overrides
    def _child_check_if_valid_network(self):
        # No extra checks on top of those done in the base class
        return True

    def _child_attrs_topological_sort(self, top_order):
        # We compute layer masses before topo sort, so we need to keep this array in order
        reorder_list_or_array(self.layer_masses, top_order)

    @classmethod
    def _get_child_layer_groups(cls):
        return layer_group_types()

    @classmethod
    def _get_layer_group_for_layer_label(cls, layer_label):
        if layer_label in layer_group_types():
            # Semantically, this weird check is needed for Dragonfly
            return None
        return layer_group_from_label(layer_label)
