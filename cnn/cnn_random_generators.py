import numpy as np

from .cnn_architecture import CnnArchitecture
from .cnn_schema import get_schema


def random_conv_layer_type(block_idx, layer_idx, layer_type, relu_prob=0.9, pre_pool_prob=0.25):
    i, j = block_idx, layer_idx
    schema = get_schema()
    layer = {"type": layer_type}
    if layer_type == "Conv2D":
        layer["ker_size"] = schema[f"conv{i}-l{j}-ker-size"].uniform_random_value()
        layer["filters"] = schema[f"conv{i}-l{j}-filters"].uniform_random_value()
        layer["2x_stride"] = schema[f"conv{i}-l{j}-2x-stride"].uniform_random_value()
    elif layer_type == "1x1Conv2D":
        layer["filters"] = schema[f"conv{i}-l{j}-filters"].uniform_random_value()
    elif layer_type == "DWConv2D":
        layer["ker_size"] = schema[f"conv{i}-l{j}-ker-size"].uniform_random_value()
        layer["2x_stride"] = schema[f"conv{i}-l{j}-2x-stride"].uniform_random_value()
    else:
        raise ValueError(f"Unknown conv layer type: {layer_type}")
    layer["has_bn"] = schema[f"conv{i}-l{j}-has-bn"].uniform_random_value()
    layer["has_relu"] = (np.random.random_sample() < relu_prob)
    layer["has_prepool"] = (np.random.random_sample() < pre_pool_prob)
    return layer


def random_conv_layer(block_idx, layer_idx):
    schema = get_schema()
    layer_type = schema[f"conv{block_idx}-l{layer_idx}-type"].uniform_random_value()
    return random_conv_layer_type(block_idx, layer_idx, layer_type)


def random_conv_block(block_idx, num_layers=None):
    schema = get_schema()

    i = block_idx
    block = {
        "is_branch": False if block_idx == 0 else schema[f"conv{i}-is-branch"].uniform_random_value(),
        "layers": []
    }
    num_layers = num_layers or schema[f"conv{i}-num-layers"].uniform_random_value()
    for j in range(num_layers):
        layer = random_conv_layer(i, j)
        block["layers"].append(layer)

    return block


def random_pooling():
    schema = get_schema()
    return {
        "type": "avg" if schema["pool-is-avg"].uniform_random_value() else "max",
        "pool_size": schema["pool-size"].uniform_random_value()
    }


def random_dense_block(block_idx):
    schema = get_schema()
    i = block_idx
    return {
        "units": schema[f"dense{i}-units"].uniform_random_value(),
        "activation": "relu"
    }


def random_arch(pooling_prob=0.9):
    """
    Generates a valid architecture by sampling free variables uniformly at random.
    :param pooling_prob: Probability of pooling
    :return: The architecture
    """
    schema = get_schema()
    arch = {
        "conv_blocks": [],
        "pooling": None,
        "dense_blocks": []
    }

    num_conv_blocks = schema["num-conv-blocks"].uniform_random_value()
    for i in range(num_conv_blocks):
        block = random_conv_block(i)
        arch["conv_blocks"].append(block)

    if np.random.random_sample() < pooling_prob:
        arch["pooling"] = random_pooling()

    num_dense_blocks = schema["num-dense-blocks"].uniform_random_value()
    for i in range(num_dense_blocks):
        block = random_dense_block(i)
        arch["dense_blocks"].append(block)

    return CnnArchitecture(arch)
