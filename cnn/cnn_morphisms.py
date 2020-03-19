import numpy as np

from typing import List
from copy import deepcopy

from .cnn_schema import get_schema
from .cnn_architecture import CnnArchitecture
from .cnn_random_generators import random_conv_block, random_conv_layer, random_pooling, \
    random_dense_block, random_conv_layer_type


def remove_random_conv_block(arch):
    return remove_conv_block(arch,
                             idx=np.random.randint(len(arch["conv_blocks"])))


def remove_last_conv_block(arch):
    return remove_conv_block(arch, idx=len(arch["conv_blocks"]) - 1)


def remove_conv_block(arch, idx):
    arch = deepcopy(arch)
    num_blocks = len(arch["conv_blocks"])

    is_branch = arch["conv_blocks"][idx]["is_branch"]
    block_after_is_branch = (idx < num_blocks - 1) and arch["conv_blocks"][idx + 1]["is_branch"]

    if not is_branch and block_after_is_branch:
        arch["conv_blocks"][idx + 1]["is_branch"] = False

    arch["conv_blocks"].pop(idx)
    return arch


def add_random_block(arch):
    arch = deepcopy(arch)
    block_idx = len(arch["conv_blocks"])
    arch["conv_blocks"].append(random_conv_block(block_idx, num_layers=1))
    return arch


def change_conv_block_type(arch, block_idx):
    if block_idx == 0:
        return []

    arch = deepcopy(arch)
    arch["conv_blocks"][block_idx]["is_branch"] = not arch["conv_blocks"][block_idx]["is_branch"]
    return [arch]


def remove_random_conv_layer(arch, block_idx):
    arch = deepcopy(arch)
    block = arch["conv_blocks"][block_idx]
    idx = np.random.randint(len(block["layers"]))
    block["layers"].pop(idx)
    return arch


def remove_last_conv_layer(arch, block_idx):
    arch = deepcopy(arch)
    block = arch["conv_blocks"][block_idx]
    block["layers"].pop(len(block["layers"]) - 1)
    return arch


def add_conv_layer(arch, block_idx):
    arch = deepcopy(arch)
    block = arch["conv_blocks"][block_idx]
    layer_idx = len(block["layers"])
    block["layers"].append(random_conv_layer(block_idx, layer_idx))
    return arch


def add_random_conv_layer(arch, block_idx):
    arch = deepcopy(arch)
    block = arch["conv_blocks"][block_idx]
    layer_idx = np.random.randint(len(block["layers"]))
    block["layers"].insert(layer_idx, random_conv_layer(block_idx, layer_idx))
    return arch


def change_pooling(arch):
    arch = deepcopy(arch)
    if arch["pooling"]:
        arch["pooling"] = None
    else:
        arch["pooling"] = random_pooling()
    return arch


def change_pooling_type(arch):
    arch = deepcopy(arch)
    pool_type = arch["pooling"]["type"]
    arch["pooling"]["type"] = "avg" if pool_type == "max" else "max"
    return arch


def change_pooling_size(arch, change):
    arch = deepcopy(arch)
    arch["pooling"]["pool_size"] += change
    return arch


def remove_random_dense_block(arch):
    arch = deepcopy(arch)
    idx = np.random.randint(len(arch["dense_blocks"]))
    arch["dense_blocks"].pop(idx)
    return arch


def add_dense_block(arch):
    arch = deepcopy(arch)
    idx = len(arch["dense_blocks"])
    arch["dense_blocks"].append(random_dense_block(idx))
    return arch


def change_dense_units(arch, block_idx, change):
    arch = deepcopy(arch)
    arch["dense_blocks"][block_idx]["units"] += change
    return arch


def change_conv_layer_boolean_property(arch, block_idx, layer_idx, property):
    arch = deepcopy(arch)
    layer = arch["conv_blocks"][block_idx]["layers"][layer_idx]
    layer[property] = not layer[property]
    return arch


def change_conv_layer_type(arch, block_idx, layer_idx):
    schema = get_schema()
    i, j = block_idx, layer_idx
    layer_types = deepcopy(schema[f"conv{i}-l{j}-type"].values)
    layer_types.remove(arch["conv_blocks"][i]["layers"][j]["type"])

    morphs = []
    for lt in layer_types:
        arch = deepcopy(arch)
        arch["conv_blocks"][i]["layers"][j] = random_conv_layer_type(i, j, lt)
        morphs.append(arch)
    return morphs


def change_kernel_size(arch, block_idx, layer_idx, kernel_adjustment):
    arch = deepcopy(arch)
    arch["conv_blocks"][block_idx]["layers"][layer_idx]["ker_size"] += kernel_adjustment
    return arch


def change_filters(arch, block_idx, layer_idx, filters_adjustment):
    arch = deepcopy(arch)
    arch["conv_blocks"][block_idx]["layers"][layer_idx]["filters"] += filters_adjustment
    return arch


def produce_all_morphs(arch: CnnArchitecture) -> List[CnnArchitecture]:
    schema = get_schema()
    parent = arch.architecture

    morphs = []

    num_conv_blocks = len(parent["conv_blocks"])
    min_conv_blocks, max_conv_blocks = schema["num-conv-blocks"].bounds
    if num_conv_blocks > min_conv_blocks:
        morphs.append(remove_random_conv_block(parent))
    if num_conv_blocks < max_conv_blocks:
        morphs.append(add_random_block(parent))

    for i, conv_block in enumerate(parent["conv_blocks"]):
        morphs.extend(change_conv_block_type(parent, i))

        num_conv_layers = len(conv_block["layers"])
        min_conv_layers, max_conv_layers = schema[f"conv{i}-num-layers"].bounds
        if num_conv_layers == min_conv_layers and num_conv_blocks > min_conv_blocks:
            morphs.append(remove_conv_block(parent, i))
        if num_conv_layers > min_conv_layers:
            morphs.append(remove_random_conv_layer(parent, i))
        if num_conv_layers < max_conv_layers:
            morphs.append(add_random_conv_layer(parent, i))

        for j, conv_layer in enumerate(conv_block["layers"]):
            morphs.extend([
                change_conv_layer_boolean_property(parent, i, j, "has_bn"),
                change_conv_layer_boolean_property(parent, i, j, "has_relu"),
                change_conv_layer_boolean_property(parent, i, j, "has_prepool")
            ])
            morphs.extend(change_conv_layer_type(parent, i, j))

            if conv_layer["type"] in ["Conv2D", "1x1Conv2D"]:
                min_filters, max_filters = schema[f"conv{i}-l{j}-filters"].bounds
                for d in [-5, -3, -1, +1, +3, +5]:
                    if min_filters <= conv_layer["filters"] + d <= max_filters:
                        morphs.append(change_filters(parent, i, j, d))
            if conv_layer["type"] in ["Conv2D", "DWConv2D"]:
                min_ker_size, max_ker_size = schema[f"conv{i}-l{j}-ker-size"].bounds
                increment = schema[f"conv{i}-l{j}-ker-size"].increment
                for d in [-increment, +increment]:
                    if min_ker_size <= conv_layer["ker_size"] + d <= max_ker_size:
                        morphs.append(change_kernel_size(parent, i, j, d))
                morphs.append(change_conv_layer_boolean_property(parent, i, j, "2x_stride"))

    morphs.append(change_pooling(parent))
    if parent["pooling"]:
        morphs.append(change_pooling_type(parent))
        pool_size = parent["pooling"]["pool_size"]
        min_pool_size, max_pool_size = schema["pool-size"].bounds
        increment = schema["pool-size"].increment
        for d in [-increment, +increment]:
            if min_pool_size <= pool_size + d <= max_pool_size:
                morphs.append(change_pooling_size(parent, d))

    num_dense_blocks = len(parent["dense_blocks"])
    min_dense_blocks, max_dense_blocks = schema["num-dense-blocks"].bounds
    if num_dense_blocks > min_dense_blocks:
        morphs.append(remove_random_dense_block(parent))
    if num_dense_blocks < max_dense_blocks:
        morphs.append(add_dense_block(parent))

    # (1 layer in a dense block)
    for i, dense_layer in enumerate(parent["dense_blocks"]):
        min_units, max_units = schema[f"dense{i}-units"].bounds
        for d in [-5, -3, -1, +1, +3, +5]:
            if min_units <= dense_layer["units"] + d <= max_units:
                morphs.append(change_dense_units(parent, i, d))

    return [CnnArchitecture(m) for m in morphs]
