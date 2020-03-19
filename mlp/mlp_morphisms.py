import numpy as np
from copy import deepcopy
from typing import List

from .mlp_architecture import MlpArchitecture
from .mlp_schema import get_schema
from .mlp_random_generators import random_layer


def kill_random_layer(layers):
    layers = deepcopy(layers)
    idx = np.random.randint(len(layers))
    layers.pop(idx)
    return layers


def add_layer(layers):
    layers = deepcopy(layers)
    layers.append(random_layer(layer_idx=len(layers)))
    return layers


def adjust_number_of_units(layers, layer_idx, amount):
    layers = deepcopy(layers)
    layers[layer_idx]["units"] += amount
    return layers


def produce_all_morphs(arch: MlpArchitecture) -> List[MlpArchitecture]:
    schema = get_schema()
    parent = arch.architecture
    num_layers = len(parent)

    morphs = []

    # 1. Decrease or increase the number of layers by 1
    min_layers, max_layers = schema["num-dense-layers"].bounds
    if num_layers > min_layers:
        morphs.append(kill_random_layer(layers=parent))
    if num_layers < max_layers:
        morphs.append(add_layer(layers=parent))

    # 2. Adjust the number of units in each layer by [-5, -1, +1, +5]
    for i, l in enumerate(parent):
        min_units, max_units = schema[f"dense{i}-units"].bounds
        for adjustment in [-5, -1, 1, 5]:
            if min_units <= l["units"] + adjustment <= max_units:
                morphs.append(adjust_number_of_units(parent, i, adjustment))

    return [MlpArchitecture(m) for m in morphs]
