from typing import Dict
from schema_types import Discrete, Boolean, Categorical, ValueType

MAX_CONV_BLOCKS = 10
MAX_LAYERS_PER_CONV_BLOCK = 3
MAX_DENSE_BLOCKS = 3

_SCHEMA = None


def build_schema() -> Dict[str, ValueType]:
    """
    Defines the search space. In this search space, we create CNN models that consist of:
    * up to 10 blocks of CNN layers (up to 3 layers per each block, arranged in series or in parallel w/ prev block)
    * each convolutional layer is either a 2D convolution, a 1x1 2D convolution or a 2D depthwise sep. convolution
    :returns Free search space variables, keyed by name.
    """
    keys = []
    keys.append(Discrete("num-conv-blocks", bounds=(1, MAX_CONV_BLOCKS)))
    for c in range(MAX_CONV_BLOCKS):
        keys.append(Boolean(f"conv{c}-is-branch", can_be_optional=(c > 0)))
        keys.append(Discrete(f"conv{c}-num-layers", bounds=(1, MAX_LAYERS_PER_CONV_BLOCK), can_be_optional=True))
        for i in range(MAX_LAYERS_PER_CONV_BLOCK):
            keys.extend([
                Categorical(f"conv{c}-l{i}-type", values=["Conv2D", "1x1Conv2D", "DWConv2D"], can_be_optional=True),
                Discrete(f"conv{c}-l{i}-ker-size", bounds=(3, 7), increment=2, can_be_optional=True),
                Discrete(f"conv{c}-l{i}-filters", bounds=(1, 128), can_be_optional=True),
                Boolean(f"conv{c}-l{i}-2x-stride", can_be_optional=True),
                Boolean(f"conv{c}-l{i}-has-pre-pool", can_be_optional=True),
                Boolean(f"conv{c}-l{i}-has-bn", can_be_optional=True),
                Boolean(f"conv{c}-l{i}-has-relu", can_be_optional=True),
            ])

    keys.extend([
        Boolean("pool-is-avg", can_be_optional=True),
        Discrete("pool-size", bounds=(2, 6), increment=2, can_be_optional=True)
    ])

    keys.append(Discrete("num-dense-blocks", bounds=(1, MAX_DENSE_BLOCKS)))
    for d in range(MAX_DENSE_BLOCKS):
        keys.extend([
            Discrete(f"dense{d}-units", bounds=(10, 256), can_be_optional=(d > 0)),
        ])

    return {k.name: k for k in keys}


def get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = build_schema()
    return _SCHEMA


def compute_search_space_size():
    schema = get_schema()
    options = 1
    for v in schema.values():
        size = 1
        if isinstance(v, Boolean):
            size = 2
        if isinstance(v, Discrete):
            min, max = v.bounds
            size = max - min + 1
        if isinstance(v, Categorical):
            size = len(v.values)
        options *= size
    return options
