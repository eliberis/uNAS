from typing import Dict
from schema_types import Discrete, ValueType

MAX_DENSE_LAYERS = 7

_SCHEMA = None


def build_schema() -> Dict[str, ValueType]:
    """
    Defines the search space. In this search space, we create MLP models that consist of a flattening layer,
    followed by several fully connected layers at various widths.
    """
    keys = []
    keys.append(Discrete("num-dense-layers", bounds=(1, MAX_DENSE_LAYERS)))
    for d in range(MAX_DENSE_LAYERS):
        keys.append(Discrete(f"dense{d}-units", bounds=(10, 256), can_be_optional=(d > 0)))

    return {k.name: k for k in keys}


def get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = build_schema()
    return _SCHEMA


