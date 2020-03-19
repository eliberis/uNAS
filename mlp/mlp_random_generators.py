from .mlp_architecture import MlpArchitecture
from .mlp_schema import get_schema


def random_layer(layer_idx):
    schema = get_schema()
    return {"units": schema[f"dense{layer_idx}-units"].uniform_random_value()}


def random_arch():
    schema = get_schema()
    num_layers = schema["num-dense-layers"].uniform_random_value()
    return MlpArchitecture([random_layer(i) for i in range(num_layers)])
