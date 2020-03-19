import numpy as np
from typing import List, Union


class TensorDesc:
    """
    Describes a tensor: name, shape, element size and whether it's a constant (e.g. weight) or
    produced by an operator (e.g. an activation matrix).
    """
    def __init__(self, name=None, shape=None, elem_size=None, is_constant=False, producer=None,
                 sparse_size=None):
        self.name = name
        self.shape = shape
        self.elem_size = elem_size
        self.is_constant = is_constant
        self.producer = producer
        self.consumers = []
        self.predecessors = []
        self.sparse_size = sparse_size

    @property
    def size(self):
        return np.prod(self.shape) * self.elem_size

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"TensorDesc[name={self.name}, shape={self.shape}]"


class OperatorDesc:
    """
    Describes an operator. Serves as a base class for all ops.
    """
    def __init__(self, name):
        self.graph = _get_current_graph()
        self.name = self.graph.register_operator(self, proposed_name=name)
        self.inputs = []
        self.output = None

    def _add_weight(self, shape, suffix="weight", sparse_size=None):
        t = self.graph.add_tensor(f"{self.name}_{suffix}", shape, is_constant=True,
                                  sparse_size=sparse_size)
        t.consumers.append(self)
        self.inputs.append(t)

    def _produce_output(self, shape):
        assert all(d > 0 for d in shape), f"{self.name} has an invalid shape: {shape}"
        t = self.graph.add_tensor(self.name + ":0", shape, producer=self)
        t.predecessors.extend(self.inputs)
        for i in self.inputs:
            t.predecessors.extend(i.predecessors)
        self.output = t
        return t

    def __call__(self, inputs: Union[List[TensorDesc], TensorDesc]):
        if type(inputs) is list:
            self.inputs.extend(inputs)
        else:
            self.inputs.append(inputs)

    def __repr__(self):
        return f"{self.__class__.__name__}[name={self.name}]"

_current_graph = None

class _GraphContext:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        global _current_graph
        _current_graph = self.graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_graph
        _current_graph = None
        return exc_type is None


def _disambiguate_name(name, existing_names):
    if name not in existing_names:
        return name
    else:
        ctr = 1
        while f"{name}_{ctr}" in existing_names:
            ctr += 1
        return f"{name}_{ctr}"


class Graph:
    """
    Represents a computation graph: a collection of tensors and operators, where operators
    take one or more tensors as an input and produce _one_ tensor as their output. Abstractions
    here allow us to track the relationships between tensors and operators, as well as their sizes.
    :param element_type A numpy.dtype of all tensors in the graph.
    """
    def __init__(self, element_type=np.float32):
        self.tensors = {}
        self.operators = {}
        self.element_size = element_type().itemsize
        self.inputs = []
        self.outputs = []

    def as_current(self):
        return _GraphContext(self)

    def register_operator(self, o: OperatorDesc, proposed_name: str):
        name = _disambiguate_name(proposed_name, self.operators)
        self.operators[name] = o
        return name

    def add_tensor(self, proposed_name, shape, is_constant=False, producer=None, sparse_size=None):
        name = _disambiguate_name(proposed_name, self.tensors)
        t = TensorDesc(name, shape, self.element_size, is_constant, producer, sparse_size)
        self.tensors[t.name] = t
        return t

    def add_output(self, tensors: Union[TensorDesc, List[TensorDesc]]):
        if type(tensors) is list:
            self.outputs.extend(tensors)
        else:
            self.outputs.append(tensors)

    def add_input(self, tensors: Union[TensorDesc, List[TensorDesc]]):
        if type(tensors) is list:
            self.inputs.extend(tensors)
        else:
            self.inputs.append(tensors)


def _get_current_graph() -> Graph:
    global _current_graph
    if _current_graph is None:
        raise ValueError("Wrap your model assembling code in a `with g.as_current()` block")
    return _current_graph

