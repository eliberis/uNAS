import numpy as np
from math import ceil
from typing import List, Optional, Tuple, Union
from .graph import OperatorDesc, TensorDesc, _get_current_graph


def Input(shape, name=None):
    graph = _get_current_graph()
    i = graph.add_tensor("input" if name is None else name, shape)
    graph.add_input(i)
    return i


class Conv2D(OperatorDesc):
    """
    2D Convolution
    """
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, use_bias: bool = True,
                 batch_norm: bool = False, activation: Optional[str] = None, padding: str = "valid",
                 sparse_kernel_size: Optional[int] = None, name: str = None):
        assert padding in ["valid", "same"]
        super().__init__("conv2d" if name is None else name)
        self.num_filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.sparse_kernel_size = sparse_kernel_size

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 4
        batch_size, h, w, in_channels = x.shape
        self._add_weight(shape=(self.kernel_size, self.kernel_size, in_channels, self.num_filters),
                         suffix="weight", sparse_size=self.sparse_kernel_size)
        if self.use_bias:
            self._add_weight(shape=(self.num_filters, ), suffix="bias")
        half_k = (self.kernel_size - 1) // 2
        h_ = h if self.padding == "same" else h - 2 * half_k
        w_ = w if self.padding == "same" else w - 2 * half_k
        output_shape = (batch_size, ceil(h_ / self.stride), ceil(w_ / self.stride), self.num_filters)
        return self._produce_output(shape=output_shape)


class DWConv2D(OperatorDesc):
    """
    Depthwise 2D Convolution (nb. does not include the 1x1 convolution that typically follows d/wise convolution).
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: str = "same", use_bias: bool = True,
                 batch_norm: bool = False, activation: Optional[str] = None,
                 sparse_kernel_size: Optional[int] = None, name: str = None):
        assert padding in ["valid", "same"]
        super().__init__("dw_conv2d" if name is None else name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.sparse_kernel_size = sparse_kernel_size

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 4
        batch_size, h, w, in_channels = x.shape
        self._add_weight(shape=(self.kernel_size, self.kernel_size, in_channels, 1),
                         suffix="weight", sparse_size=self.sparse_kernel_size)
        if self.use_bias:
            self._add_weight(shape=(in_channels, ), suffix="bias")
        half_k = (self.kernel_size - 1) // 2
        h_ = h if self.padding == "same" else h - 2 * half_k
        w_ = w if self.padding == "same" else w - 2 * half_k
        output_shape = (batch_size, ceil(h_ / self.stride), ceil(w_ / self.stride), in_channels)
        return self._produce_output(shape=output_shape)


class Dense(OperatorDesc):
    def __init__(self, units: int, preflatten_input: bool = False, use_bias: bool = True,
                 batch_norm: bool = False, activation: Optional[str] = None,
                 sparse_kernel_size: Optional[int] = None, name: str = None):
        super().__init__("dense" if name is None else name)
        self.units = units
        self.flatten = preflatten_input
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.sparse_kernel_size = sparse_kernel_size

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 2 or (self.flatten and len(x.shape) > 2)
        batch_size, input_dim = x.shape[0], np.prod(x.shape[1:])
        self._add_weight(shape=(input_dim, self.units), suffix="weight",
                         sparse_size=self.sparse_kernel_size)
        if self.use_bias:
            self._add_weight(shape=(self.units, ), suffix="bias")
        return self._produce_output(shape=(batch_size, self.units))


class Pool(OperatorDesc):
    def __init__(self, pool_size: Union[int, Tuple[int, int]], type: str, name: str = None):
        super().__init__("pool" if name is None else name)
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.type = type

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 4
        batch_size, h, w, in_channels = x.shape
        pool_h, pool_w = self.pool_size
        assert pool_h <= h and pool_w <= w, f"Can't apply {self.pool_size} pooling to {x.shape}"
        output_shape = (batch_size, ceil(h / pool_h), ceil(w / pool_w), in_channels)
        return self._produce_output(shape=output_shape)


class Add(OperatorDesc):
    def __init__(self, all_equal_shape: bool = True, name: str = None):
        super().__init__("add" if name is None else name)
        # if `all_equal_shape` is False, inputs can be of different sizes (but same dimensionality) and will be padded
        self.all_equal_shape = all_equal_shape

    def __call__(self, xs: List[TensorDesc]):
        super().__call__(xs)

        def all_equal(l):
            return l[1:] == l[:-1]

        assert len(xs) >= 2

        if self.all_equal_shape:
            assert all_equal([x.shape for x in xs])
            output_shape = xs[0].shape
        else:
            assert all_equal([len(x.shape) for x in xs])
            output_shape = [1, ] * len(xs[0].shape)
            for i in range(len(output_shape)):
                output_shape[i] = max(x.shape[i] for x in xs)
            output_shape = tuple(output_shape)
        return self._produce_output(shape=output_shape)
