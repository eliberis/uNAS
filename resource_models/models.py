import sys
import numpy as np

from functools import lru_cache
from typing import List, Union
from .graph import Graph, OperatorDesc
from .ops import Conv2D, DWConv2D, Pool, Dense, Add, Input


def peak_memory_usage(g: Graph, exclude_weights=True, exclude_inputs=True):
    def occupies_memory(x):
        is_input = (not x.is_constant) and x.producer is None
        is_weight = x.is_constant
        return not ((exclude_inputs and is_input) or (exclude_weights and is_weight))

    def sum_of_tensor_sizes(tensors):
        return sum(x.size for x in tensors if occupies_memory(x))

    @lru_cache(maxsize=None)
    def mem(tensors):
        # Computes the peak memory usage of a runtime system that computes all tensors in a set `tensors`.
        constants = [t for t in tensors if t.producer is None]
        if constants:
            upstream_mem_use, op_order = mem(frozenset(t for t in tensors if t.producer is not None))
            return sum_of_tensor_sizes(constants) + upstream_mem_use, op_order
        if not tensors:
            return 0, []

        min_use = sys.maxsize  # A reasonably large integer
        op_order = []
        # For each of tensors in our working set, we try to unapply the operator that produced it
        for t in tensors:
            rest = tensors - {t}
            # We constrain the search to never consider evaluating an operator (`t.producer`) more than once ---
            # so we prevent cases where we consider unapplying `t.producer` but it's actually necessary for other
            # tensors in the working set.
            if any(t in r.predecessors for r in rest):
                continue
            inputs = frozenset(t.producer.inputs)
            new_set = rest | inputs
            upstream_mem_use, operators = mem(new_set)

            def last_use_point(i):
                return all(o in operators for o in i.consumers if o != t.producer)

            if isinstance(t.producer, Add) and any(i.shape == t.shape and last_use_point(i) for i in inputs):
                # When evaluating Add, instead of creating a separate output buffer, we can accumulate into one
                # of its inputs, provided it's no longer used anywhere else (either not consumed elsewhere or its
                # other consumers have already been evaluated).
                current_mem_use = sum_of_tensor_sizes(new_set)
            else:
                current_mem_use = sum_of_tensor_sizes(new_set | {t})

            mem_use = max(upstream_mem_use, current_mem_use)
            if mem_use < min_use:
                min_use = mem_use
                op_order = operators + [t.producer]
        return min_use, op_order

    mem.cache_clear()
    if len(g.outputs) == 0:
        raise ValueError("Provided graph has no outputs. Did you call `g.add_output(...)`?.")
    peak_usage, _ = mem(frozenset(g.outputs))
    return peak_usage


def model_size(g: Graph, sparse=False):
    return sum(x.size if not sparse else (x.sparse_size or x.size)
               for x in g.tensors.values() if x.is_constant)


def macs(g: Union[Graph, OperatorDesc, List[OperatorDesc]]):
    return inference_latency(g, mem_access_weight=0, compute_weight=1)


def inference_latency(g: Union[Graph, OperatorDesc, List[OperatorDesc]],
                      mem_access_weight=0, compute_weight=1):
    if isinstance(g, Graph):
        ops = g.operators.values()
    elif isinstance(g, OperatorDesc):
        ops = [g]
    else:
        ops = g

    latency = 0
    for op in ops:
        loads, compute = 0, 0
        if isinstance(op, Conv2D):
            k_h, k_w, i_c, o_c = op.inputs[1].shape
            n, o_h, o_w, _ = op.output.shape
            work = n * o_h * o_w * o_c * k_h * k_w * i_c
            loads, compute = 2 * work, work
            if op.use_bias:
                loads += n * o_h * o_w * o_c
        if isinstance(op, DWConv2D):
            k_h, k_w, c, _ = op.inputs[1].shape
            n, o_h, o_w, _ = op.output.shape
            work = n * c * o_h * o_w * k_h * k_w
            loads, compute = 2 * work, work
            if op.use_bias:
                loads += n * c * o_h * o_w
        if isinstance(op, Pool):
            n, o_h, o_w, c = op.output.shape
            pool_h, pool_w = op.pool_size
            work = n * o_h * o_w * c * pool_h * pool_w
            loads, compute = work, work
        if isinstance(op, Dense):
            n, _ = op.output.shape
            in_dim, out_dim = op.inputs[1].shape
            work = n * in_dim * out_dim
            loads, compute = 2 * work, work
            if op.use_bias:
                loads += n * out_dim
        if isinstance(op, Add):
            # TODO: not precise when inputs are of different shapes
            num_terms = len(op.inputs)
            elems_per_term = np.prod(op.output.shape)
            loads = num_terms * elems_per_term
            compute = (num_terms - 1) * elems_per_term
        latency += mem_access_weight * loads + compute_weight * compute
    return latency


def main():
    g = Graph(element_type=np.uint8)
    with g.as_current():
        x = Input(shape=(1, 32, 32, 3))
        x = Conv2D(filters=5, kernel_size=3, stride=1)(x)
        x = Dense(units=10, preflatten_input=True)(x)
        g.add_output(x)

    print("PMU:", peak_memory_usage(g))
    print("Size:", model_size(g))
    print("Latency:", inference_latency(g))


if __name__ == "__main__":
    main()
