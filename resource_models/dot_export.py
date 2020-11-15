from .graph import Graph, TensorDesc


def to_dot(g: Graph, output_file: str = None):
    identifiers = {}

    def get_id(name):
        if name not in identifiers:
            identifiers[name] = f"_{len(identifiers)}"
        return identifiers[name]

    def get_node_desc(t: TensorDesc):
        return "\n".join(
            [t.name, "Parameters:"] +
            (["None"] if t.producer is None
                         or len([i for i in t.producer.inputs if i.is_constant]) == 0 else
             [f"{i.name}: " +
              (f"{i.sparse_size}/{i.size // i.elem_size}"
               if i.sparse_size else f"{i.size // i.elem_size}")
              for i in t.producer.inputs if i.is_constant]) +
            [f"Shape: {t.shape[1:]}"])

    contents = f'''
        digraph g {{
            {"; ".join(f'{get_id(o.name)} [label="{get_node_desc(o)}", shape=box]' 
                       for o in g.tensors.values() if not o.is_constant)}
            {"; ".join(f"{get_id(i.name)} -> {get_id(o.name)}" 
                        for o in g.tensors.values() if o.producer is not None
                        for i in o.producer.inputs if not i.is_constant)}
        }}
    '''
    with open(output_file, "w") as f:
        f.write(contents)
