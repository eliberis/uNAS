from typing import List

from search_space import SearchSpace, ArchType, SchemaType
from .cnn_morphisms import produce_all_morphs
from .cnn_random_generators import random_arch
from .cnn_schema import get_schema


class CnnSearchSpace(SearchSpace):
    input_shape = None
    num_classes = None

    def __init__(self, dropout=0.0):
        self.dropout = dropout

    @property
    def schema(self) -> SchemaType:
        return get_schema()

    def random_architecture(self) -> ArchType:
        return random_arch()

    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        return produce_all_morphs(arch)

    def to_keras_model(self, arch: ArchType, input_shape=None, num_classes=None, **kwargs):
        input_shape = input_shape or self.input_shape
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes,
                                   dropout=self.dropout,
                                   **kwargs)
