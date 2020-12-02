from typing import List

from search_space import SearchSpace, SchemaType, ArchType
from .mlp_random_generators import random_arch
from .mlp_morphisms import produce_all_morphs
from .mlp_schema import get_schema


class MlpSearchSpace(SearchSpace):
    input_shape = None
    num_classes = None

    @property
    def schema(self) -> SchemaType:
        return get_schema()

    def random_architecture(self) -> ArchType:
        return random_arch()

    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        return produce_all_morphs(arch)
