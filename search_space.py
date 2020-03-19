from tensorflow.keras.models import Model
from abc import ABC, abstractmethod
from typing import List, Union, TypeVar, Dict
from architecture import Architecture
from resource_models.graph import Graph
from schema_types import ValueType


ArchType = TypeVar('ArchType', bound=Architecture)
SchemaType = Dict[str, ValueType]


class SearchSpace(ABC):
    @property
    @abstractmethod
    def schema(self) -> SchemaType:
        pass

    @abstractmethod
    def random_architecture(self) -> ArchType:
        pass

    @abstractmethod
    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        pass

    def to_keras_model(self, arch: ArchType, *args, **kwargs) -> Model:
        return arch.to_keras_model(*args, **kwargs)

    def to_resource_graph(self, arch: ArchType, *args, **kwargs) -> Graph:
        return arch.to_resource_graph(*args, **kwargs)
