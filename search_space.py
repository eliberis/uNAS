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

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    def to_keras_model(self, arch: ArchType, input_shape=None, num_classes=None, **kwargs) -> Model:
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes, **kwargs)

    def to_resource_graph(self, arch: ArchType,
                          input_shape=None, num_classes=None, **kwargs) -> Graph:
        return arch.to_resource_graph(input_shape=input_shape or self.input_shape,
                                      num_classes=num_classes or self.num_classes, **kwargs)
