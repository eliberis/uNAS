from abc import ABC, abstractmethod

from tensorflow.keras.models import Model
from resource_models.graph import Graph


class Architecture(ABC):
    """Base class for all candidate architectures"""

    @abstractmethod
    def to_keras_model(self, input_shape, num_classes, inherit_weights_from=None, **kwargs) -> Model:
        """
        Constructs a Keras model for the candidate architecture
        :param input_shape: Shape of the input image (excl. batch dimension)
        :param num_classes: Number of output classes
        :param inherit_weights_from: If the candidate architecture is a morph of some other architecture,
                                     this parameter is set to the parent's Keras model
        :return: A tf.keras.Model object
        """
        pass

    @abstractmethod
    def to_resource_graph(self, input_shape, num_classes, **kwargs) -> Graph:
        pass

