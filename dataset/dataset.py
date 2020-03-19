import tensorflow as tf
import numpy as np

from typing import Tuple
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from functools import lru_cache


class Dataset(ABC):
    @abstractmethod
    def train_dataset(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def test_dataset(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def validation_dataset(self) -> tf.data.Dataset:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, int, int]:
        pass

    @lru_cache(maxsize=None)
    def class_weight(self):
        classes = np.arange(self.num_classes)
        all_data = self.train_dataset().concatenate(self.validation_dataset()).concatenate(self.test_dataset())
        labels = np.array([y for x, y in all_data.as_numpy_iterator()])
        return compute_class_weight("balanced", classes, labels)

    @staticmethod
    def _train_test_split(X, y, split_size, random_state=0, stratify=None):
        if split_size > 0:
            return train_test_split(X, y, test_size=split_size, random_state=random_state, stratify=stratify)
        else:
            return X, np.zeros((0, ) + X.shape[1:]), y, np.zeros((0, ) + y.shape[1:])
