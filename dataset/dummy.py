from typing import Tuple

import tensorflow as tf

from .dataset import Dataset


class Dummy(Dataset):
    """A dummy lightweight dataset providing zeros for benchmarking purposes."""
    def __init__(self, img_shape, num_classes, length=None):
        self._img_shape = img_shape
        self._num_classes = num_classes
        self._length = length

    def _dataset(self):
        x = tf.zeros(self._img_shape, dtype=tf.float32)
        y = tf.zeros([], dtype=tf.int64)
        return tf.data.Dataset.from_tensors((x, y)).repeat(count=self._length)

    def train_dataset(self) -> tf.data.Dataset:
        return self._dataset()

    def validation_dataset(self) -> tf.data.Dataset:
        return self._dataset()

    def test_dataset(self) -> tf.data.Dataset:
        return self._dataset()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._img_shape
