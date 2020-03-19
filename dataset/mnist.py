import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

from .dataset import Dataset
from .utils import with_probability, random_shift, random_rotate


class MNISTBase(Dataset, ABC):
    def __init__(self, train, test, validation_split=0.1, seed=0, binary=False):
        self.binary = binary
        (x_train, y_train), (x_test, y_test) = train, test

        # Preprocessing
        def preprocess(x, y):
            x = np.expand_dims(x, axis=-1).astype('float32') / 255
            if binary:
                y = (y < 5).astype(np.uint8)
            return x, y

        x_train, y_train = preprocess(x_train, y_train)
        x_train, x_val, y_train, y_val = \
            self._train_test_split(x_train, y_train, split_size=validation_split, random_state=seed, stratify=y_train)

        self.train = (x_train, y_train)
        self.val = (x_val, y_val)
        self.test = preprocess(x_test, y_test)

    @abstractmethod
    def get_augment_fn(self):
        pass

    def train_dataset(self):
        train_data = tf.data.Dataset.from_tensor_slices(self.train)
        train_data = train_data.map(self.get_augment_fn(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return train_data

    def validation_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.val)

    def test_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.test)

    @property
    def num_classes(self):
        return 10 if not self.binary else 2

    @property
    def input_shape(self):
        return (28, 28, 1)


class FashionMNIST(MNISTBase):
    def __init__(self, **kwargs):
        train, test = tf.keras.datasets.fashion_mnist.load_data()
        super().__init__(train, test, **kwargs)

    def get_augment_fn(self):
        def augment(x, y):
            # TODO: random zoom? cutout?
            x = tf.image.random_flip_left_right(x)
            x = with_probability(0.3, lambda: random_rotate(x, 0.2), lambda: x)
            x = random_shift(x, 2, 2)
            return x, y
        return augment


class MNIST(MNISTBase):
    def __init__(self, **kwargs):
        train, test = tf.keras.datasets.mnist.load_data()
        super().__init__(train, test, **kwargs)

    def get_augment_fn(self):
        def augment(x, y):
            # TODO: random zoom? cutout?
            x = with_probability(0.3, lambda: random_rotate(x, 0.2), lambda: x)
            x = random_shift(x, 2, 2)
            return x, y
        return augment
