import tensorflow as tf
import numpy as np

from .dataset import Dataset
from .utils import with_probability, random_shift, random_rotate


class CIFAR10(Dataset):
    def __init__(self, validation_split=0.1, seed=0, binary=False):
        self.binary = binary
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Preprocessing
        def preprocess(x, y):
            x = x.astype('float32') / 255
            x = (x - np.array((0.4914, 0.4822, 0.4465))) / np.array((0.2023, 0.1994, 0.2010))
            if binary:
                y = (y < 5).astype(np.uint8)
            return x, y

        x_train, y_train = preprocess(x_train, y_train)
        x_train, x_val, y_train, y_val = \
            self._train_test_split(x_train, y_train, split_size=validation_split, random_state=seed, stratify=y_train)

        self.train = (x_train, y_train)
        self.val = (x_val, y_val)
        self.test = preprocess(x_test, y_test)

    @staticmethod
    def _augment(x, y):
        x = tf.image.random_flip_left_right(x)

        # x = tfa.image.random_hsv_in_yiq(
        #     x,
        #     max_delta_hue=0.5,
        #     lower_saturation=0.1,
        #     upper_saturation=0.9,
        #     lower_value=0.3,
        #     upper_value=0.8)

        # x = with_probability(0.6, lambda: random_rotate(x, 0.3), lambda: x)
        x = random_shift(x, 4, 4)
        # x = tf.clip_by_value(x, 0.0, 1.0)

        return x, y

    def train_dataset(self):
        train_data = tf.data.Dataset.from_tensor_slices(self.train)
        train_data = train_data.map(CIFAR10._augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
        return (32, 32, 3)
