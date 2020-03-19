from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path

from .dataset import Dataset
from .utils import with_probability, random_shift, random_rotate


class VisualWakeWords(Dataset):
    """Provides the Visual Wake Words dataset (https://arxiv.org/abs/1906.05721) from the generated TFRecords."""

    def __init__(self, records_dir, validation_split=10000, img_size=(192, 220)):
        self.records_dir = Path(records_dir)
        self.train_records = sorted(p.as_posix() for p in self.records_dir.glob("train.record-*"))
        self.val_records = sorted(p.as_posix() for p in self.records_dir.glob("val.record-*"))
        self.train_size = 82783
        self.val_size = 40504
        self.val_split = int(validation_split * self.train_size if validation_split < 1.0 else validation_split)
        self.img_size = img_size

    def parse_func(self):
        def parse(record):
            parsed = tf.io.parse_single_example(record, features={
                "image/class/label": tf.io.FixedLenFeature([], tf.int64),
                "image/encoded": tf.io.FixedLenFeature([], tf.string)
            })
            img = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
            label = parsed["image/class/label"]

            # Follow the "mobilenet_v1" preprocessing (= "inception" preprocessing) from:
            # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
            img = tf.image.central_crop(img, central_fraction=0.875)
            img = tf.image.resize(img, self.img_size)

            img = tf.cast(img, tf.float32) / 255   # Standard image conventions
            return img, label
        return parse

    def augment_func(self):
        def augment(x, y):
            x = tf.image.random_flip_left_right(x)

            # x = tfa.image.random_hsv_in_yiq(
            #     x,
            #     max_delta_hue=0.2,
            #     lower_saturation=0.5,
            #     upper_saturation=1.0,
            #     lower_value=0.7,
            #     upper_value=1.1)

            # x = with_probability(0.4, lambda: random_rotate(x, 0.15), lambda: x)

            shift_by = 0.1
            h_shift = int(self.img_size[0] * shift_by)
            w_shift = int(self.img_size[1] * shift_by)
            x = random_shift(x, h_shift, w_shift)

            # x = tf.clip_by_value(x, 0.0, 1.0)
            return x, y
        return augment

    def train_dataset(self) -> tf.data.Dataset:
        train_data = tf.data.TFRecordDataset(self.train_records)
        train_data = train_data.apply(tf.data.experimental.assert_cardinality(self.train_size)).skip(self.val_split)
        train_data = train_data.map(self.parse_func(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data = train_data.map(self.augment_func(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return train_data

    def validation_dataset(self) -> tf.data.Dataset:
        valid_data = tf.data.TFRecordDataset(self.train_records)
        valid_data = valid_data.apply(tf.data.experimental.assert_cardinality(self.train_size)).take(self.val_split)
        valid_data = valid_data.map(self.parse_func(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return valid_data

    def test_dataset(self) -> tf.data.Dataset:
        test_data = tf.data.TFRecordDataset(self.val_records)
        test_data = test_data.apply(tf.data.experimental.assert_cardinality(self.val_size))
        test_data = test_data.map(self.parse_func(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return test_data

    @property
    def num_classes(self):
        return 2

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self.img_size + (3, )
