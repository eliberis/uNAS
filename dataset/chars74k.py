import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from pathlib import Path
from typing import Tuple
from .dataset import Dataset
from .utils import with_probability, random_rotate, random_shift


class Chars74K(Dataset):
    """Provides EnglishImg part of the Chars74K dataset."""
    def __init__(self, english_img_dir, img_size, validation_split=705, binary=False, seed=0):
        """
        Creates the dataset object. The dataset contains 7705 images, which will be split into groups of 5000, 705 and
        2000 images for training, validation and testing sets, respectively.
        :param english_img_dir: Path to the extracted EnglishImg Chars74K images. The loader will only look into
                                English/Img/GoodImg/Bmp subdirectory.
        :param img_size: Desired image size (h x w).
        :param seed: Seed for the train/val/test split.
        """
        self.img_size = img_size

        base_dir = Path(english_img_dir) / "English/Img/GoodImg/Bmp"
        if not base_dir.exists():
            raise ValueError("Specified dataset directory doesn't exist.")
        files = list(base_dir.rglob("img*.png"))

        # The dataset is quite small, so we can load it fully into RAM
        # Perform a stratified split into train, val and test sets
        self.binary = binary
        X, y = self._load_data(files)
        X, y = X.numpy(), y.numpy()

        if validation_split < 1.0:
            validation_split = int(validation_split * len(X))

        X_train_and_val, X_test, y_train_and_val, y_test = \
            self._train_test_split(X, y, split_size=2000, random_state=seed, stratify=y)

        X_train, X_val, y_train, y_val = \
            self._train_test_split(X_train_and_val, y_train_and_val, split_size=validation_split,
                                   random_state=seed, stratify=y_train_and_val)

        self.train_data = (X_train, y_train)
        self.valid_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

    def _load_data(self, filenames):
        images, labels = np.zeros((len(filenames),) + self.input_shape, dtype=np.float32), []
        for i, img_path in enumerate(filenames):
            # Read and preprocess the image
            img = tf.io.read_file(img_path.as_posix())
            img = tf.io.decode_png(img, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255

            label = int(img_path.name[3:6]) - 1  # Recover label number from the filename
            images[i] = img
            labels.append(label if not self.binary else int(label < 31))

        images = tf.stack(images)
        labels = tf.convert_to_tensor(labels)
        return images, labels

    def augment_func(self):
        def augment(x, y):
            # x = tfa.image.random_hsv_in_yiq(
            #     x,
            #     max_delta_hue=0.3,
            #     lower_saturation=0.2,
            #     upper_saturation=1.0,
            #     lower_value=0.7,
            #     upper_value=1.1)

            # x = with_probability(0.4, lambda: random_rotate(x, 0.2), lambda: x)

            shift_by = 0.1
            h_shift = int(self.img_size[0] * shift_by)
            w_shift = int(self.img_size[1] * shift_by)
            x = random_shift(x, h_shift, w_shift)

            # x = tf.clip_by_value(x, 0.0, 1.0)
            return x, y
        return augment

    def train_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(self.train_data)\
            .map(self.augment_func(), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def validation_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(self.valid_data)

    def test_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(self.test_data)

    @property
    def num_classes(self) -> int:
        return 62 if not self.binary else 2

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self.img_size + (3,)


