import tensorflow as tf

from .dataset import Dataset
from .speech_dataset import SpeechDataset

DEFAULT_WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']


class SpeechCommands(Dataset):
    def __init__(self, data_dir, words=None):
        self.words = words or DEFAULT_WORDS
        self.loader = SpeechDataset(words=self.words, data_dir=data_dir)
        self.data_dir = data_dir

    def train_dataset(self) -> tf.data.Dataset:
        return self.loader.training_dataset()

    def test_dataset(self) -> tf.data.Dataset:
        return self.loader.testing_dataset()

    def validation_dataset(self) -> tf.data.Dataset:
        return self.loader.validation_dataset()

    @property
    def num_classes(self):
        return self.loader.label_count()

    @property
    def input_shape(self):
        return self.loader.sample_shape()
