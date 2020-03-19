import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import TrainingConfig, BayesOptConfig, BoundConfig
from dataset import MNIST
from cnn import CnnSearchSpace

training_config = TrainingConfig(
    dataset=MNIST(binary=True),
    epochs=25,
    batch_size=128,
    optimizer=lambda: tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
    callbacks=lambda: [EarlyStopping(patience=10, min_delta=0.005)],
)

search_config = BayesOptConfig(
    search_space=CnnSearchSpace(),
    starting_points=10,
    checkpoint_dir="artifacts/cnn_mnist"
)

bound_config = BoundConfig(
    error_bound=0.01,
    peak_mem_bound=2000,
    model_size_bound=2000,
    mac_bound=1000000
)
