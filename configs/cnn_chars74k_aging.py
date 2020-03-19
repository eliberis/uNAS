import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LearningRateScheduler

from config import TrainingConfig, AgingEvoConfig, BoundConfig
from dataset import Chars74K
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch


def lr_schedule(epoch):
    if 0 <= epoch < 35:
        return 0.01
    return 0.005


training_config = TrainingConfig(
    dataset=Chars74K("/datasets/chars74k", img_size=(48, 48)),
    epochs=60,
    batch_size=80,
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.01, momentum=0.9, weight_decay=0.0001),
    callbacks=lambda: [LearningRateScheduler(lr_schedule)]
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(dropout=0.15),
    checkpoint_dir="artifacts/cnn_chars74k"
)

bound_config = BoundConfig(
    error_bound=0.3,
    peak_mem_bound=10000,
    model_size_bound=20000,
    mac_bound=1000000
)
