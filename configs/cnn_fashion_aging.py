from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import SGDW

from config import TrainingConfig, BoundConfig, AgingEvoConfig
from dataset import FashionMNIST
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch


def lr_schedule(epoch):
    if 0 <= epoch < 25:
        return 0.01
    if 25 <= epoch < 35:
        return 0.005
    return 0.001


training_config = TrainingConfig(
    dataset=FashionMNIST(),
    batch_size=128,
    epochs=45,
    optimizer=lambda: SGDW(lr=0.01, momentum=0.9, weight_decay=1e-5),
    callbacks=lambda: [LearningRateScheduler(lr_schedule)]
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(dropout=0.15),
    checkpoint_dir="artifacts/cnn_fashion"
)

bound_config = BoundConfig(
    error_bound=0.10,
    peak_mem_bound=64000,
    model_size_bound=64000,
    mac_bound=30000000
)
