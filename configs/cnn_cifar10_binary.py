import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import TrainingConfig, BayesOptConfig, BoundConfig, PruningConfig, AgingEvoConfig
from dataset import CIFAR10
from cnn import CnnSearchSpace
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch


def lr_schedule(epoch):
    if 0 <= epoch < 35:
        return 0.01
    if 35 <= epoch < 65:
        return 0.005
    return 0.001


training_config = TrainingConfig(
    dataset=CIFAR10(binary=True),
    optimizer=lambda: tfa.optimizers.SGDW(lr=0.01, momentum=0.9, weight_decay=1e-5),
    batch_size=128,
    epochs=80,
    callbacks=lambda: []
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    checkpoint_dir="artifacts/cnn_cifar10"
)

bound_config = BoundConfig(
    error_bound=0.3,
    peak_mem_bound=3000,
    model_size_bound=2000,
    mac_bound=1000000
)
