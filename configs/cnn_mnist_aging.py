import tensorflow_addons as tfa

from cnn import CnnSearchSpace
from config import AgingEvoConfig, TrainingConfig, BoundConfig
from dataset import MNIST
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch

training_config = TrainingConfig(
    dataset=MNIST(),
    epochs=30,
    batch_size=128,
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.005, momentum=0.9, weight_decay=4e-5),
    callbacks=lambda: [],
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    checkpoint_dir="artifacts/cnn_mnist"
)

bound_config = BoundConfig(
    error_bound=0.035,
    peak_mem_bound=2500,
    model_size_bound=4500,
    mac_bound=30000000
)
