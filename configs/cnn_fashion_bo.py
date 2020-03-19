from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import AdamW

from config import TrainingConfig, BayesOptConfig, BoundConfig
from dataset import FashionMNIST
from cnn import CnnSearchSpace

training_config = TrainingConfig(
    dataset=FashionMNIST(),
    epochs=75,
    optimizer=lambda: AdamW(lr=0.001, weight_decay=1e-5),
    callbacks=lambda: [LearningRateScheduler(lambda e: 0.001 if e < 25 else 0.00025)]
)

search_config = BayesOptConfig(
    search_space=CnnSearchSpace(),
    starting_points=15,
    checkpoint_dir="artifacts/cnn_fashion"
)

bound_config = BoundConfig(
    error_bound=0.10,
    peak_mem_bound=64000,
    model_size_bound=64000,
    mac_bound=1000000
)
