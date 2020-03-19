import tensorflow as tf

from config import TrainingConfig, BayesOptConfig, BoundConfig
from dataset import FashionMNIST
from mlp import MlpSearchSpace

training_config = TrainingConfig(
    dataset=FashionMNIST(),
    optimizer=lambda: tf.optimizers.Adam(learning_rate=0.001),
    callbacks=lambda: [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)]
)

search_config = BayesOptConfig(
    search_space=MlpSearchSpace(),
    starting_points=10,
    checkpoint_dir="artifacts/mlp_fashion"
)

bound_config = BoundConfig(
   error_bound=0.18,
   peak_mem_bound=200,
   model_size_bound=30000,
   mac_bound=30000
)
