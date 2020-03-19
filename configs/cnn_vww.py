import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import TrainingConfig, BayesOptConfig, BoundConfig
from dataset import VisualWakeWords
from cnn import CnnSearchSpace

training_config = TrainingConfig(
    dataset=VisualWakeWords("/datasets/COCO/2014/records"),
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.001, weight_decay=5e-5),
    batch_size=64,
    epochs=30,
    callbacks=lambda: [EarlyStopping(patience=15, verbose=1)]
)

search_config = BayesOptConfig(
    search_space=CnnSearchSpace(),
    starting_points=15,
    checkpoint_dir="artifacts/cnn_vww"
)

bound_config = BoundConfig(
    error_bound=0.10,
    peak_mem_bound=250000,
    model_size_bound=250000,
    mac_bound=30000000
)
