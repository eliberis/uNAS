from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import SGDW

from cnn import CnnSearchSpace
from config import TrainingConfig
from configs.cnn_chars74k_aging import bound_config, search_config, search_algorithm
from dataset import Chars74K


def lr_schedule(epoch):
    if 0 <= epoch < 42:
        return 0.01
    return 0.005


training_config = TrainingConfig(
    dataset=Chars74K("/datasets/chars74k", img_size=(32, 32), binary=True),
    epochs=60,
    batch_size=128,
    optimizer=lambda: SGDW(lr=0.01, momentum=0.9, weight_decay=4e-5),
    callbacks=lambda: [LearningRateScheduler(lr_schedule)]
)
