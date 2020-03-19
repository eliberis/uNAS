from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import AdamW

from config import TrainingConfig, BoundConfig, AgingEvoConfig
from cnn import CnnSearchSpace
from dataset.speech_commands import SpeechCommands
from search_algorithms import AgingEvoSearch

search_algorithm = AgingEvoSearch


def lr_schedule(epoch):
    if 0 <= epoch < 20:
        return 0.0005
    if 20 <= epoch < 40:
        return 0.0001
    return 0.00002


training_config = TrainingConfig(
    dataset=SpeechCommands("/datasets/speech_commands_v0.02"),
    epochs=45,
    batch_size=50,
    optimizer=lambda: AdamW(lr=0.0005, weight_decay=1e-5),
    callbacks=lambda: [
        LearningRateScheduler(lr_schedule)
    ]
)

search_config = AgingEvoConfig(
    search_space=CnnSearchSpace(),
    rounds=2000,
    checkpoint_dir="artifacts/cnn_speech_commands"
)

bound_config = BoundConfig(
    error_bound=0.085,
    peak_mem_bound=60000,
    model_size_bound=40000,
    mac_bound=20000000,
)
