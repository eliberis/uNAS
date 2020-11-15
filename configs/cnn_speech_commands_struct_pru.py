from config import PruningConfig
from configs.cnn_speech_commands_aging import training_config, bound_config, search_config, \
    search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=20,
    finish_pruning_by_epoch=40,
    min_sparsity=0.10,
    max_sparsity=0.90
)
