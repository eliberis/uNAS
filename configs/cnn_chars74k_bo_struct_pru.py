from config import PruningConfig
from configs.cnn_chars74k_bo import search_algorithm, search_config, training_config, \
    bound_config, training_config

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=20,
    finish_pruning_by_epoch=53,
    min_sparsity=0.1,
    max_sparsity=0.85
)
