from config import PruningConfig, BoundConfig
from configs.cnn_chars74k_binary import training_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=20,
    finish_pruning_by_epoch=53,
    min_sparsity=0.1,
    max_sparsity=0.85
)

bound_config = BoundConfig(
    error_bound=0.26,
    peak_mem_bound=1500,
    model_size_bound=1000,
    mac_bound=1000000
)
