from config import PruningConfig, BoundConfig
from configs.cnn_cifar10_binary import search_algorithm, search_config, training_config

training_config.pruning = PruningConfig(
    structured=False,
    start_pruning_at_epoch=30,
    finish_pruning_by_epoch=60,
    min_sparsity=0.1,
    max_sparsity=0.98
)

bound_config = BoundConfig(
    error_bound=0.25,
    peak_mem_bound=None,
    model_size_bound=2000,
    mac_bound=None
)
