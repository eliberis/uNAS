from config import PruningConfig, BoundConfig
from configs.cnn_cifar10_aging import training_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=90,
    finish_pruning_by_epoch=120,
    min_sparsity=0.1,
    max_sparsity=0.90
)

bound_config = BoundConfig(
    error_bound=0.10,
    peak_mem_bound=50000,
    model_size_bound=50000,
    mac_bound=60000000
)
