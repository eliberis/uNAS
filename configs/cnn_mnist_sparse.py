from config import PruningConfig, BoundConfig
from configs.cnn_mnist_aging import training_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=False,
    start_pruning_at_epoch=3,
    finish_pruning_by_epoch=18,
    min_sparsity=0.2,
    max_sparsity=0.98
)

bound_config = BoundConfig(
    error_bound=0.025,
    peak_mem_bound=None,
    model_size_bound=1000,
    mac_bound=None
)
