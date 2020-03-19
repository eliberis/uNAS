from config import PruningConfig
from configs.cnn_cifar10_binary import training_config, bound_config, search_config, \
    search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=30,
    finish_pruning_by_epoch=60,
    min_sparsity=0.1,
    max_sparsity=0.6
)
