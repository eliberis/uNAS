from config import PruningConfig
from configs.cnn_mnist_bo import training_config, bound_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=3,
    finish_pruning_by_epoch=18,
    min_sparsity=0.05,
    max_sparsity=0.8
)
