from config import BoundConfig
from configs.cnn_mnist_aging import search_algorithm, training_config, search_config

bound_config = BoundConfig(
    error_bound=0.035,
    peak_mem_bound=2500,
    model_size_bound=None,
    mac_bound=30000000
)
