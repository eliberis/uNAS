from config import DistillationConfig
from configs.cnn_chars74k_aging import training_config, search_config, bound_config, search_algorithm

training_config.distillation = DistillationConfig(
    distill_from="cnn-chars74k.h5"
)
training_config.batch_size = 75
