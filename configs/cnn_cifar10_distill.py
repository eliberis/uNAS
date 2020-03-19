from config import DistillationConfig
from configs.cnn_cifar10_aging import training_config, search_config, bound_config, search_algorithm

training_config.distillation = DistillationConfig(
    distill_from="mobilenetv2-cifar10.h5"
)
