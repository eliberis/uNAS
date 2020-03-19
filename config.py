import tensorflow as tf

from typing import List, Callable, Optional
from dataclasses import dataclass
from search_space import SearchSpace
from dataset import Dataset


@dataclass
class DistillationConfig:
    distill_from: str  # a Path to a tf.keras.Model
    alpha: float = 0.3  # Weight factor for the distillation loss (D_KL between teacher and student)
    temperature: float = 1.0  # Softening factor for teacher's logits


@dataclass
class PruningConfig:
    structured: bool = False
    start_pruning_at_epoch: int = 0
    finish_pruning_by_epoch: int = None
    min_sparsity: float = 0
    max_sparsity: float = 0.995


@dataclass
class TrainingConfig:
    dataset: Dataset
    optimizer: Callable[[], tf.optimizers.Optimizer]
    callbacks: Callable[[], List[tf.keras.callbacks.Callback]]
    batch_size: int = 128
    epochs: int = 75
    distillation: Optional[DistillationConfig] = None  # No distillation if `None`
    use_class_weight: bool = False  # Compute and use class weights to re-balance the data
    pruning: Optional[PruningConfig] = None


@dataclass
class BayesOptConfig:
    search_space: SearchSpace
    # Enables multi-fidelity optimisation for the accuracy/error model. Note that this can
    # discard areas of the search space with low accuracy without taking other objectives into account.
    multifidelity: bool = False
    starting_points: int = 15
    rounds: int = 800
    checkpoint_dir: str = "artifacts"


@dataclass
class AgingEvoConfig:
    search_space: SearchSpace
    population_size: int = 100
    sample_size: int = 25
    initial_population_size: Optional[int] = None  # if None, equal to population_size
    rounds: int = 2000
    max_parallel_evaluations: Optional[int] = None
    checkpoint_dir: str = "artifacts"


@dataclass
class BoundConfig:
    # NAS will attempt to find models whose metrics are below the specified bounds.
    # Specify `None` if you're not interested in optimising a particular metric.
    error_bound: Optional[float] = None
    peak_mem_bound: Optional[int] = None
    model_size_bound: Optional[int] = None
    mac_bound: Optional[int] = None
