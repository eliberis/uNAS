import ray
import logging
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Optional

from architecture import Architecture
from config import AgingEvoConfig, TrainingConfig, BoundConfig
from model_trainer import ModelTrainer
from resource_models.models import peak_memory_usage, model_size, inference_latency
from utils import Scheduler, debug_mode


@dataclass
class ArchitecturePoint:
    arch: Architecture
    sparsity: Optional[float] = None


@dataclass
class EvaluatedPoint:
    point: ArchitecturePoint
    val_error: float
    test_error: float
    resource_features: List[Union[int, float]]


@ray.remote(num_gpus=0 if debug_mode() else 1, num_cpus=1 if debug_mode() else 6)
class GPUTrainer:
    def __init__(self, search_space, trainer):
        self.trainer = trainer
        self.ss = search_space
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def evaluate(self, point):
        log = logging.getLogger("Worker")

        data = self.trainer.dataset
        arch = point.arch
        model = self.ss.to_keras_model(arch, data.input_shape, data.num_classes)
        results = self.trainer.train_and_eval(model, sparsity=point.sparsity)
        val_error, test_error = results["val_error"], results["test_error"]
        rg = self.ss.to_resource_graph(arch, data.input_shape, data.num_classes,
                                       pruned_weights=results["pruned_weights"])
        unstructured_sparsity = self.trainer.config.pruning and \
                                not self.trainer.config.pruning.structured
        resource_features = [peak_memory_usage(rg), model_size(rg, sparse=unstructured_sparsity),
                             inference_latency(rg, compute_weight=1, mem_access_weight=0)]
        log.info(f"Training complete: val_error={val_error:.4f}, test_error={test_error:.4f}, "
                 f"resource_features={resource_features}.")
        return EvaluatedPoint(point=point,
                              val_error=val_error, test_error=test_error,
                              resource_features=resource_features)


class AgingEvoSearch:
    def __init__(self,
                 experiment_name: str,
                 search_config: AgingEvoConfig,
                 training_config: TrainingConfig,
                 bound_config: BoundConfig):
        self.log = logging.getLogger(name=f"AgingEvoSearch [{experiment_name}]")
        self.config = search_config
        self.trainer = ModelTrainer(training_config)

        self.root_dir = Path(search_config.checkpoint_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        if training_config.pruning and not training_config.pruning.structured:
            self.log.warning("For unstructured pruning, we can only meaningfully use the model "
                             "size resource metric.")
            bound_config.peak_mem_bound = None
            bound_config.mac_bound = None
        self.pruning = training_config.pruning

        # We establish an order of objective in the feature vector, all functions must ensure the order is the same
        self.constraint_bounds = [bound_config.error_bound,
                                  bound_config.peak_mem_bound,
                                  bound_config.model_size_bound,
                                  bound_config.mac_bound]

        self.history: List[EvaluatedPoint] = []
        self.population: List[EvaluatedPoint] = []

        self.population_size = search_config.population_size
        self.initial_population_size = search_config.initial_population_size or self.population_size
        self.rounds = search_config.rounds
        self.sample_size = search_config.sample_size
        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
        self.max_parallel_evaluations = search_config.max_parallel_evaluations or num_gpus

    def save_state(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.history, f)
        self.log.info(f"Saved {len(self.history)} architectures to {file}.")

    def load_state(self, file):
        with open(file, "rb") as f:
            self.history = pickle.load(f)
        self.population = self.history[-self.population_size:]
        self.log.info(f"Loaded {len(self.history)} architectures from {file}")

    def maybe_save_state(self, save_every):
        if len(self.history) % save_every == 0:
            file = self.root_dir / f"{self.experiment_name}_agingevosearch_state.pickle"
            self.save_state(file.as_posix())

    def get_mo_fitness_fn(self):
        lambdas = np.random.uniform(low=0.0, high=1.0, size=4)

        def normalise(x, l=0, u=1, cap=10.0):
            return min((x - l) / (u - l), cap)

        def fitness(i: EvaluatedPoint):
            features = [i.val_error] + i.resource_features
            # All objectives must be non-negative and scaled to the same magnitude of
            # between 0 and 1. Values that exceed required bounds will therefore be mapped
            # to a factor > 1, and be hit by the optimiser first.
            normalised_features = [normalise(f, u=c) / l
                                   for f, c, l in zip(features, self.constraint_bounds, lambdas)
                                   if c is not None]  # bound = None means ignored objective
            return -max(normalised_features)  # Negated, due to function being maximised
        return fitness

    def bounds_log(self, history_size=25):
        def to_feature_vector(i):
            return [i.val_error] + i.resource_features
        within_bounds = \
            [all(o <= b
                 for o, b in zip(to_feature_vector(i), self.constraint_bounds)
                 if b is not None)
             for i in self.history[-history_size:]]
        self.log.info(f"In bounds: {sum(within_bounds)} within "
                      f"the last {len(within_bounds)} architectures.")

    def evolve(self, point: ArchitecturePoint):
        arch = np.random.choice(self.config.search_space.produce_morphs(point.arch))
        sparsity = None
        if self.pruning:
            incr = np.random.normal(loc=0.0, scale=0.05)
            sparsity = np.clip(point.sparsity + incr,
                               self.pruning.min_sparsity, self.pruning.max_sparsity)
        return ArchitecturePoint(arch=arch, sparsity=sparsity)

    def random_sample(self):
        arch = self.config.search_space.random_architecture()
        sparsity = None
        if self.pruning:
            sparsity = np.random.uniform(self.pruning.min_sparsity, self.pruning.max_sparsity)
        return ArchitecturePoint(arch=arch, sparsity=sparsity)

    def search(self, load_from: str = None, save_every: int = None):
        if load_from:
            self.load_state(load_from)

        ray.init(local_mode=debug_mode())

        trainer = ray.put(self.trainer)
        ss = ray.put(self.config.search_space)
        scheduler = Scheduler([GPUTrainer.remote(ss, trainer)
                               for _ in range(self.max_parallel_evaluations)])
        self.log.info(f"Searching with {self.max_parallel_evaluations} workers.")

        def should_submit_more(cap):
            return (len(self.history) + scheduler.pending_tasks() < cap) \
               and scheduler.has_a_free_worker()

        def point_number():
            return len(self.history) + scheduler.pending_tasks() + 1

        while len(self.history) < self.initial_population_size:
            if should_submit_more(cap=self.initial_population_size):
                self.log.info(f"Populating #{point_number()}...")
                scheduler.submit(self.random_sample())
            else:
                info = scheduler.await_any()
                self.population.append(info)
                self.history.append(info)
                self.maybe_save_state(save_every)

        while len(self.history) < self.rounds:
            if should_submit_more(cap=self.rounds):
                self.log.info(f"Searching #{point_number()}...")
                sample = np.random.choice(self.population, size=self.sample_size)
                parent = max(sample, key=self.get_mo_fitness_fn())

                scheduler.submit(self.evolve(parent.point))
            else:
                info = scheduler.await_any()
                self.population.append(info)
                while len(self.population) > self.population_size:
                    self.population.pop(0)
                self.history.append(info)
                self.maybe_save_state(save_every)
                self.bounds_log()
