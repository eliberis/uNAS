import logging
import pickle
import time
from pathlib import Path

from dragonfly_adapters import NNFunctionCaller, patch_with_func_caller, get_optimiser_options, Optimiser
from config import BayesOptConfig, TrainingConfig, BoundConfig
from dragonfly_adapters.ray_worker_manager import RayWorkerManager
from model_trainer import ModelTrainer


class BayesOpt:
    def __init__(self,
                 experiment_name: str,
                 search_config: BayesOptConfig,
                 training_config: TrainingConfig,
                 bound_config: BoundConfig):
        assert search_config.starting_points >= 1

        self.log = logging.getLogger(name=f"BayesOpt [{experiment_name}]")
        self.config = search_config
        self.trainer = ModelTrainer(training_config)

        self.root_dir = Path(search_config.checkpoint_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        if training_config.pruning and not training_config.pruning.structured:
            self.log.warning("For unstructured pruning, we can only use the model size resource metric.")
            bound_config.peak_mem_bound = None
            bound_config.mac_bound = None

        # We establish an order of objective in the feature vector, all functions must ensure the order is the same
        self.constraint_bounds = [bound_config.error_bound, bound_config.peak_mem_bound,
                                  bound_config.model_size_bound, bound_config.mac_bound]

    def search(self, load_from: str = None, save_every: int = None):
        func_caller = NNFunctionCaller(self.trainer, self.config.search_space,
                                       self.constraint_bounds, self.acq_opt_method,
                                       is_mf=self.config.multifidelity)
        options = get_optimiser_options()

        options.init_set_to_fidel_to_opt_with_prob = 0.5
        options.capital_type = "return_value"
        options.init_capital = self.config.starting_points
        options.mode = "asy"
        # options.gpb_hp_tune_criterion = "ml"

        options.progress_load_from = load_from
        options.progress_save_to = (self.root_dir /
                                    f"{self.experiment_name}_bo_search_state.pickle").as_posix()
        options.progress_save_every = save_every

        # options.acq_opt_max_evals = 600

        patch_with_func_caller(func_caller, options)

        worker_manager = RayWorkerManager(None, default_func_caller=func_caller)

        opt = Optimiser(func_caller, worker_manager,
                        is_mf=func_caller.is_mf(),
                        reporter=func_caller.reporter,
                        options=options)

        start_time = time.time()
        pareto_architectures = opt.optimise(self.config.rounds)
        end_time = time.time()

        self.log.info(f"Search completed! Time taken (wall clock): {(end_time - start_time) / 60:.2f} min.")

        self.save_results(pareto_architectures,
                          output_path=(self.root_dir / f"{self.experiment_name}_pareto_archs.pickle").as_posix())

    def save_results(self, results, output_path):
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        self.log.info("Saved search results to: " + output_path)

    @property
    def acq_opt_method(self):
        return "aging"
