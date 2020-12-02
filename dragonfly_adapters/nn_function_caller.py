import logging
from typing import List

import numpy as np
from argparse import Namespace

from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.opt.gpb_acquisitions import maximise_acquisition, get_gp_sampler_for_parallel_strategy
from dragonfly.utils.reporters import BasicReporter
from dragonfly_adapters.neural_network import NeuralNetworkWrapper
from dragonfly.nn.nn_domains import NNDomain
from dragonfly.exd.domains import CartesianProductDomain, IntegralDomain, EuclideanDomain

from model_trainer import ModelTrainer
from search_space import SearchSpace

from .kernel import get_distance_computer
from .aging_ga_optimiser import build_aging_cp_ga_optimiser
from .local_ga_optimiser import build_local_cp_ga_optimiser


class NNFunctionCaller(CPFunctionCaller):
    def __init__(self, model_trainer: ModelTrainer, space: SearchSpace, constraint_bounds,
                 acq_opt_method="aging", is_mf=False):
        self.model_trainer = model_trainer
        self.space = space
        self.constraint_bounds = constraint_bounds
        self.acq_opt_method = acq_opt_method

        self.log = logging.getLogger(name="NNExperiment")
        pruning = self.model_trainer.config.pruning

        opt_domain = {"network": NNDomain("unas-net", constraint_checker=lambda x: True)}
        if pruning:
            opt_domain["sparsity"] = EuclideanDomain([[pruning.min_sparsity, pruning.max_sparsity]])

        domains = []
        dim_ordering = []
        index_ordering = []
        kernel_ordering = []
        name_ordering = []
        raw_name_ordering = []

        for i, (k, d) in enumerate(opt_domain.items()):
            nested = False if isinstance(d, NNDomain) else True
            domains.append(d)
            dim_ordering.append([''] if nested else '')
            index_ordering.append([i] if nested else i)
            kernel_ordering.append('')
            name_ordering.append([k] if nested else k)
            raw_name_ordering.append(k)

        # Multifidelity set-up: fidelity space is a cartesian product domain with one integer,
        # which determines the number of epochs a model will be trained for.
        min_epochs = max(1, int(0.3 * model_trainer.config.epochs))
        max_epochs = model_trainer.config.epochs
        self.fidel_space = CartesianProductDomain([IntegralDomain([[min_epochs, max_epochs]])])
        self.fidel_space_orderings = Namespace(dim_ordering=[['']], index_ordering=[[0]], kernel_ordering=[''],
                                               name_ordering=[['epochs']], raw_name_ordering=['epochs'])

        class LogReporter(BasicReporter):
            def __init__(self):
                super().__init__(None)
                self.log = logging.getLogger(name="Dragonfly")

            def write(self, msg: str, *_):
                if msg.endswith("\n"):
                    msg = msg[:-1]
                self.log.info(msg)

        self.reporter = LogReporter()
        super(NNFunctionCaller, self).__init__(
            self._validation_error if is_mf else (
                lambda *args, **kwargs: self._validation_error(None, *args, **kwargs)),
            descr='',
            domain=CartesianProductDomain(domains),
            domain_orderings=Namespace(dim_ordering=dim_ordering,
                                       index_ordering=index_ordering,
                                       kernel_ordering=kernel_ordering,
                                       name_ordering=name_ordering,
                                       raw_name_ordering=raw_name_ordering),
            fidel_to_opt=[[self.model_trainer.config.epochs]] if is_mf else None,
            fidel_space=self.fidel_space if is_mf else None,
            fidel_cost_func=self.fidelity_cost if is_mf else None,
            fidel_space_orderings=self.fidel_space_orderings if is_mf else None)

    def decode_point(self, point):
        x = self.get_raw_domain_point_from_processed(point)
        return {k: v for k, v in zip(self.domain_orderings.raw_name_ordering, x)}

    def fidelity_cost(self, fidel_point):
        epochs = self.get_raw_fidel_from_processed(fidel_point)[0]
        return epochs / self.model_trainer.config.epochs

    def random_architecture(self):
        return self.wrap_architecture(self.space.random_architecture())

    def wrap_architecture(self, arch):
        return NeuralNetworkWrapper(arch, self.space,
                                    self.model_trainer.dataset.input_shape,
                                    self.model_trainer.dataset.num_classes)

    def mutate(self, wrapped_nn: NeuralNetworkWrapper) -> NeuralNetworkWrapper:
        morphs = self.space.produce_morphs(wrapped_nn.arch)
        return self.wrap_architecture(np.random.choice(morphs))

    def mutate_all(self, wrapped_nn: NeuralNetworkWrapper) -> List[NeuralNetworkWrapper]:
        morphs = self.space.produce_morphs(wrapped_nn.arch)
        return [self.wrap_architecture(m) for m in morphs]

    def scoring_fn(self, error_gp, anc_data, *args, **kwargs):
        # The trick to do multiobjective optimisation: when evaluating points for the experiment, we only return
        # the error of a model, since that's the only thing we want to be modelled. But when BO chooses which
        # point/neural network to evaluate next, this acquisition function will take all objectives into account.
        lambdas = np.random.uniform(low=0.0, high=1.0, size=4)
        # TODO: change weight sampler and / or investigate other acqs?

        def normalise(x, l=0, u=1, cap=10.0):
            return min((x - l) / (u - l), cap)

        error_sampler = get_gp_sampler_for_parallel_strategy(error_gp, anc_data)

        def multiobjective_score(x):
            error = -error_sampler(x)[0]
            x = self.decode_point(x[0])
            features = [error] + x["network"].estimate_resource_features(sparsity=x.get("sparsity"))
            # All objectives must be non-negative and scaled to the same magnitude of between 0 and 1
            # Values that exceed required bounds will therefore be mapped to a factor > 1, and be hit by the
            # optimiser first.
            normalised_features = [normalise(f, u=c) / l
                                   for f, c, l in zip(features, self.constraint_bounds, lambdas)
                                   if c is not None]  # bound = None means ignored objective
            return -max(normalised_features)  # Negated, due to acq being maximised

        return self.maximise_acquisition(multiobjective_score, anc_data, *args, **kwargs)

    def maximise_acquisition(self, acq_fn, anc_data, *args, **kwargs):
        # Merged from multiple places in Dragonfly's code to select the right optimiser
        # Avoids patching "cp_ga_optimiser_from_proc_args" globally
        domain, max_evals = anc_data.domain, anc_data.max_evals
        obj_in_func_caller = CPFunctionCaller(lambda x: acq_fn([x]), domain, domain_orderings=None)
        worker_manager = SyntheticWorkerManager(1, time_distro='const')
        if "prev_evaluations" in kwargs:
            options = {"prev_evaluations": Namespace(qinfos=kwargs["prev_evaluations"])}
        else:
            options = {}

        if self.acq_opt_method == "aging":
            _, ga_max_pt, _ = \
                build_aging_cp_ga_optimiser(obj_in_func_caller, domain, worker_manager, max_evals,
                                            mode='asy', options=None)
        else:
            assert self.acq_opt_method == "local"
            _, ga_max_pt, _ = \
                build_local_cp_ga_optimiser(obj_in_func_caller, domain, self.mutate_all,
                                            worker_manager, max_evals, mode='asy', options=options)
        return ga_max_pt

    def _validation_error(self, fidel, point):
        # This is the experiment function --- the function called by the "function caller"
        self.log.info("Commencing training...")
        decoded_point = self.decode_point(point)
        epochs = self.get_raw_fidel_from_processed(fidel)[0] if fidel else None
        wrapped_nn = decoded_point["network"]
        sparsity = decoded_point.get("sparsity")
        results = self.model_trainer.train_and_eval(wrapped_nn.to_keras_model(self.space),
                                                    epochs=epochs, sparsity=sparsity)
        wrapped_nn.val_error = results["val_error"]
        wrapped_nn.test_error = results["test_error"]
        if sparsity:
            rg = wrapped_nn.to_resource_graph(self.space, pruned_weights=results["pruned_weights"])
            wrapped_nn.resource_features = wrapped_nn.compute_resource_features(rg, sparse=True)
        self.log.info(f"Training complete: val_error={wrapped_nn.val_error:.4f}, "
                      f"test_error={wrapped_nn.test_error:.4f}, "
                      f"resource_features={wrapped_nn.resource_features}.")
        return -wrapped_nn.val_error

    # All of the following functions are replacements for functions for Dragonfly modules
    # They are monkey-patched in in `patch_tools.py`
    def get_single_nn_mutation_op(self, *args):
        return lambda nn, *args: self.mutate(nn)

    def random_sample_from_nn_domain(self, type, num_samples, *args, **kwargs):
        return [self.random_architecture() for _ in range(num_samples)]

    def get_otmann_distance_computer_from_args(self, *args, **kwargs):
        return get_distance_computer()
