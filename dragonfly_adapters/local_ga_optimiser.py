import logging
import numpy as np
from argparse import Namespace

from dragonfly.exd.experiment_caller import CPFunctionCaller, ExperimentCaller
from dragonfly.opt.blackbox_optimiser import BlackboxOptimiser, blackbox_opt_args
from dragonfly.opt.cp_ga_optimiser import get_default_mutation_op
from dragonfly.utils.option_handler import load_options

from timeit import default_timer as timer


class LocalCPGAOptimiser(BlackboxOptimiser):
    def __init__(self, func_caller, nn_morphs_op, worker_manager=None,
                 options=None, reporter=None, ask_tell_mode=False):
        """ Constructor. """
        self.nn_morphs_op = nn_morphs_op

        options = load_options(blackbox_opt_args, partial_options=options)
        assert options.prev_evaluations is not None
        self.pivots = [qinfo.point for qinfo in options.prev_evaluations.qinfos]
        options.prev_evaluations = None
        options.init_capital = 0

        super().__init__(func_caller, worker_manager, model=None,
                         options=options, reporter=reporter,
                         ask_tell_mode=ask_tell_mode)

        self.nn_idx = \
            [dom.get_type() for dom in self.domain.list_of_domains].index("neural_network")
        self.single_mutation_ops = \
            [get_default_mutation_op(dom) for dom in self.domain.list_of_domains]
        self.points_to_evaluate = None

    def mutation_op(self, x):
        nn_morphs = self.nn_morphs_op(x[self.nn_idx])
        mutations = []
        for m in nn_morphs:
            # Single-mutate other elements, keep nn from morphs list
            y = [m if i == self.nn_idx else self.single_mutation_ops[i](p)
                 for i, p in enumerate(x)]
            mutations.append(y)
        return mutations

    def perform_initial_queries(self):
        # No initial points should be loaded for this method
        pass

    def _opt_method_set_up(self):
        self.method_name = 'LocalCPGA'

    def _opt_method_optimise_initialise(self):
        pass

    def _add_data_to_model(self, qinfos):
        pass

    def _child_build_new_model(self):
        pass

    def _compute_query_points(self):
        idx = np.random.choice(len(self.pivots))
        parent = self.pivots[idx]
        self.points_to_evaluate = self.mutation_op(parent)
        self.options.max_num_steps = len(self.points_to_evaluate) + self.step_idx

    def _determine_next_query(self):
        if self.points_to_evaluate is None:
            self._compute_query_points()
        return Namespace(point=self.points_to_evaluate.pop(0))

    def _determine_next_batch_of_queries(self, batch_size):
        qinfos = [self._determine_next_query() for _ in range(batch_size)]
        return qinfos

    def _get_method_str(self):
        return 'LocalCPGA'

    def is_an_mf_method(self):
        return False


def build_local_cp_ga_optimiser(func_caller, cp_domain, nn_mutate_op, worker_manager, max_capital,
                                mode='asy', orderings=None, options=None, reporter="silent"):
    """ A GA optimiser on Cartesian product space from the function caller. """

    log = logging.getLogger("LocalGA Optimiser")

    if not isinstance(func_caller, ExperimentCaller):
        func_caller = CPFunctionCaller(func_caller, cp_domain, domain_orderings=orderings)
    options = load_options(blackbox_opt_args, partial_options=options)
    options.mode = mode
    options.capital_type = 'return_value'

    start_time = timer()
    log.info(f"Starting GA optimisation with capital {max_capital} ({options.capital_type}).")
    result = LocalCPGAOptimiser(func_caller, nn_mutate_op, worker_manager, options=options,
                                reporter=reporter).optimise(max_capital)
    end_time = timer()
    log.info(f"GA optimisation finished, took {(end_time - start_time):.2f}s.")
    return result
