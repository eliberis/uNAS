import logging
import numpy as np
from argparse import Namespace

from dragonfly.exd.exd_utils import get_cp_domain_initial_qinfos
from dragonfly.exd.experiment_caller import CPFunctionCaller, ExperimentCaller
from dragonfly.opt.blackbox_optimiser import BlackboxOptimiser, blackbox_opt_args
from dragonfly.opt.cp_ga_optimiser import get_default_mutation_op
from dragonfly.utils.option_handler import load_options, get_option_specs

from timeit import default_timer as timer

ga_specific_opt_args = [
    get_option_specs('population_size', False, 100,
                     'Population size.'),
    get_option_specs('sample_size', False, 15,
                     'Number of candidates to sample to determine the parent.'),
]

ga_opt_args = ga_specific_opt_args + blackbox_opt_args


class AgingEvoOptimiser(BlackboxOptimiser):
    def __init__(self, func_caller, worker_manager=None, mutation_op=None,
                 options=None, reporter=None, ask_tell_mode=False):
        options = load_options(ga_opt_args, partial_options=options)
        options.init_capital = options.population_size

        super().__init__(func_caller, worker_manager, model=None,
                         options=options, reporter=reporter,
                         ask_tell_mode=ask_tell_mode)
        self.mutation_op = mutation_op
        self.pool = []

    def _opt_method_set_up(self):
        self.method_name = 'AgingEvoGA'
        self.population_size = self.options.population_size
        self.sample_size = self.options.sample_size

    def _opt_method_optimise_initialise(self):
        """ No initialisation for GA. """
        pass

    def _add_data_to_model(self, qinfos):
        """ Update the optimisation model. """
        pass

    def _child_build_new_model(self):
        """ Build new optimisation model. """
        pass

    def _determine_next_query(self):
        all_prev_eval_points = self.prev_eval_points + self.history.query_points
        all_prev_eval_vals = self.prev_eval_vals + self.history.query_vals
        pool = [Namespace(point=p, value=v) for p, v in zip(all_prev_eval_points, all_prev_eval_vals)][-self.population_size:]

        sample = np.random.choice(pool, size=self.sample_size)
        parent = max(sample, key=lambda x: x.value)  # Pick the one with max value

        child = self.mutation_op([parent.point], [1])[0]
        return Namespace(point=child)

    def _determine_next_batch_of_queries(self, batch_size):
        qinfos = [self._determine_next_query() for _ in range(batch_size)]
        return qinfos

    def _get_method_str(self):
        return 'agingevoga'

    def is_an_mf_method(self):
        return False


class CPGAOptimiser(AgingEvoOptimiser):
    def __init__(self, func_caller, worker_manager=None, single_mutation_ops=None,
                 single_crossover_ops=None, options=None, reporter=None, ask_tell_mode=False):
        """ Constructor. """
        options = load_options(ga_opt_args, partial_options=options)
        super(CPGAOptimiser, self).__init__(func_caller, worker_manager,
                                            mutation_op=self._mutation_op,
                                            options=options, reporter=reporter, ask_tell_mode=ask_tell_mode)
        self._set_up_single_mutation_ops(single_mutation_ops)
        self._set_up_single_crossover_ops(single_crossover_ops)

    def _set_up_single_mutation_ops(self, single_mutation_ops):
        """ Set up mutation operations. """
        if single_mutation_ops is None:
            single_mutation_ops = [None] * self.domain.num_domains
        for idx, dom in enumerate(self.domain.list_of_domains):
            if single_mutation_ops[idx] is None:
                single_mutation_ops[idx] = get_default_mutation_op(dom)
        self.single_mutation_ops = single_mutation_ops

    def _set_up_single_crossover_ops(self, crossover_ops):
        """ Set up cross-over operations. """
        # pylint: disable=unused-argument
        self.crossover_ops = crossover_ops

    def _mutation_op(self, X, num_mutations):
        """ The mutation operator for the product domain. """
        if hasattr(num_mutations, '__iter__'):
            num_mutations_for_each_x = num_mutations
        else:
            choices_for_each_mutation = np.random.choice(len(X), num_mutations, replace=True)
            num_mutations_for_each_x = [np.sum(choices_for_each_mutation == i) for i
                                        in range(len(X))]
        ret = []
        # Now extend
        for idx in range(len(X)):
            ret.extend(self._get_mutation_for_single_x(X[idx],
                                                       num_mutations_for_each_x[idx]))
        np.random.shuffle(ret)
        return ret

    def _get_mutation_for_single_x(self, x, num_mutations):
        """ Gets the mutation for single x. """
        ret = []
        for _ in range(num_mutations):
            curr_mutation = []
            for idx, elem in enumerate(x):
                curr_mutation.append(self.single_mutation_ops[idx](elem))
            ret.append(curr_mutation)
        return ret

    def _get_initial_qinfos(self, num_init_evals, *args, **kwargs):
        """ Gets num_init_evals initial points. """
        return get_cp_domain_initial_qinfos(self.domain, num_init_evals,
                                            dom_euclidean_sample_type='latin_hc',
                                            dom_integral_sample_type='latin_hc',
                                            dom_nn_sample_type='rand', *args, **kwargs)


def build_aging_cp_ga_optimiser(func_caller, cp_domain, worker_manager, max_capital,
                                mode='asy', orderings=None, single_mutation_ops=None,
                                single_crossover_ops=None, options=None,
                                reporter="silent"):
    """ A GA optimiser on Cartesian product space from the function caller. """

    log = logging.getLogger("AgingGA Optimiser")

    if not isinstance(func_caller, ExperimentCaller):
        func_caller = CPFunctionCaller(func_caller, cp_domain, domain_orderings=orderings)
    options = load_options(ga_opt_args, partial_options=options)
    options.mode = mode
    options.capital_type = 'return_value'
    options.population_size = 100
    options.sample_size = 25

    start_time = timer()
    log.info(f"Starting GA optimisation with capital {max_capital} ({options.capital_type}).")
    result = CPGAOptimiser(func_caller, worker_manager,
                           single_mutation_ops=single_mutation_ops,
                           single_crossover_ops=single_crossover_ops,
                           options=options, reporter=reporter).optimise(max_capital)
    end_time = timer()
    log.info(f"GA optimisation finished, took {(end_time - start_time):.2f}s.")
    return result
