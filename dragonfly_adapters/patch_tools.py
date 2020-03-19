import dragonfly.nn.nn_modifiers
import dragonfly.nn.otmann
import dragonfly.nn.nn_opt_utils
import dragonfly.opt.cp_ga_optimiser

from .nn_function_caller import NNFunctionCaller


def patch_with_func_caller(func_caller: NNFunctionCaller, options):
    """ Monkey-patches internal functions in Dragonfly to achieve necessary functionality. """
    # TODO: is it worth maintaining our own fork of Dragonfly instead?
    dragonfly.nn.nn_modifiers.get_single_nn_mutation_op = func_caller.get_single_nn_mutation_op
    dragonfly.nn.otmann.get_otmann_distance_computer_from_args = func_caller.get_otmann_distance_computer_from_args
    dragonfly.nn.nn_opt_utils.random_sample_from_nn_domain = func_caller.random_sample_from_nn_domain
