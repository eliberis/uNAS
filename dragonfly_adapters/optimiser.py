from dragonfly.opt import gpb_acquisitions
from dragonfly.opt.gp_bandit import get_all_mf_cp_gp_bandit_args, CPGPBandit
from dragonfly.utils.option_handler import load_options
from dragonfly.utils.general_utils import update_pareto_set
from argparse import Namespace


def get_optimiser_options():
    options = load_options(get_all_mf_cp_gp_bandit_args())
    options.acq = "custom_mobo_ts"
    return options


class Optimiser(CPGPBandit):
    def __init__(self, *args, **kwargs):
        self.pareto_front_values = []
        self.pareto_front_points = []
        super().__init__(*args, **kwargs)

    def update_pareto_front(self, qinfo):
        value = [qinfo.true_val] + [-r for r in qinfo.point[0].resource_features]
        self.pareto_front_values, self.pareto_front_points = update_pareto_set(
            self.pareto_front_values, self.pareto_front_points, value, qinfo)

    def _child_handle_data_loaded_from_file(self, loaded_data_from_file):
        # Fixes a bug where the model is not updated properly when data is loaded from a file.
        # (The same bug should exist when loading from the options object, but we don't use that functionality.)
        n = super()._child_handle_data_loaded_from_file(loaded_data_from_file)
        # Recover qinfo objects
        qinfos = []
        for i in range(n):
            qinfo = Namespace(point=self.prev_eval_points[i],
                              val=self.prev_eval_vals[i],
                              true_val=self.prev_eval_true_vals[i])
            if self.func_caller.is_mf():
                qinfo.fidel = self.prev_eval_fidels[i]
            qinfos.append(qinfo)
            self.update_pareto_front(qinfo)
        self._add_data_to_model(qinfos)
        return n

    def _update_history(self, qinfo):
        super()._update_history(qinfo)
        self.update_pareto_front(qinfo)

    def _asynchronous_run_experiment_routine(self):
        # This differs from the original implementation by the reordering of
        # `_determine_next_query` and `_wait_for_a_free_worker`, which allows us
        # to determine the next evaluation paint on the main process in parallel with
        # point evaluation on a Ray worker process.
        qinfo = self._determine_next_query()
        self._wait_for_a_free_worker()
        if self.experiment_caller.is_mf() and not hasattr(qinfo, 'fidel'):
            qinfo.fidel = self.experiment_caller.fidel_to_opt
        self._dispatch_single_experiment_to_worker_manager(qinfo)
        self.step_idx += 1

    def _determine_next_query(self):
        """ Determine the next point for evaluation. """
        anc_data = self._get_ancillary_data_for_acquisition("ei")
        anc_data.curr_acq = "custom_mobo_ts"
        select_pt_func = self.func_caller.scoring_fn
        prev_evaluations = self.history.query_qinfos
        qinfo = Namespace(curr_acq=anc_data.curr_acq,
                          hp_tune_method=self.gp_processor.hp_tune_method)
        if self.is_an_mf_method():
            assert self.options.mf_strategy == 'boca'
            next_eval_fidel, next_eval_point = \
                gpb_acquisitions.boca(lambda *args: select_pt_func(*args, prev_evaluations=prev_evaluations),
                                      self.gp, anc_data, self.func_caller)
            qinfo.fidel = next_eval_fidel
            qinfo.point = next_eval_point
        else:
            qinfo.point = select_pt_func(self.gp, anc_data, prev_evaluations=prev_evaluations)
        return qinfo

    def optimise(self, max_capital):
        super().optimise(max_capital)

        def process(qinfo):
            nn = qinfo.point[0]
            info = {"architecture": nn.arch, "test_error": nn.test_error}
            for i, objective in enumerate(["val_error", "peak_memory_usage", "model_size", "macs"]):
                info[objective] = -qinfo.val[i]
            return info

        return [process(p) for p in self.pareto_front_points]
