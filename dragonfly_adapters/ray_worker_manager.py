import ray
import logging
import tensorflow as tf

from dragonfly.exd.worker_manager import AbstractWorkerManager

from utils import Scheduler, debug_mode


class Worker:
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def evaluate(self, func_caller, qinfo, **kwargs):
        return func_caller.eval_from_qinfo(qinfo, **kwargs)


class RayWorkerManager(AbstractWorkerManager):
    """
    Adapts Dragonfly's "workers" that execute a function at a point to use Ray's actors
    """

    def __init__(self, max_pending=None, default_func_caller=None, cpu_only=False):
        num_gpus_available = len(tf.config.experimental.list_physical_devices('GPU'))

        if max_pending is None:
            max_pending = 4 if cpu_only else max(1, num_gpus_available)
        super().__init__(max_pending)

        worker_resources = {
            "num_gpus": 0 if debug_mode() or cpu_only or num_gpus_available == 0 else 1,
        }

        self.worker_cls = ray.remote(**worker_resources)(Worker)

        if not ray.is_initialized():
            ray.init(local_mode=debug_mode(), ignore_reinit_error=True)

        self.max_pending = max_pending
        self.default_func_caller = default_func_caller
        if self.default_func_caller:
            self.caller_handle = ray.put(default_func_caller)
        self.scheduler = Scheduler(self.worker_cls.remote() for _ in range(max_pending))
        self.last_receive_time = 0

    def _child_reset(self):
        pass

    def close_all_queries(self):
        pass

    def a_worker_is_free(self, force_await=False):
        if not self.scheduler.has_a_free_worker() or force_await:
            qinfo = self.scheduler.await_any()
            if not hasattr(qinfo, 'true_val'):
                qinfo.true_val = qinfo.val

            if hasattr(qinfo, 'caller_eval_cost') and qinfo.caller_eval_cost is not None:
                qinfo.eval_time = qinfo.caller_eval_cost
            else:
                qinfo.eval_time = 1.0
            qinfo.receive_time = qinfo.send_time + qinfo.eval_time
            qinfo.worker_id = 0
            self.last_receive_time = qinfo.receive_time

            self.latest_results.append(qinfo)

        return self.last_receive_time

    def all_workers_are_free(self):
        num_pending_tasks = self.scheduler.pending_tasks()
        for _ in range(num_pending_tasks):
            self.a_worker_is_free(force_await=True)
        return self.last_receive_time

    def _dispatch_experiment(self, func_caller, qinfo, **kwargs):
        if func_caller is self.default_func_caller:
            func_caller = self.caller_handle
        self.scheduler.submit(func_caller, qinfo, **kwargs)

    def dispatch_single_experiment(self, func_caller, qinfo, **kwargs):
        self._dispatch_experiment(func_caller, qinfo, **kwargs)

    def dispatch_batch_of_experiments(self, func_caller, qinfos, **kwargs):
        for qinfo in qinfos:
            self.dispatch_single_experiment(func_caller, qinfo, **kwargs)

    def get_time_distro_info(self):
        return 'caller_eval_cost'

    def get_poll_time_real(self):
        return 5.0
