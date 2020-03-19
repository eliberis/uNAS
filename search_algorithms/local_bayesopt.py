from search_algorithms import BayesOpt


class LocalBayesOpt(BayesOpt):
    @property
    def acq_opt_method(self):
        return "local"
