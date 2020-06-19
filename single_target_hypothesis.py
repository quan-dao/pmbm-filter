from gaussian_density import State


class SingleTargetDensity(object):
    """
    Represents a possible data association of Track (aka a target) as a Bernoulli RFS
    """
    def __init__(self, log_weight: float = None, prob_existence: float = None, state: State = None, assoc_meas_idx: int = None):
        self.log_weight = log_weight
        self.prob_existence = prob_existence
        self.state = state
        self.assoc_meas_idx = assoc_meas_idx  # index of measurement associated with this STH (-1: miss, >= 0: detected)
