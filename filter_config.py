import numpy as np


class FilterConfig(object):
    def __init__(self,
                 state_dim: int,
                 measurement_dim: int,
                 prob_survival: float = 0.9,
                 prob_detection: float = 0.9,
                 poisson_birth_weight: float = np.log(5e-3),
                 poisson_birth_gating_size: float = 11.0,
                 poisson_prune_threshold: float = -3,
                 poisson_merge_threshold: float = 2.0,
                 poisson_clutter_intensity: float = 1e-4,
                 pmbm_desired_num_global_hypotheses: int = 100,
                 pmbm_prune_single_hypothesis_existence: float = 1e-3,
                 pmbm_prune_global_hypothesis_log_weight: float = -5.0):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.prob_survival = prob_survival
        self.prob_detection = prob_detection
        self.poisson_birth_weight = poisson_birth_weight
        self.poisson_birth_gating_size = poisson_birth_gating_size
        self.poisson_prune_threshold = poisson_prune_threshold
        self.poisson_merge_threshold = poisson_merge_threshold
        self.poisson_clutter_intensity = poisson_clutter_intensity
        self.pmbm_desired_num_global_hypotheses = pmbm_desired_num_global_hypotheses
        self.pmbm_prune_single_hypothesis_existence = pmbm_prune_single_hypothesis_existence
        self.pmbm_prune_global_hypothesis_log_weight = pmbm_prune_global_hypothesis_log_weight
