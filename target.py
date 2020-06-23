import numpy as np
from typing import List

from object_detection import ObjectDetection
from gaussian_density import GaussianDensity, State
from single_target_hypothesis import SingleTargetHypothesis


class Target(object):
    """
    Represent a track (or a hypothesis tree) that is a sequence. Each element of a sequence is a set of all possible
    data association to an object
    """
    def __init__(self,
                 target_id: int,
                 obj_type: str,
                 time_of_birth: int,
                 prob_survival: float,
                 prob_detection: float,
                 first_single_target_hypo: SingleTargetHypothesis,
                 density_hdl: GaussianDensity,
                 gating_size: float = 5.0,
                 prune_prob_existence: float = 1e-3):
        self.target_id = target_id  # a unique number to distinguish this target with others
        self.obj_type = obj_type
        self.time_of_birth = time_of_birth  # time step when this target is born
        self.prob_survival = prob_survival
        self.prob_detection = prob_detection
        self.density_hdl = density_hdl  # handler to PMBM's Gaussian density
        self.current_time_step = time_of_birth
        self.gating_size = gating_size
        self.prune_prob_existence = prune_prob_existence
        self.single_id_to_give = 0
        # set ID for 1st single target hypo & store it
        first_single_target_hypo.single_id = 0
        self.single_target_hypotheses = {first_single_target_hypo.single_id: first_single_target_hypo}  # store all STHs at this current time step (not the whole tree)

    def __repr__(self):
        return '<Target | TID: {}, \tclass: {} \nSingle Target Hypothese:\n{}>\n'.format(
            self.target_id,
            self.obj_type,
            self.single_target_hypotheses
        )

    def get_new_STH_id(self) -> int:
        new_STH_id = self.single_id_to_give
        self.single_id_to_give += 1
        return new_STH_id

    def reset_single_id_to_give(self) -> None:
        """
        Reset sing_id_to_give as the Target finished update for this time step, so that at the next time step, STH ID
        starts from 0 again
        """
        self.single_id_to_give = 0

    def predict(self):
        """
        Perform KF prediction for all single target hypotheses in this track
        Reference: Algorithm 2.2 (chalmer thesis)
        """
        for _, single_target in self.single_target_hypotheses.items():
            single_target.state = self.density_hdl.predict(single_target.state)
            single_target.prob_existence *= self.prob_survival

    def update(self, measurements: List[ObjectDetection]):
        """
        Perform update for PREVIOUSLY DETECTED target (a.k.a all single target hypotheses in this track)
        Reference: Algorithm 2.3 (chalmer thesis)
        :param measurements: list of all measurements (not gated yet)
        """
        for _, single_target in self.single_target_hypotheses.items():
            # "update" populates STH's children by associating a STH from predict step to new measurements
            assert single_target.children == {}, 'STH children is not cleaned up'
            # create Misdetection hypothesis
            mis_state = single_target.state
            mis_log_w = single_target.log_weight + \
                        np.log(1 - single_target.prob_existence + single_target.prob_existence * (1 - self.prob_detection))
            mis_prob_exis = single_target.prob_existence * (1 - self.prob_detection) / \
                            (1 - single_target.prob_existence + single_target.prob_existence * (1 - self.prob_detection))
            mis_cost = mis_log_w
            mis_hypo = SingleTargetHypothesis(log_weight=mis_log_w,
                                              prob_existence=mis_prob_exis,
                                              state=mis_state,
                                              assoc_meas_idx=-1,
                                              single_target_hypo_id=self.get_new_STH_id(),
                                              time_of_birth=self.current_time_step,
                                              cost=mis_cost)
            single_target.children[-1] = mis_hypo

            # create Detection hypotheses
            meas_in_gate = self.density_hdl.ellipsoidal_gating(single_target.state, measurements, self.gating_size)
            for j_meas in meas_in_gate:
                updated_state = self.density_hdl.update(single_target.state, measurements[j_meas])
                log_w = single_target.log_weight + np.log(single_target.prob_existence * self.prob_detection) + \
                        self.density_hdl.log_likelihood(single_target.state, measurements[j_meas].z)
                detect_cost = log_w - mis_cost
                detect_hypo = SingleTargetHypothesis(log_weight=log_w,
                                                     prob_existence=1,
                                                     state=updated_state,
                                                     assoc_meas_idx=j_meas,
                                                     single_target_hypo_id=self.get_new_STH_id(),
                                                     time_of_birth=self.current_time_step,
                                                     cost=detect_cost)
                single_target.children[j_meas] = detect_hypo

        # reset single_id_to_give so that next time step single_target_id starts from 0
        self.reset_single_id_to_give()

    def prune_single_target_hypo(self) -> None:
        """
        Prune STH whose prob_existence below a threshold
        """
        sth_to_prune = [sth_id for sth_id, single_target in self.single_target_hypotheses.items()
                        if single_target.prob_existence < self.prune_prob_existence]
        for sth_id in sth_to_prune:
            del self.single_target_hypotheses[sth_id]
