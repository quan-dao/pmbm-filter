import numpy as np
from typing import List

from murty import Murty
from global_hypothesis import GlobalHypothesis
from target import Target
from utils import INF

# TODO: find a place to merge new_targets_pool with targets_pool


class PoissonMultiBernoulliMixture(object):
    def __init__(self,
                 targets_pool: List[Target] = None,
                 new_targets_pool: List[Target] = None,
                 global_hypotheses: List[GlobalHypothesis] = None):
        self.targets_pool = targets_pool  # collection of all targets since the begin of time to this time step includes new STH created after update
        self.new_targets_pool = new_targets_pool  # all possibly new targets that are detected for the 1st time at this time step
        self.global_hypotheses = global_hypotheses

    def create_cost_matrix(self, global_hypo: GlobalHypothesis, num_of_measurements: int) -> np.ndarray:
        """
        Create cost matrix for a global hypothesis (at the previous time step, i.e. before update)
        :param global_hypo: list of Tuple(target_id, sth_id)
        :param num_of_measurements: number of measurements of this time step
        :return:
        """
        assert len(self.new_targets_pool) == num_of_measurements, \
            'Number of new targets ({}) != number of measurements ({})'.format(len(self.new_targets_pool), num_of_measurements)
        num_of_targets = global_hypo.get_num_obj()
        cost_matrix = np.zeros((num_of_measurements, num_of_targets)) + INF
        # create cost for targets detected in the previous time step
        idx_target = 0  # index of columns of cost_matrix associated with a target in global_hypo
        for target_id, parent_sth_id in global_hypo.pairs_id:
            for meas_idx, child_single_target in self.targets_pool[target_id].single_target_hypotheses[parent_sth_id].children.items():
                # here a child target is a STH resulted from
                # associating parent STH (defined in the track named target_id) with measurement identified by meas_idx
                if meas_idx == -1: continue  # misdetection, let's move on
                cost_matrix[meas_idx, idx_target] = -child_single_target.cost  # TODO check this minus sign
            # increment idx_target (to move to the next column of cost_matrix)
            idx_target += 1

        # create cost for newly detected target
        for i_meas, new_target in enumerate(self.new_targets_pool):
            cost_matrix[i_meas, num_of_targets + i_meas] = new_target.single_target_hypotheses[0].cost  # new target only has 1 STH

        return cost_matrix

    def compute_global_hypo_weight(self, global_hypo: GlobalHypothesis):
        pass


