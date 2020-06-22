import numpy as np
from typing import List

from murty import Murty
from global_hypothesis import GlobalHypothesis
from target import Target
from utils import INF, normalize_log_weights
from poisson import PointPoissonProcess
from filter_config import FilterConfig
from gaussian_density import GaussianDensity


# TODO: find a place to merge new_targets_pool with targets_pool
class PoissonMultiBernoulliMixture(object):
    def __init__(self,
                 config: FilterConfig,
                 density_hdl: GaussianDensity,
                 current_time_step: int = 0,
                 targets_pool: List[Target] = None,
                 new_targets_pool: List[Target] = None,
                 global_hypotheses: List[GlobalHypothesis] = None):
        self.targets_pool = targets_pool  # collection of all targets since the begin of time to this time step includes new STH created after update
        self.new_targets_pool = new_targets_pool  # all possibly new targets that are detected for the 1st time at this time step
        self.global_hypotheses = global_hypotheses
        self.desired_num_global_hypotheses = config.pmbm_desired_num_global_hypotheses
        self.prune_single_hypothesis_existence = config.pmbm_prune_single_hypothesis_existence
        self.prune_global_hypothesis_log_weight = config.pmbm_prune_global_hypothesis_log_weight
        self.current_time_step = current_time_step
        self.poisson = PointPoissonProcess(config.state_dim,
                                           config.measurement_dim,
                                           config.prob_survival,
                                           config.prob_detection,
                                           density_hdl,
                                           config.poisson_birth_weight,
                                           config.poisson_birth_gating_size,
                                           config.poisson_prune_threshold,
                                           config.poisson_merge_threshold,
                                           config.poisson_clutter_intensity,
                                           current_time_step)

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
        cost_matrix = np.zeros((num_of_measurements, num_of_targets + num_of_measurements)) + INF
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

    def compute_global_hypo_weight(self, global_hypo: GlobalHypothesis) -> float:
        """
        Compute log weight of a global hypothesis
        :param global_hypo:
        :return:
        """
        log_w = 0
        for target_id, sth_id in global_hypo.pairs_id:
            log_w += self.targets_pool[target_id].single_target_hypotheses[sth_id].log_weight
        return log_w

    def create_new_global_hypotheses(self, num_of_measurement: int) -> List[GlobalHypothesis]:
        """
        Solve the data association at this time step by creating multiple new global hypotheses
        from the set of curernt global hypotheses
        :param num_of_measurement:
        :return:
        """
        # compute weight of all global hypotheses
        log_weights_unnorm = [self.compute_global_hypo_weight(global_hypo) for global_hypo in self.global_hypotheses]
        log_weights, _ = normalize_log_weights(log_weights_unnorm)

        new_global_hypothese = []
        for log_w, global_hypo in zip(log_weights, self.global_hypotheses):
            murty_k = int(np.ceil(self.desired_num_global_hypotheses * np.exp(log_w)))
            cost_matrix = self.create_cost_matrix(global_hypo, num_of_measurement)
            murty_solver = Murty(cost_matrix)
            for iteration in range(murty_k):
                ok, cost, column_for_meas = murty_solver.draw()
                assert ok, 'Murty solver is not ok with cost matrix {}'.format(cost_matrix)
                assert cost < INF, 'Optimal cost {} is too high'.format(cost)
                column_for_meas = column_for_meas.tolist()
                new_global_hypo = GlobalHypothesis()

                # add hypotheses of previously detected object or objects detected for the first time (i.e. objects that
                # associated with a measurement
                detected_targets = []
                for i_meas, j_colummn in enumerate(column_for_meas):
                    target_id, parent_sth_id = global_hypo.pairs_id[j_colummn]
                    detected_targets.append(target_id)
                    sth_id = self.targets_pool[target_id].single_target_hypotheses[parent_sth_id].children[i_meas].get_id()
                    new_global_hypo.pairs_id.append((target_id, sth_id))

                # add hypotheses of objects that is undetected, if prob of existence of this hypo above a threshold
                for target_id, parent_sth_id in global_hypo.pairs_id:
                    if target_id in detected_targets: continue  # this target is detected by a measurement, move on
                    child_single_target = self.targets_pool[target_id].single_target_hypotheses[parent_sth_id].children[-1]  # -1 indicates miss hypothesis
                    if child_single_target.get_prob_existence() > self.prune_single_hypothesis_existence:
                        new_global_hypo.pairs_id.append((target_id, child_single_target.get_id()))

                new_global_hypothese.append(new_global_hypo)

        return new_global_hypothese

    def prune_global_hypotheses(self):
        """
        Prune newly created global hypotheses whose weight samller than a threshold
        :return:
        """
        # only invoked after new global hypotheses are created
        assert self.new_targets_pool == [], 'new_target_pool is not cleaned up after creating new global hypotheses'  # TODO: clean new_target_pool after createing new global hypotheses

        # compute weight of all global hypotheses
        log_weights_unnorm = [self.compute_global_hypo_weight(global_hypo) for global_hypo in self.global_hypotheses]
        log_weights, _ = normalize_log_weights(log_weights_unnorm)

        to_prune_hypo = [i for i, log_w in enumerate(log_weights) if log_w < self.prune_global_hypothesis_log_weight]
        for i_prune in reversed(to_prune_hypo):
            del self.global_hypotheses[i_prune]

    def recycle_targets(self):
        """
        TODO: return bernoulli back to poisson
        :return:
        """
        pass

    def predict(self):
        """
        TODO: perform predict for both Poisson and Multi Bernoulli Mixture
        :return:
        """
        pass

    def update(self):
        """
        TODO: perform update for both Poisson and Multi Bernoulli Mixture
        :return:
        """
        pass

    def estimate_targets(self):
        """
        TODO: estimate targets by choosing the global hypothese with highest weights
        :return:
        """
        pass

