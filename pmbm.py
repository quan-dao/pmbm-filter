import numpy as np
from typing import List, Dict

from murty import Murty
from global_hypothesis import GlobalHypothesis
from target import Target
from utils import INF, normalize_log_weights
from poisson import PointPoissonProcess
from filter_config import FilterConfig
from gaussian_density import GaussianDensity
from object_detection import ObjectDetection


class PoissonMultiBernoulliMixture(object):
    def __init__(self,
                 config: FilterConfig,
                 density_hdl: GaussianDensity,
                 current_time_step: int = 0):
        # collection of all targets since the begin of time to this time step includes new STH created after update
        # key or targets_pool is the target_id of the target in value
        self.targets_pool: Dict[int, Target] = {}
        # all possibly new targets that are detected for the 1st time at this time step
        # key of new_target_pool is the index of the measurement that gives rise to the target,
        # not target_id (as in targets_pool)
        self.new_targets_pool: Dict[int, Target] = {}
        self.global_hypotheses: List[GlobalHypothesis] = []
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
        self.estimation_result: Dict[int, Dict] = {}

    def __repr__(self):
        return '<PMBM | Num Targets: {},\t Num Poisson: {},\t Global Hypos:\n {}>'.format(
            len(self.targets_pool),
            len(self.poisson.intensity),
            self.global_hypotheses
        )

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
                cost_matrix[meas_idx, idx_target] = -child_single_target.cost
            # increment idx_target (to move to the next column of cost_matrix)
            idx_target += 1

        # create cost for newly detected target
        all_meas_idx = np.arange(0, num_of_measurements)
        for idx_target in range(num_of_targets):
            meas_in_gate = all_meas_idx[cost_matrix[:, idx_target] < INF]
            if len(meas_in_gate) > 1:  # this previously detected target has more than 1 measurements in its gate, at least one of them can create new target
                for meas_idx in meas_in_gate:
                    new_target = self.new_targets_pool[meas_idx]
                    cost_matrix[meas_idx, num_of_targets + meas_idx] = -new_target.single_target_hypotheses[0].cost
        for i_meas, new_target in self.new_targets_pool.items():
            if np.all(cost_matrix[i_meas, : num_of_targets] == INF):
                # This measurement is not in gate of any previously detected object
                cost_matrix[i_meas, num_of_targets + i_meas] = -new_target.single_target_hypotheses[0].cost  # new target only has 1 STH, there were a minus here

        return cost_matrix

    def create_new_global_hypotheses(self, num_of_measurement: int) -> List[GlobalHypothesis]:
        """
        Solve the data association at this time step by creating multiple new global hypotheses
        from the set of curernt global hypotheses
        :param num_of_measurement:
        :return:
        """
        assert len(self.global_hypotheses) > 0, 'global_hypotheses has not been initialized'
        # compute weight of all global hypotheses
        log_weights_unnorm = [global_hypo.log_weight for global_hypo in self.global_hypotheses]
        log_weights, _ = normalize_log_weights(log_weights_unnorm)

        new_global_hypothese = []
        for log_w, global_hypo in zip(log_weights, self.global_hypotheses):
            murty_k = int(np.ceil(self.desired_num_global_hypotheses * np.exp(log_w)))
            cost_matrix = self.create_cost_matrix(global_hypo, num_of_measurement)
            murty_solver = Murty(cost_matrix)
            num_of_targets = global_hypo.get_num_obj()  # number of old targets in this global hypothesis
            for iteration in range(murty_k):
                ok, cost, column_for_meas = murty_solver.draw()
                if not ok:
                    'Murty solver is not ok with cost matrix {}'.format(cost_matrix)
                    break
                # assert cost < INF, 'Optimal cost {} is too high'.format(cost)
                if cost > 0.5 * INF: break
                column_for_meas = column_for_meas.tolist()
                new_global_hypo = GlobalHypothesis()

                # add hypotheses of previously detected object or objects detected for the first time (i.e. objects that
                # associated with a measurement
                detected_targets = []
                for i_meas, j_colummn in enumerate(column_for_meas):
                    if j_colummn >= num_of_targets:
                        # the target this measurement is assigned to is a newly created target,
                        # not in the current global hypothesis
                        target_id = self.new_targets_pool[j_colummn - num_of_targets].target_id
                        sth_id = 0
                        # new_global_hypo.log_weight += self.new_targets_pool[j_colummn - num_of_targets].single_target_hypotheses[0].log_weight
                        new_global_hypo.new_targets_id.append(target_id)
                    else:
                        # the target this measurement is assigned to is a target previously detected,
                        target_id, parent_sth_id = global_hypo.pairs_id[j_colummn]
                        sth_id = self.targets_pool[target_id].single_target_hypotheses[parent_sth_id].children[i_meas].get_id()
                        # new_global_hypo.log_weight += self.targets_pool[target_id].single_target_hypotheses[parent_sth_id].children[i_meas].log_weight
                    new_global_hypo.pairs_id.append((target_id, sth_id))
                    new_global_hypo.log_weight = log_w - cost
                    detected_targets.append(target_id)

                # add hypotheses of objects that is undetected, if prob of existence of this hypo above a threshold
                # for target_id, parent_sth_id in global_hypo.pairs_id:
                #     if target_id in detected_targets: continue  # this target is detected by a measurement, move on
                #     child_single_target = self.targets_pool[target_id].single_target_hypotheses[parent_sth_id].children[-1]  # -1 indicates miss hypothesis
                #     if child_single_target.get_prob_existence() > self.prune_single_hypothesis_existence:
                #         new_global_hypo.pairs_id.append((target_id, child_single_target.get_id()))

                new_global_hypothese.append(new_global_hypo)

        return new_global_hypothese

    def prune_global_hypotheses(self):
        """
        Prune newly created global hypotheses whose weight smaller than a threshold
        :return:
        """
        # only invoked after new global hypotheses are created
        assert self.new_targets_pool == {}, 'new_target_pool is not cleaned up after creating new global hypotheses'

        # prunning
        to_prune_hypo = [i for i, global_hypo in enumerate(self.global_hypotheses) if global_hypo.log_weight < self.prune_global_hypothesis_log_weight]
        for i_prune in reversed(to_prune_hypo):
            del self.global_hypotheses[i_prune]

        # renormalize log weights
        log_weights_unnorm = [global_hypo.log_weight for global_hypo in self.global_hypotheses]
        log_weights, _ = normalize_log_weights(log_weights_unnorm)

        # Sort self.global_hypotheses in descending order of global_hypo.log_weight, then capping (if necessary)
        sorted_indicies = np.argsort(log_weights)  # ascending order
        sorted_indicies = sorted_indicies[::-1]  # flip sorted_indicies, to have descending order
        kept_indicies = sorted_indicies[: min(len(self.global_hypotheses), self.desired_num_global_hypotheses)]
        self.global_hypotheses = [self.global_hypotheses[i] for i in kept_indicies]

        # renormalize log weights
        log_weights_unnorm = [global_hypo.log_weight for global_hypo in self.global_hypotheses]
        log_weights, _ = normalize_log_weights(log_weights_unnorm)
        for log_w, global_hypo in zip(log_weights, self.global_hypotheses):
            global_hypo.log_weight = log_w

    def recycle_targets(self):
        """
        TODO: return bernoulli back to poisson
        :return:
        """
        pass

    def predict(self, measurements: List[ObjectDetection]):
        """
        Perform predict for both Poisson and Multi Bernoulli Mixture
        :return:
        """
        assert self.new_targets_pool == {}, 'new_target_pool is not cleaned up after creating new global hypotheses'
        self.poisson.predict(measurements)  # birth process is also done here
        for _, target in self.targets_pool.items():
            target.predict()

    def update(self, measurements: List[ObjectDetection]):
        """
        Perform update for both Poisson and Multi Bernoulli Mixture
        :return:
        """
        # update for previously undetected target and stay undetected now
        self.poisson.update()
        # update for objects detected for the 1st time
        self.new_targets_pool = self.poisson.create_new_targets(measurements)
        # update for objects detected in the previous time step
        for _, target in self.targets_pool.items():
            target.update(measurements)
        # create new global hypothesis
        if len(self.global_hypotheses) == 0:
            assert len(self.targets_pool) == 0, \
                'In initialization phase, targets_pool has to be empty (current len: {})'.format(len(self.targets_pool))
            # create first global hypothesis with newly detected objects
            first_global_hypo = GlobalHypothesis()
            for target_id, target in self.new_targets_pool.items():
                assert target_id == target.target_id, 'Weird, they are not the same'
                sth_id = target.single_target_hypotheses[0].single_id
                first_global_hypo.log_weight += target.single_target_hypotheses[0].log_weight
                first_global_hypo.pairs_id.append((target_id, sth_id))
                first_global_hypo.new_targets_id.append(target_id)
            self.global_hypotheses.append(first_global_hypo)
        else:
            self.global_hypotheses = self.create_new_global_hypotheses(len(measurements))

        # normalize global hypothesis weights
        log_weights_unnorm = [global_hypo.log_weight for global_hypo in self.global_hypotheses]
        log_weights, _ = normalize_log_weights(log_weights_unnorm)
        for log_w, global_hypo in zip(log_weights, self.global_hypotheses):
            global_hypo.log_weight = log_w

        # organize things in Poisson, Target, and PMBM to get ready for next time step
        self.prepare_for_next_time_step()

    def prepare_for_next_time_step(self):
        """
        Update miscellaneous thing of Poisson, Target, and PMBM to move on to next time step
        """
        # in Target, update single_target_hypotheses with all sth's children
        for _, target in self.targets_pool.items():
            new_single_target_hypotheses = {}
            for _, parent_sth in target.single_target_hypotheses.items():
                for _, child_sth in parent_sth.children.items():
                    new_single_target_hypotheses[child_sth.single_id] = child_sth
            target.single_target_hypotheses = new_single_target_hypotheses

        # merge self.new_targets_pool & self.targets_pool, as in next time step current new_targets_pool is not new
        for i_meas, new_target in self.new_targets_pool.items():
            assert new_target.target_id not in self.targets_pool.keys(), \
                'new_target ID ({}) is in old_targets_id'.format(new_target.target_id)
            self.targets_pool[new_target.target_id] = new_target
        # clean up new_targets_pool
        self.new_targets_pool: Dict[int, Target] = {}

    def increment_internal_timestep(self) -> None:
        """
        To increment internal timestep counter of PMBM, Poisson, Target
        """
        # increment current_time_step & update the same in Poisson & Target
        self.current_time_step += 1
        self.poisson.current_time_step = self.current_time_step
        for _, target in self.targets_pool.items():
            target.current_time_step = self.current_time_step

    def reduction(self):
        """
        Do all the pruning here
        """
        self.poisson.prune()

        # prune single target hypotheses whose prob of existence smaller than a threshold
        all_prune_pairs = []
        all_recycle_pairs = []
        for _, target in self.targets_pool.items():
            prune_pairs, recycle_pairs = target.prune_single_target_hypo()
            all_prune_pairs += prune_pairs
            all_recycle_pairs += recycle_pairs

        # recycle Bernoulli
        for target_id, sth_id in all_recycle_pairs:
            log_w = self.targets_pool[target_id].single_target_hypotheses[sth_id].log_weight
            state = self.targets_pool[target_id].single_target_hypotheses[sth_id].state
            self.poisson.intensity.append({'w': log_w, 's': state})
            del self.targets_pool[target_id].single_target_hypotheses[sth_id]

        # remove all global hypotheses where a prune pair appear
        prune_global_hypo = []
        for i, global_hypo in enumerate(self.global_hypotheses):
            for removed_pair in (all_prune_pairs + all_recycle_pairs):
                if removed_pair in global_hypo.pairs_id and i not in prune_global_hypo:
                    prune_global_hypo.append(i)
        for i in reversed(prune_global_hypo):
            del self.global_hypotheses[i]

        # prune global hypotheses whose weights is smaller than a threshold
        self.prune_global_hypotheses()

        # remove Bernoulli doesn't appear in any global hypotheses
        for _, target in self.targets_pool.items():
            unused_sth_id = []
            for single_id, single_hypo in target.single_target_hypotheses.items():
                assert single_id == single_hypo.single_id, 'Weird, they are not the same'
                pair = (target.target_id, single_id)
                need_to_prune = True
                for global_hypo in self.global_hypotheses:
                    if pair in global_hypo.pairs_id:
                        need_to_prune = False
                        break
                if need_to_prune: unused_sth_id.append(single_id)
            for sth_id in unused_sth_id:
                # recycle instead of remove
                log_w = target.single_target_hypotheses[sth_id].log_weight
                state = target.single_target_hypotheses[sth_id].state
                self.poisson.intensity.append({'w': log_w, 's': state})
                del target.single_target_hypotheses[sth_id]

        # remove targets which don't have any Bernoulli in its single_target_hypotheses
        empty_targets = [i_target for i_target, target in self.targets_pool.items()
                         if len(target.single_target_hypotheses.keys()) == 0]
        for i_target in empty_targets:  # don't need reversed(), cuz self.targets_pool is now a Dict
            del self.targets_pool[i_target]

        # set ID of the next Target to be born
        self.poisson.target_id_to_give = max(self.targets_pool.keys()) + 1

    def run(self, measurements: List[ObjectDetection]):
        """
        Main program of PMBM
        :return:
        """
        self.predict(measurements)
        self.update(measurements)
        self.reduction()
        self.estimate_targets()
        self.increment_internal_timestep()

    def estimate_targets(self):
        """
        TODO: estimate targets by choosing the global hypothese with highest weights
        :return:
        """
        chosen_global_hypo = self.global_hypotheses[0]
        print('Estimation | Num Targets: {}\n {} New Targets ID: {}'.format(
            len(chosen_global_hypo.pairs_id),
            chosen_global_hypo,
            chosen_global_hypo.new_targets_id
        ))

        estimation = {}
        for target_id, sth_id in chosen_global_hypo.pairs_id:
            state = self.targets_pool[target_id].single_target_hypotheses[sth_id].state
            estimation[target_id] = {
                'translation': [state.x[0, 0], state.x[1, 0]],
                'orientation': state.x[2, 0],
                'class': state.obj_type
            }
        self.estimation_result = {self.current_time_step: estimation}

        # clean up new_targets id in global hypotheses
        for global_hypo in self.global_hypotheses:
            global_hypo.new_targets_id = []
