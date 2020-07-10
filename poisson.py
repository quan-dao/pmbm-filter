import numpy as np
from typing import List, Dict

from gaussian_density import State, GaussianDensity
from object_detection import ObjectDetection
from target import Target
from utils import normalize_log_weights
from single_target_hypothesis import SingleTargetHypothesis


class PointPoissonProcess(object):
    """
    Represent undetected targets
    """
    def __init__(self,
                 state_dim: int,
                 measurement_dim: int,
                 prob_survival: float,
                 prob_detection: float,
                 density_hdl: GaussianDensity,
                 birth_weight: float,
                 birth_gating_size: float,
                 prune_threshold: float,
                 merge_threshold: float,
                 clutter_intensity: float,
                 current_time_step: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.prob_survival = prob_survival
        self.prob_detection = prob_detection
        self.density_hdl = density_hdl  # a handler to PMBM's GaussianDensity for KF functionality
        self.birth_weight = birth_weight  # suggestion log(5e-3)
        self.birth_gating_size = birth_gating_size  # to check a measurement is close to a component of PPP's intensity
        self.prune_threshold = prune_threshold  # in log domain
        self.merge_threshold = merge_threshold  # size of mahalanobis distance to consider 2 gaussian are close
        self.intensity = []  # Dict{w: log_w, s: State}
        self.target_id_to_give = 0  # birth only comes from PPP, this number defines each target id at its birth
        self.clutter_intensity = clutter_intensity
        self.current_time_step = current_time_step

    def get_new_target_id(self) -> int:
        new_target_id = self.target_id_to_give
        self.target_id_to_give += 1
        return new_target_id

    def give_birth(self, measurements: List[ObjectDetection], birth_per_meas=0) -> None:
        """
        Add new Gaussian to the mixture (which represent the poisson intensity). This birth process is driven by
        measurements. Each measurement induce 3 birth of the same class by adding noise (uniformly distributed )
        to its value
        :param measurements: [x, y, yaw]
        :param classes:
        :param birth_per_meas: number of birth induced by a measurement
        """
        trans_width = 10
        trans_cov = (2*trans_width)**2
        angle_cov = (2*np.pi)**2
        for meas in measurements:
            delta_x = np.random.uniform(-trans_width, trans_width, size=birth_per_meas).tolist() + [0]  # to ensure each measurement spawns a track
            delta_y = np.random.uniform(-trans_width, trans_width, size=birth_per_meas).tolist() + [0]
            delta_yaw = np.random.uniform(-np.pi, np.pi, size=birth_per_meas).tolist() + [0]
            for i in range(len(delta_x)):
                mean = np.array([
                    [meas.z[0, 0] + delta_x[i]],  # x
                    [meas.z[1, 0] + delta_y[i]],  # y
                    [meas.z[2, 0] + delta_yaw[i]],  # yaw
                    [0],  # dx
                    [0],  # dy
                    [0]  # dyaw
                ])
                # if meas.obj_type == 'bicycle':
                #     cov = np.diag([0.05390982, 0.05039431, 1.29464435, 0.04560422, 0.04097244, 1.21635902])
                # elif meas.obj_type == 'bus':
                #     cov = np.diag([0.17546469, 0.13818929, 0.1979503, 0.13263319, 0.11508148, 0.22529652])
                # elif meas.obj_type == 'car':
                #     cov = np.diag([0.08900372, 0.09412005, 1.00535696, 0.08120681, 0.08224643, 0.99492726])
                # elif meas.obj_type == 'motorcycle':
                #     cov = np.diag([0.04052819, 0.0398904, 1.06442726, 0.0437039 , 0.04327734, 1.30414345])
                # elif meas.obj_type == 'pedestrian':
                #     cov = np.diag([0.03855275, 0.0377111, 2.0751833, 0.04237008, 0.04092393, 2.0059979])
                # elif meas.obj_type == 'trailer':
                #     cov = np.diag([0.23228021, 0.22229261, 1.05163481, 0.2138643 , 0.19625241, 0.97082174])
                # else:
                #     cov = np.diag([0.14862173, 0.1444596, 0.73122169, 0.10683797, 0.10248689, 0.76188901])

                cov = np.diag([trans_cov, trans_cov, angle_cov, 10, 10, 2*np.pi])
                self.intensity.append({'w': self.birth_weight,
                                       's': State(mean, cov, meas.obj_type, empty_constructor=False)})

    def predict(self, measurements: List[ObjectDetection]) -> None:
        # predict for components already in the intensity
        for component in self.intensity:
            component['w'] += np.log(self.prob_survival)
            component['s'] = self.density_hdl.predict(component['s'])
        # create new components for poisson intensity by giving birth
        self.give_birth(measurements)

    def update(self) -> None:
        """
        Update for targets that were undetected at the previous time step and stay undetected at the current time step
        """
        for component in self.intensity:
            component['w'] += np.log(1 - self.prob_detection)

    def prune(self) -> None:
        """
        Remove components of the intensity whose weight smaller than prune_threshold
        """
        self.intensity = [component for component in self.intensity if component['w'] > self.prune_threshold]

    def merge(self) -> None:
        """
        Reduce the intensity by merging the "close" components  (close in the sense of Mahalanobis distance)
        """
        log_weights = [component['w'] for component in self.intensity]
        states = [component['s'] for component in self.intensity]
        reduced_log_weights, reduced_states = self.density_hdl.mixture_reduction(log_weights, states, self.merge_threshold)
        self.intensity = [{'w': log_w, 's': state} for log_w, state in zip(reduced_log_weights, reduced_states)]

    def create_new_targets(self, measurements: List[ObjectDetection]) -> Dict[int, Target]:
        """
        Perform update for targets detected for the first time
        :return: a list new tracks spawned by this list of measurements
        """
        gating_matrix = np.zeros((len(self.intensity), len(measurements)))  # gating_matrix[i, j] = 1 if meas j in gate of component i
        for i_com, component in enumerate(self.intensity):
            meas_in_gate = self.density_hdl.ellipsoidal_gating(component['s'], measurements, self.birth_gating_size)
            gating_matrix[i_com, meas_in_gate] = 1

        # for each measurement spawn a new track (i.e. target)
        new_targets = {}
        for j_meas, meas in enumerate(measurements):
            assert np.sum(gating_matrix[:, j_meas]) > 0, 'Measurement {} is not in gate of any poisson componenets'.format(j_meas)
            # get index of poisson components which has this measurement in its gate
            all_indices = np.arange(0, len(self.intensity))
            in_gate_of_components = all_indices[gating_matrix[:, j_meas] == 1]
            # compute w_i - unnormalized weight of components of new target mixture
            log_w_unnorm = [self.intensity[k]['w'] + self.density_hdl.log_likelihood(self.intensity[k]['s'], meas.z)
                            for k in in_gate_of_components]
            # compute e
            log_w, log_sum_w_unnorm = normalize_log_weights(log_w_unnorm)
            log_e = np.log(self.prob_detection) + log_sum_w_unnorm
            # compute rho
            rho = np.exp(log_e) + self.clutter_intensity
            # prob of existence of the STH in this new target
            prob_existence = 1.0 - self.clutter_intensity / rho
            # find mixuture representing the state of the STH in this new target
            mixture_states = [self.density_hdl.update(self.intensity[k]['s'], meas) for k in in_gate_of_components]
            # approximate this mixture by a single Gaussian
            merged = self.density_hdl.moment_matching(mixture_states, log_w, is_unnormalized=False)

            # create STH & the associated track
            first_STH = SingleTargetHypothesis(log_weight=np.log(rho),
                                               prob_existence=prob_existence,
                                               state=merged,
                                               assoc_meas_idx=j_meas,
                                               time_of_birth=self.current_time_step,
                                               cost=np.log(rho))
            target = Target(target_id=self.get_new_target_id(),
                            obj_type=meas.obj_type,
                            time_of_birth=self.current_time_step,
                            prob_survival=self.prob_survival,
                            prob_detection=self.prob_detection,
                            first_single_target_hypo=first_STH,
                            density_hdl=self.density_hdl)
            new_targets[j_meas] = target

        return new_targets
