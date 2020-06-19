import numpy as np
from typing import List

from gaussian_density import State, GaussianDensity
from object_detection import ObjectDetection
from target import Target


class PointPoissonProcess(object):
    """
    Represent undetected targets
    """
    def __init__(self, state_dim: float, measurement_dim: float, prob_survival: float, prob_detection: float, density_hdl: GaussianDensity,
                 birth_weight: float, prune_threshold: float, merge_threshold: float):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.prob_survival = prob_survival
        self.prob_detection = prob_detection
        self.density_hdl = density_hdl  # a handler to PMBM's GaussianDensity for KF functionality
        self.birth_weight = birth_weight  # suggestion log(5e-3)
        self.prune_threshold = prune_threshold  # in log domain
        self.merge_threshold = merge_threshold  # size of mahalanobis distance to consider 2 gaussian are close
        self.intensity = []  # Dict{w: log_w, s: State}
        self.target_id_to_give = 0  # birth only comes from PPP, this number defines each target id at its birth

    def give_birth(self, measurements: List[ObjectDetection], birth_per_meas=3) -> None:
        """
        Add new Gaussian to the mixture (which represent the poisson intensity). This birth process is driven by
        measurements. Each measurement induce 3 birth of the same class by adding noise (uniformly distributed )
        to its value
        :param measurements: [x, y, yaw]
        :param classes:
        :param birth_per_meas: number of birth induced by a measurement
        """
        for meas in measurements:
            delta_x = np.random.uniform(-5, 5, size=birth_per_meas).tolist() + [0]  # to ensure each measurement spawns a track
            delta_y = np.random.uniform(-5, 5, size=birth_per_meas).tolist() + [0]
            delta_yaw = np.random.uniform(-np.pi, np.pi, size=birth_per_meas).tolist() + [0]
            for i in range(birth_per_meas):
                mean = np.array([
                    [meas.z[0, 0] + delta_x[i]],  # x
                    [meas.z[1, 0] + delta_y[i]],  # y
                    [meas.z[2, 0] + delta_yaw[i]],  # yaw
                    [0],  # dx
                    [0],  # dy
                    [0]  # dyaw
                ])
                cov = np.diag([10, 10, 2*np.pi, 10, 10, 2*np.pi])
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

    def create_new_targets(self, measurements: List[ObjectDetection]) -> List[Target]:
        """
        TODO: Perform update for targets detected for the first time
        :return: a list new tracks spawned by this list of measurements
        """

