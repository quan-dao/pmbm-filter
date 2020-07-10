import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Dict

from utils import normalize_log_weights, put_in_range
from object_detection import ObjectDetection

class State(object):
    """
    Represent state of a random variable obeys Gaussian Distribution
    """

    def __init__(self, mean: np.ndarray = None, covariance: np.ndarray = None, obj_type: str = '', empty_constructor=True):
        if not empty_constructor:
            assert mean.shape[1] == 1, 'mean is not a vector: mean.shape = {}'.format(mean.shape)
            assert mean.shape[0] == covariance.shape[0], \
                'Input error, incompatible dimension: mean.shape ={}, covariance.shape = {}'.format(mean.shape, covariance.shape)
        self.x = mean
        self.P = covariance
        self.obj_type = obj_type

    def __repr__(self):
        return '<State Class \n obj_type {} \n x = {}>'.format(self.obj_type, self.x.squeeze())


class GaussianDensity(object):
    """
    Hold the functionality of Kalman Filter with linear dynamic & mesurement
    """
    def __init__(self, state_dim: int, meas_dim: int,
                 F: np.ndarray,
                 Q: Dict[str, np.ndarray],
                 H: np.ndarray,
                 R: Dict[str, np.ndarray]):
        NUSCENES_TRACKING_NAMES = [
            'bicycle',
            'bus',
            'car',
            'motorcycle',
            'pedestrian',
            'trailer',
            'truck'
        ]
        for k in Q.keys():
            assert k in NUSCENES_TRACKING_NAMES, 'object {} is not in NUSCENES_TRACKING_NAMES'.format(k)
        for k in R.keys():
            assert k in NUSCENES_TRACKING_NAMES, 'object {} is not in NUSCENES_TRACKING_NAMES'.format(k)
        assert F.shape[0] == state_dim, \
            'Incompatible dimension (expect equal): F.shape[0] = {}, state_dim = {}'.format(F.shape[0], state_dim)
        assert F.shape[0] == Q['car'].shape[0], \
            'Input error, F and Q have different dim: F.shape = {}, Q.shape = {}'.format(F.shape, Q['car'].shape)
        assert H.shape[0] == meas_dim, \
            'Incompatible dimension (expect equal): H.shape[0] = {}, meas_dim = {}'.format(H.shape[0], meas_dim)
        assert H.shape[1] == state_dim, \
            'Incompatible dimension (expect equal): H.shape[1] = {}, state_dim = {}'.format(H.shape[1], state_dim)
        assert H.shape[0] == R['car'].shape[0], \
            'Input error, H and R have different dim: H.shape = {}, R.shape = {}'.format(H.shape, R['car'].shape)
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def predict(self, state: State) -> State:
        # TODO: put predicted yaw in [-pi, pi]
        predicted = State()
        predicted.x = self.F @ state.x
        predicted.P = self.F @ state.P @ self.F.transpose() + self.Q[state.obj_type]
        predicted.obj_type = state.obj_type
        # put predicted yaw in [-pi, pi]
        predicted.x[2, 0] = put_in_range(predicted.x[2, 0])
        return predicted

    def update(self, state: State, measurement: ObjectDetection) -> State:
        """
        Perform update step of Kalman filter
        :param state:
        :param z: measurement vector - shape [meas_dim, 1]
        :return: updated state
        """
        z = measurement.z
        # TODO: put updated yaw in [-pi, pi]
        assert z.shape[1] == 1, 'measurement is not a column vector: z.shape = {}'.format(z.shape)
        psi = state.P @ self.H.transpose()
        S = self.H @ state.P @ self.H.transpose() + self.R[state.obj_type]  # innovation covariance
        S = (S + S.transpose()) / 2  # numerical stability
        K = psi @ np.linalg.inv(S)  # kalman gain
        updated = State()
        updated.x = state.x + K @ (z - self.H @ state.x)
        updated.P = state.P - K @ psi.transpose()
        updated.obj_type = measurement.obj_type
        # put updated yaw in [-pi, pi]
        updated.x[2, 0] = put_in_range(updated.x[2, 0])
        return updated

    def log_likelihood(self, state: State, z: np.ndarray) -> float:
        """
        Compute likelihood of a measurement in log domain
        :param state:
        :param z: measurement vector
        :return: likelihood
        """
        assert z.shape[1] == 1, 'measurement is not a column vector: z.shape = {}'.format(z.shape)
        S = self.H @ state.P @ self.H.transpose() + self.R[state.obj_type]  # innovation covariance
        S = (S + S.transpose()) / 2  # numerical stability
        mean = self.H @ state.x
        return multivariate_normal.logpdf(z.squeeze(), mean=mean.squeeze(), cov=S)

    def ellipsoidal_gating(self, state: State, measurements: List[ObjectDetection], gating_size: float) -> List[int]:
        """
        Perform gating to eliminate irrelevant (low likelihood) measurements
        :param state:
        :param measurements: List of measurement vectors
        :param gating_size:
        :return: List of index of measurements inside the gate of this this state
        """
        S = self.H @ state.P @ self.H.transpose() + self.R[state.obj_type]  # innovation covariance
        S = (S + S.transpose()) / 2.0  # numerical stability
        inv_S = np.linalg.inv(S)  # cache invert of S
        z_mean = self.H @ state.x

        measurements_in_gate = []
        for ix, meas in enumerate(measurements):
            if meas.obj_type != state.obj_type: continue  # skip measurements indicate different class
            innov = meas.z - z_mean  # innovation vector
            d = (innov.transpose() @ inv_S @ innov)
            if d < gating_size: measurements_in_gate.append(ix)

        return measurements_in_gate

    def moment_matching(self, states: List[State], log_weights: List[float], is_unnormalized=True) -> State:
        """
        Approximate a Gaussian Mixture parameterized by a list of states and a list of weights by 1 single Gaussian
        :param states:
        :param log_weights: in log domain
        :param is_unnormalized: True if log_weights are unnormalized
        :return:
        """
        # early stop if Lists of states contains 1 elements only
        if len(states) == 1:
            return states[0]

        if is_unnormalized:
            log_weights, _ = normalize_log_weights(log_weights)
        weights = np.exp(log_weights)
        # get object type & ensure every state to be merged have the same object type
        obj_type = states[0].obj_type
        if len(states) > 1:
            for state in states:
                assert state.obj_type == obj_type, \
                    'State to merge do not have the same object type: {}, expect {}'.format(state.obj_type, obj_type)
        merged = State(np.zeros((self.state_dim, 1)), np.zeros((self.state_dim, self.state_dim)), obj_type, empty_constructor=False)
        for w, state in zip(weights, states):
            merged.x += w * state.x
        for w, state in zip(weights, states):
            merged.P += w * (state.P + (merged.x - state.x) @ (merged.x - state.x).transpose())

        return merged

    def mixture_reduction(self, log_weights: List[float], states: List[State], threshold: float) -> (List[float], List[State]):
        """
        Perform greedy merging to reduce the number of Gaussian components for a Gaussian mixture density
        :param log_weights: unnormalized weights in log domain
        :param states:
        :return: a new mixture with less components
        """
        if len(log_weights) == 1:
            return log_weights, states

        new_log_weights = []
        new_states = []

        while states:
            # find the component with the highest weight
            idx_max_weight = np.argmax(log_weights).item()
            inv_P = np.linalg.inv(states[idx_max_weight].P)
            idx_to_merge, log_weights_to_merge, states_to_merge = [], [], []
            for i, state in enumerate(states):
                # find Mahalanobis distance to state with max weight
                diff = state.x - states[idx_max_weight].x
                d = diff.transpose() @ inv_P @ diff
                if d < threshold:
                    idx_to_merge.append(i)
                    log_weights_to_merge.append(log_weights[i])
                    states_to_merge.append(state)

            # perform moment matching for states that close to state with max weights
            norm_log_w, log_sum_w = normalize_log_weights(log_weights_to_merge)
            merged_state = self.moment_matching(states_to_merge, norm_log_w, is_unnormalized=False)
            new_log_weights.append(log_sum_w)
            new_states.append(merged_state)

            # remove merged states from original states
            for i in reversed(idx_to_merge):
                del states[i]
                del log_weights[i]

        return new_log_weights, new_states

