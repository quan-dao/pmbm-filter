import numpy as np
from scipy.stats import multivariate_normal
from typing import List

from utils import normalize_log_weights


class State(object):
    """
    Represent state of a random variable obeys Gaussian Distribution
    """
    def __init__(self, mean=None, covariance=None, empty_constructor=True):
        if not empty_constructor:
            assert mean.shape[1] == 1, 'mean is not a vector: mean.shape = {}'.format(mean.shape)
            assert mean.shape[0] == covariance.shape[0], \
                'Input error, incompatible dimension: mean.shape ={}, covariance.shape = {}'.format(mean.shape, covariance.shape)
            self.x = mean
            self.P = covariance
        else:
            # no argument constructor
            self.x = None
            self.P = None

    def __repr__(self):
        return 'state: x = {}\n P = {}'.format(self.x, self.P)


class GaussianDensity(object):
    """
    Hold the functionality of Kalman Filter with linear dynamic & mesurement
    """
    def __init__(self, state_dim: int, meas_dim: int, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, gating_size: float):
        assert F.shape[0] == state_dim, \
            'Incompatible dimension (expect equal): F.shape[0] = {}, state_dim = {}'.format(F.shape[0], state_dim)
        assert F.shape[0] == Q.shape[0], \
            'Input error, F and Q have different dim: F.shape = {}, Q.shape = {}'.format(F.shape, Q.shape)
        assert H.shape[0] == meas_dim, \
            'Incompatible dimension (expect equal): H.shape[0] = {}, meas_dim = {}'.format(H.shape[0], meas_dim)
        assert H.shape[1] == state_dim, \
            'Incompatible dimension (expect equal): H.shape[1] = {}, state_dim = {}'.format(H.shape[1], state_dim)
        assert H.shape[0] == R.shape[0], \
            'Input error, H and R have different dim: H.shape = {}, R.shape = {}'.format(H.shape, R.shape)
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.gating_size = gating_size

    def predict(self, state:State) -> State:
        predicted = State()
        predicted.x = self.F @ state.x
        predicted.P = self.F @ state.P @ self.F.transpose()
        return predicted

    def update(self, state: State, z: np.ndarray) -> State:
        """
        Perform update step of Kalman filter
        :param state:
        :param z: measurement vector - shape [meas_dim, 1]
        :return: updated state
        """
        assert z.shape[1] == 1, 'measurement is not a column vector: z.shape = {}'.format(z.shape)
        psi = state.P @ self.H.transpose()
        S = self.H @ state.P @ self.H.transpose() + self.R  # innovation covariance
        S = (S + S.transpose()) / 2  # numerical stability
        K = psi @ np.linalg.inv(S)  # kalman gain
        updated = State()
        updated.x = state.x + K @ (z - self.H @ state.x)
        updated.P = state.P - K @ psi.transpose()
        return updated

    def log_likelihood(self, state: State, z: np.ndarray) -> float:
        """
        Compute likelihood of a measurement in log domain
        :param state:
        :param z: measurement vector
        :return: likelihood
        """
        assert z.shape[1] == 1, 'measurement is not a column vector: z.shape = {}'.format(z.shape)
        S = self.H @ state.P @ self.H.transpose() + self.R  # innovation covariance
        S = (S + S.transpose()) / 2  # numerical stability
        mean = self.H @ state.x
        return multivariate_normal.logpdf(z.squeeze(), mean=mean.squeeze(), cov=S)

    def ellipsoidal_gating(self, state:State, measurements: List[np.ndarray]) -> List[int]:
        """
        Perform gating to eliminate irrelevant (low likelihood) measurements
        :param state:
        :param measurements: List of measurement vectors
        :return: List of index of measurements inside the gate of this this state
        """
        S = self.H @ state.P @ self.H.transpose() + self.R  # innovation covariance
        S = (S + S.transpose()) / 2  # numerical stability
        inv_S = np.linalg.inv(S)  # cache invert of S
        z_mean = self.H @ state.x

        measurements_in_gate = []
        for ix, z in enumerate(measurements):
            innov = z - z_mean  # innovation vector
            d = innov.transpose() @ inv_S @ innov
            if d < self.gating_size: measurements_in_gate.append(ix)

        return measurements_in_gate

    def moment_matching(self, states: List[State], log_weights: List[float], is_unnormalized=True) -> State:
        """
        Approximate a Gaussian Mixture parameterized by a list of states and a list of weights by 1 single Gaussian
        :param states:
        :param weights: in log domain
        :param is_unnormalized: True if log_weights are unnormalized
        :return:
        """
        if is_unnormalized:
            log_weights, _ = normalize_log_weights(log_weights)
        weights = np.exp(log_weights)

        merged = State(mean=np.zeros((self.state_dim, 1)), covariance=np.zeros((self.state_dim, self.state_dim)), empty_constructor=False)
        for w, state in zip(weights, states):
            merged.x += w * state.x
        for w, state in zip(weights, states):
            merged.P += w * (state.P + (merged.x - state.x) @ (merged.x - state.x).transpose())

        return merged


