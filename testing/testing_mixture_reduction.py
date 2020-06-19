import numpy as np
from gaussian_density import GaussianDensity, State
from utils import EPS

state_dim = 2
meas_dim = 2
F = np.eye(2)
Q = np.eye(2)
H = np.eye(2)
R = np.eye(2)
gating_size = 10

density = GaussianDensity(state_dim, meas_dim, F, Q, H, R, gating_size)

s1 = State(mean=np.array([0, 0]).reshape(2, 1), covariance=np.eye(2), empty_constructor=False)
s2 = State(mean=np.array([0.5, 0]).reshape(2, 1), covariance=np.eye(2), empty_constructor=False)
s3 = State(mean=np.array([0, 0.5]).reshape(2, 1), covariance=np.eye(2), empty_constructor=False)
s4 = State(mean=np.array([2, 2]).reshape(2, 1), covariance=np.eye(2), empty_constructor=False)

log_weights = np.log([2, 1, 1, 1]).tolist()

new_log_weights, new_states = density.mixture_reduction(log_weights, [s1, s2, s3, s4], 1)
assert len(new_log_weights) == 2, 'Wrong number of components \t Expected: 2 \t Computed: {}'.format(len(new_log_weights))
assert np.abs(new_log_weights[0] - np.log(4)) < EPS, 'Wrong log weights'
