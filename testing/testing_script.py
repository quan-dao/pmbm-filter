from gaussian_density import GaussianDensity, State
import numpy as np
from utils import EPS


state_dim = 4
meas_dim = 2
gating_size = 18.0427
F = np.array([
             [1,     0,     1,     0],
             [0,     1,     0,     1],
             [0,     0,     1,     0],
             [0,     0,     0,     1]
            ])
Q = np.array([
             [1,     0,     2,     0],
             [0,     1,     0,     2],
             [2,     0,     4,     0],
             [0,     2,     0,     4]
            ])
H = np.array([
             [1,     0,     0,     0],
             [0,     1,     0,     0],
            ])
R = np.array([
            [100,    0],
            [0,      100]
            ])

data = np.loadtxt('test_gating_measurements.txt')
measurements = [data[:, i].reshape(2, 1) for i in range(data.shape[1])]
# print('Example measurement:\n', measurements[0])
# gt_measurements_in_gate = [1, 42, 45, 53]  # matlab index
gt_measurements_in_gate = [0, 41, 44, 52]  # python index

density = GaussianDensity(state_dim, meas_dim, F, Q, H, R, gating_size)

state = State()
state.x = np.array([5, 5, 5, 5]).reshape(4, 1)
state.P = np.eye(4)

measurements_in_gate = density.ellipsoidal_gating(state, measurements)

assert measurements_in_gate == gt_measurements_in_gate, \
    'Test gating failed,\n meausrements_in_gate = {} \n gt_meausrements_in_gate = {}'.format(measurements_in_gate, gt_measurements_in_gate)

# Test moment matching
log_weights = [-3.6335,   -4.3945,   -1.4594,   -0.8595,   -1.1859]

data = np.loadtxt('test_moment_matching_x_P_5_total.txt')
x_mat = data[:, :5]
P_mat = data[:, 5:]
states = []
for i in range(5):
    s = State()
    s.x = x_mat[:, i].reshape(-1, 1)
    s.P = P_mat[:, 4*i : 4*i + 4]
    states.append(s)

gt_mean = np.array([1.1810,    3.0536,    1.2013,    0.0530])
gt_cov = np.array([
    [10.5504,    3.9671,    4.6697,   14.9861],
    [3.9671,    5.6466,    5.2883,   12.8995],
    [4.6697,    5.2883,   12.9538,   23.6850],
    [14.9861,   12.8995,   23.6850,   70.4611]
])

merged = density.moment_matching(states, log_weights)
assert (np.abs(merged.x.squeeze() - gt_mean) < 1e-3).all(), 'Wrong merged mean:\ncalculated: {}\nnground truth: {}'.format(merged.x.squeeze(), gt_mean)
assert (np.abs(merged.P - gt_cov) < 1e-3).all(), 'Wrong merged covariance:\ncalculated: {}\nnground truth: {}'.format(merged.P, gt_cov)
