from gaussian_density import GaussianDensity, State
import numpy as np
from utils import EPS
from poisson import PointPoissonProcess
from object_detection import ObjectDetection


state_dim = 4
meas_dim = 2
gating_size = 18.0427
prob_survival = 0.9
prob_detection = 0.9
poisson_birth_weight = 0
poisson_prune_threshold = 0
poisson_merge_threshold = -5

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

density = GaussianDensity(state_dim, meas_dim, F, Q, H, R, gating_size)

poisson = PointPoissonProcess(state_dim, meas_dim, prob_survival, prob_detection, density, poisson_birth_weight, poisson_prune_threshold, poisson_merge_threshold)

# get measurement
data = np.loadtxt('test_gating_measurements.txt')
measurements = [data[:, i].reshape(2, 1) for i in range(data.shape[1])]
z = measurements[0]
z = np.array([z[0, 0], z[1, 0], 0]).reshape(-1, 1)
detection = ObjectDetection(z, 'dummy', False)

poisson.predict([detection])
for component in poisson.intensity:
    print('weight:{}\tstate:{}'.format(component['w'], component['s']))

poisson.update()
for component in poisson.intensity:
    print('weight:{}\tstate:{}'.format(component['w'], component['s']))

print('Number of components before pruning {}'.format(len(poisson.intensity)))
poisson.intensity[-1]['w'] = 1
poisson.prune()
for component in poisson.intensity:
    print('weight:{}\tstate:{}'.format(component['w'], component['s']))
print('Number of components before pruning {}'.format(len(poisson.intensity)))
