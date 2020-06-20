import numpy as np
from typing import List

# Definition of some constance
EPS = 1e-4
INF = 1e4


def normalize_log_weights(log_weights: List[float]) -> (List[float], float):
    """
    Normalize a list of log weights (so that sum(exp(log_weights)) = 1)
    :param log_weights:
    :return: normalized_log_weights and log_sum_unnormalized (log of sum of all unnormalized weights)
    """
    if len(log_weights) == 1:
        # compute sum of log_weights (all weights are unnormalized at this moment)
        log_sum_unnormalized = log_weights[0]
        # normalize
        log_weights[0] -= log_sum_unnormalized
        return [log_weights], log_sum_unnormalized

    log_weights = np.array(log_weights)
    arg_order = np.argsort(log_weights)  # ascending
    log_sum = log_weights[arg_order[-1]] + np.log(1 + np.sum(np.exp(log_weights[arg_order[:-1]] - log_weights[arg_order[-1]])))
    log_weights -= log_sum
    return log_weights.tolist(), log_sum


def put_in_range(angle):
    """
    Put an angle in range of [-pi, pi]
    """
    while angle <= -np.pi: angle += 2*np.pi
    while angle > np.pi: angle -= 2*np.pi
    return angle


if __name__ == '__main__':
    # test normalize_log_weights
    log_w = [-5.3551,   12.7862,    5.3548,    8.6605,    5.1929]
    gt_log_sum = 12.8033
    gt_log_w_normalized = [-18.1584,   -0.0171,   -7.4485,   -4.1428,   -7.6104]
    log_w, log_sum = normalize_log_weights(log_w)
    assert np.abs(log_sum - gt_log_sum) < EPS, "Wrong log_sum. calculated = {}\tground truth = {}".format(log_w, gt_log_sum)
    for w, gt_w in zip(log_w, gt_log_w_normalized):
        assert np.abs(w - gt_w) < EPS, "Wrong log_weight. calculated = {}\tground truth = {}".format(w, gt_w)


