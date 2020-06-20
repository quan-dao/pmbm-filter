from typing import List

from object_detection import ObjectDetection
from gaussian_density import GaussianDensity
from single_target_hypothesis import SingleTargetHypothesis


class Target(object):
    """
    Represent a track (or a hypothesis tree) that is a sequence. Each element of a sequence is a set of all possible
    data association to an object
    """
    def __init__(self, target_id: int, obj_type: str, time_of_birth: int,
                 first_single_target_hypo: SingleTargetHypothesis,
                 density_hdl: GaussianDensity):
        self.target_id = target_id  # a unique number to distinguish this target with others
        self.obj_type = obj_type
        self.time_of_birth = time_of_birth  # time step when this target is born
        self.density_hdl = density_hdl  # handler to PMBM's Gaussian density
        self.single_id_to_give = 0
        # set ID for 1st single target hypo & store it
        first_single_target_hypo.single_id = self.get_new_STH_id()
        self.single_target_hypotheses = [first_single_target_hypo]  # store all STHs at this current time step (not the whole tree)

    def __repr__(self):
        #TODO: finish this with information of each STH
        return 'Target ID {} - class {}'.format(self.target_id, self.obj_type)

    def get_new_STH_id(self) -> int:
        new_STH_id = self.single_id_to_give
        self.single_id_to_give += 1
        return new_STH_id

    def predict(self):
        """
        TODO: Perform KF prediction for all single target hypotheses in this track
        Reference: Algorithm 2.2 (chalmer thesis)
        """
        pass

    def update(self, measurements: List[ObjectDetection]):
        """
        TODO: Peform update for previously detected target (a.k.a all single target hypotheses in this track)
        Reference: Algorithm 2.3 (chalmer thesis)
        :param measurements: list of all measurements (not gated yet)
        """
        pass
