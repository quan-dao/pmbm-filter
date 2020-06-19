from typing import List

from object_detection import ObjectDetection
from gaussian_density import GaussianDensity


class Target(object):
    """
    Represent a track (or a hypothesis tree) that is a sequence. Each element of a sequence is a set of all possible
    data association to an object
    """
    def __init__(self, target_id: int, obj_type: str, time_of_birth: int, density_hdl: GaussianDensity):
        self.target_id = target_id  # a unique number to distinguish this target with others
        self.obj_type = obj_type
        self.time_of_birth = time_of_birth  # time step when this target is born
        self.single_target_hypotheses = []  # store all STHs at this current time step (not the whole tree)
        self.density_hdl = density_hdl  # handler to PMBM's Gaussian density

    def __repr__(self):
        #TODO: finish this with information of each STH
        return 'Target {} - class {}'.format(self.target_id, self.obj_type)

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
