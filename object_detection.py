import numpy as np
from typing import List


class ObjectDetection(object):
    """
    Represents the output of an object detector
    """

    def __init__(self, z: np.ndarray = None, obj_type: str = None, size: List = None, height: float = None,
                 score: float = None,
                 empty_constructor=True):
        if not empty_constructor:
            assert z.shape[1] == 1, 'z must be a column vector, z.shape = {}'.format(z.shape)
        self.z = z
        self.obj_type = obj_type
        self.size = size
        self.height = height
        self.score = score

    def __repr__(self):
        return '<ObjectDetection | Type: {}, \tz:{}>'.format(self.obj_type, self.z.squeeze())
