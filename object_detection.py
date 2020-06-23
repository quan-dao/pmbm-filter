import numpy as np


class ObjectDetection(object):
    """
    Represents the output of an object detector
    """

    def __init__(self, z: np.ndarray = None, obj_type: str = None, empty_constructor=True):
        if empty_constructor:
            self.z = None  # value of object position, orientation, size
            self.obj_type = None  # class of object
        else:
            assert z.shape[1] == 1, 'z must be a column vector, z.shape = {}'.format(z.shape)
            self.z = z
            self.obj_type = obj_type

    def __repr__(self):
        return '<ObjectDetection | Type: {}, \tz:{}>'.format(self.obj_type, self.z.squeeze())
