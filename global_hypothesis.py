from typing import List, Tuple

class GlobalHypothesis(object):
    """
    A global hypothesis is a List of (Track ID, Single Target Hypo ID)
    """
    def __init__(self, log_w: float = 0):
        self.pairs_id : List[Tuple[int, int]] = []
        self.log_weight = log_w

    def get_num_obj(self) -> int:
        """
        Get number of object in this global hypothesis
        :return:
        """
        assert self.pairs_id is not None, 'GlobalHypothesis must be initialized with pairs of Track ID & STH ID before'
        return len(self.pairs_id)

    def __repr__(self):
        return '<GlobalHypo | log_w: {}, {}>\n'.format(round(self.log_weight, 3), self.pairs_id)
