from typing import List, Tuple


class GlobalHypothesis(object):
    """
    A global hypothesis is a List of (Track ID, Single Target Hypo ID)
    """
    def __init__(self, pairs_id: List[Tuple[int, int]] = None):
        self.pairs_id = pairs_id

    def get_num_obj(self) -> int:
        """
        Get number of object in this global hypothesis
        :return:
        """
        assert self.pairs_id is not None, 'GlobalHypothesis must be initialized with pairs of Track ID & STH ID before'
        return len(self.pairs_id)

    def __repr__(self):
        return '<GlobalHypo | {}>'.format(self.pairs_id)
