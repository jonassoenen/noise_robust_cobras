import abc
import copy

from noise_robust_cobras.cobras_logger import ClusteringLogger
from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint


class MaximumQueriesExceeded(Exception):
    pass


class Querier:
    def __init__(self, logger: ClusteringLogger = None):
        self.logger = logger

    @abc.abstractmethod
    def _query_points(self, idx1, idx2):
        return

    @abc.abstractmethod
    def query_limit_reached(self):
        return

    def query(self, idx1, idx2, purpose=None):
        min_instance = min(idx1, idx2)
        max_instance = max(idx1, idx2)
        constraint_type = self._query_points(min_instance, max_instance)
        constraint = Constraint(
            min_instance, max_instance, constraint_type, purpose=purpose
        )
        if self.logger is not None:
            self.logger.log_new_user_query(copy.copy(constraint))
        return constraint

    def update_clustering(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        return

    def update_clustering_detailed(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        return
