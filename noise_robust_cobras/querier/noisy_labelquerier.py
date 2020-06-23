from noise_robust_cobras.querier.querier import MaximumQueriesExceeded, Querier
import random
import numpy as np


class ProbabilisticNoisyQuerier(Querier):
    """
        A querier where each query has a fixed probability of being wrong.
        Asking the same query twice might result in different answers
    """

    def __init__(
        self,
        logger,
        labels,
        noise_percentage,
        maximum_number_of_queries,
        random_seed=None,
    ):
        super().__init__(logger)
        if random_seed is not None:
            self.rand = np.random.default_rng(random_seed)
        else:
            self.rand = np.random.default_rng()
        self.labels = labels
        self.noise_percentage = noise_percentage
        self.max_queries = maximum_number_of_queries
        self.queries_asked = 0

    def query_limit_reached(self):
        return self.queries_asked >= self.max_queries

    def _query_points(self, i, j):
        if self.queries_asked >= self.max_queries:
            raise MaximumQueriesExceeded
        correct_answer = self.labels[i] == self.labels[j]
        if self.rand.random() < self.noise_percentage:
            answer = not correct_answer
        else:
            answer = correct_answer
        self.queries_asked += 1
        return answer
