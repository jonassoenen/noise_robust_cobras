from noise_robust_cobras.querier.querier import Querier, MaximumQueriesExceeded


class LabelQuerier(Querier):
    """
        A querier which answers each query correctly
    """

    def __init__(self, logger, labels, maximum_number_of_queries):
        super(LabelQuerier, self).__init__(logger)
        self.labels = labels
        self.max_queries = maximum_number_of_queries
        self.queries_asked = 0

    def _query_points(self, idx1, idx2):
        if self.max_queries is not None and self.queries_asked >= self.max_queries:
            raise MaximumQueriesExceeded
        self.queries_asked += 1
        return self.labels[idx1] == self.labels[idx2]

    def query_limit_reached(self):
        if self.max_queries is None:
            return False
        return self.queries_asked >= self.max_queries
