from noise_robust_cobras.querier.querier import Querier


class DummyQuerier(Querier):
    def __init__(self, cluster_list, predefined_answers=()):
        super().__init__()
        self.cluster_list = cluster_list
        self.predefined_answers = list(predefined_answers)
        self.query_log = []

    def get_predefined_answer(self, idx1, idx2):
        predef_answer = None
        for i1, i2, answer in self.predefined_answers:
            if i1 == idx1 and i2 == idx2:
                predef_answer = answer
                break
        if predef_answer is not None:
            self.predefined_answers.remove((idx1, idx2, predef_answer))
        return predef_answer

    def _query_points(self, idx1, idx2):
        answer = self.__get_answer(idx1, idx2)
        self.query_log.append((idx1, idx2, answer))
        return answer

    def __get_answer(self, idx1, idx2):
        # if there is a predefined answer return it
        predef_answer = self.get_predefined_answer(idx1, idx2)
        if predef_answer is not None:
            return predef_answer
        else:
            # answer truthfully based on clusterlist
            for cluster in self.cluster_list:
                if idx1 in cluster and idx2 in cluster:
                    return True
            return False

    def query_limit_reached(self):
        # there is no query limit
        return False
