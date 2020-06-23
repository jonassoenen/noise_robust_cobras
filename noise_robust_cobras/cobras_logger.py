import copy
import time
from typing import List

import numpy as np

from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint


class NopLogger(object):
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop


class ClusteringLogger:
    def __init__(self):
        # start time
        self.start_time = None

        # basic logging of intermediate results
        self.intermediate_results = []

        # all constraints obtained from the user
        self.all_user_constraints = []

        # algorithm phases
        self.current_phase = None
        self.algorithm_phases = []

        # (detected) noisy constraints
        self.corrected_constraint_sets = []
        self.noisy_constraint_data = []
        self.detected_noisy_constraint_data = []

        # clustering to store
        self.clustering_to_store = None

        # execution time
        self.execution_time = None

    #########################
    # information retrieval #
    #########################

    def get_all_clusterings(self):
        return [cluster for cluster, _, _ in self.intermediate_results]

    def get_runtimes(self):
        return [runtime for _, runtime, _ in self.intermediate_results]

    def get_ml_cl_constraint_lists(self):
        ml = []
        cl = []
        for constraint in self.all_user_constraints:
            if constraint.is_ML():
                ml.append(constraint.get_instance_tuple())
            else:
                cl.append(constraint.get_instance_tuple())
        return ml, cl

    def add_mistake_information(self, ground_truth_querier):
        for i, (constraint_number, constraint_copy) in enumerate(
            self.corrected_constraint_sets
        ):
            mistakes = []
            for con in constraint_copy:
                if (
                    ground_truth_querier.query(*con.get_instance_tuple()).is_ML()
                    != con.is_ML()
                ):
                    mistakes.append(con)
            self.corrected_constraint_sets[i] = (
                constraint_number,
                constraint_copy,
                mistakes,
            )

    ###################
    # log constraints #
    ###################
    def log_new_user_query(self, constraint):
        # add the constraint to all_user_constraints
        self.all_user_constraints.append(constraint)

        # keep algorithm phases up to date
        self.algorithm_phases.append(self.current_phase)

        # intermediate clustering results
        self.intermediate_results.append(
            (
                self.clustering_to_store,
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )
        )

    ##################
    # execution time #
    ##################

    def log_start_clustering(self):
        self.start_time = time.time()

    def log_end_clustering(self):
        self.execution_time = time.time() - self.start_time

    ##############
    # phase data #
    ##############

    def log_entering_phase(self, phase):
        self.current_phase = phase

    ###############
    # clusterings #
    ###############

    def update_clustering_to_store(self, clustering):
        if isinstance(clustering, np.ndarray):
            self.clustering_to_store = clustering.tolist()
        elif isinstance(clustering, list):
            self.clustering_to_store = list(clustering)
        else:
            self.clustering_to_store = clustering.construct_cluster_labeling()

    def update_last_intermediate_result(self, clustering):
        if len(self.intermediate_results) == 0:
            return
        if not isinstance(clustering, np.ndarray):
            self.intermediate_results[-1] = (
                clustering.construct_cluster_labeling(),
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )
        else:
            self.intermediate_results[-1] = (
                clustering.tolist(),
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )

    #####################
    # noisy constraints #
    #####################

    def log_corrected_constraint_set(self, constraints):
        constraint_copy: List[Constraint] = [copy.copy(con) for con in constraints]
        current_constraint_number = len(self.all_user_constraints)
        self.corrected_constraint_sets.append(
            (current_constraint_number, constraint_copy)
        )

    def log_detected_noisy_constraints(self, constraints):
        con_length = len(self.all_user_constraints)
        for con in constraints:
            self.detected_noisy_constraint_data.append((con_length, copy.copy(con)))
