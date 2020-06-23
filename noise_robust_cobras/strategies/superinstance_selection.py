import itertools
import random

import numpy as np


class SuperinstanceSelectionHeuristic:
    """
        A class representing an heuristic to select a super-instance from a set of super-instances
        This is used to select a super-instance to split further (at the start of the splitting phase)
        and in some SplitlevelEstimationStrategy's

    """

    def __init__(self):
        self.clusterer = None

    def get_name(self):
        return type(self).__name__

    def set_clusterer(self, clusterer):
        self.clusterer = clusterer

    def choose_superinstance(self, superinstances):
        pass


class RandomSelectionHeuristic(SuperinstanceSelectionHeuristic):
    def choose_superinstance(self, superinstances):
        return random.choice(superinstances)


class LeastInstancesSelectionHeuristic(SuperinstanceSelectionHeuristic):
    def choose_superinstance(self, superinstances):
        return min(superinstances, key=lambda superinstance: len(superinstance.indices))


class MostInstancesSelectionHeuristic(SuperinstanceSelectionHeuristic):
    def choose_superinstance(self, superinstances):
        return max(superinstances, key=lambda superinstance: len(superinstance.indices))


class MaximumMeanDistanceBetweenInstancesSelection(SuperinstanceSelectionHeuristic):
    def choose_superinstance(self, superinstances):
        best_superinstance = None
        best_score = None

        for superinstance in superinstances:
            mean_distance_between_instances = np.mean(
                [
                    np.linalg.norm(instance1 - instance2)
                    for instance1, instance2 in itertools.combinations(
                        superinstance.data[superinstance.indices], 2
                    )
                ]
            )
            if best_score is None or mean_distance_between_instances > best_score:
                best_score, best_superinstance = (
                    mean_distance_between_instances,
                    superinstance,
                )

        return best_superinstance


class MaximumDistanceToRepresentativeSelection(SuperinstanceSelectionHeuristic):
    def choose_superinstance(self, superinstances):
        best_superinstance = None
        best_score = None

        for superinstance in superinstances:
            representative = superinstance.data[superinstance.representative_idx]
            max_distance_to_representative = max(
                np.array(np.linalg.norm(representative - instance))
                for instance in superinstance.data[superinstance.indices]
            )
            if best_score is None or max_distance_to_representative > best_score:
                best_score, best_superinstance = (
                    max_distance_to_representative,
                    superinstance,
                )

        return best_superinstance


class MaximumMeanDistanceToRepresentativeSelection(SuperinstanceSelectionHeuristic):
    def choose_superinstance(self, superinstances):
        best_superinstance = None
        best_score = None

        for superinstance in superinstances:
            representative = superinstance.data[superinstance.representative_idx]
            mean_distance_to_representative = np.mean(
                [
                    np.linalg.norm(representative - instance)
                    for instance in superinstance.data[superinstance.indices]
                ]
            )
            if best_score is None or mean_distance_to_representative > best_score:
                best_score, best_superinstance = (
                    mean_distance_to_representative,
                    superinstance,
                )

        return best_superinstance
