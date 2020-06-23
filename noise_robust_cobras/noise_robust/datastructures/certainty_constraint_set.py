import itertools
from collections import defaultdict
from typing import Set, List, Tuple

from noise_robust_cobras.cobras_logger import ClusteringLogger
from noise_robust_cobras.noise_robust.datastructures.componenttracker import (
    MLComponent,
    ComponentTracker,
)
from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint
from noise_robust_cobras.noise_robust.datastructures.constraint_index import (
    ConstraintIndex,
)
from noise_robust_cobras.noise_robust.datastructures.cycle import Cycle
from noise_robust_cobras.noise_robust.datastructures.minimal_cycle_index import (
    MinimalCycleIndex,
)


class NewCertaintyConstraintSet:
    """
    CertaintyConstraintSet is a datastructure that allows to get a certainty for each constraint

    It uses a ConstraintIndex and a CycleIndex (UsefullCycleIndex)
    """

    def __init__(
        self,
        minimum_approximation_order,
        maximum_approximation_order,
        noise_probability,
        cobras_logger: ClusteringLogger,
        cycle_index=None,
    ):
        """

        :param minimum_approximation_order:
        :param maximum_approximation_order:
        :param noise_probability:
        :param cobras_logger:
        :param cycle_index:
        """
        self._cobras_log = cobras_logger
        self.constraint_index = ConstraintIndex()
        self.component_tracker = ComponentTracker(self.constraint_index)
        if cycle_index is None:
            self.cycle_index = MinimalCycleIndex(self.constraint_index)
        else:
            self.cycle_index = cycle_index
            self.cycle_index.constraint_index = self.constraint_index
            self.component_tracker.constraint_index = self.constraint_index

        self.minimum_approximation_order = minimum_approximation_order
        self.maximum_approximation_order = maximum_approximation_order
        self.noise_probability = noise_probability
        self.cached_noise_list = None
        self.cached_probabilities = None

    # constraint operations #

    def add_constraint(self, constraint):
        """
            Adds a new constraint to the certainty constraint set
            and updates all underlying datastructures
            :param constraint:
            :return:
        """
        # add the constraint to the constraint_index
        new_constraint = self.constraint_index.add_constraint(constraint)

        # this makes cached_probabilities out of date!
        self.cached_noise_list = None

        # if this was a new constraint the cycle_index needs to be updated as well
        # and the cached noise_list is invalidated
        if new_constraint:
            self.cached_noise_list = None
            self.cycle_index.add_constraint(constraint)
            self.component_tracker.add_constraint(constraint)

    def flip_constraint(self, constraint):
        """
            flips the given constraint in all datastructures
        """

        self.cached_probabilities = None
        self.cached_noise_list = None

        flipped_constraint = constraint.flip()
        # print("changing ", constraint, "to", flipped_constraint)
        self.constraint_index.replace_constraint(constraint, flipped_constraint)
        self.cycle_index.replace_constraint(constraint, flipped_constraint)
        self.component_tracker.flip_constraint(constraint, flipped_constraint)
        return flipped_constraint

    # UNUSED
    def remove_constraint(self, constraint_to_remove):
        """
            Removes the given constraint from all datastructures
        :param constraint_to_remove:
        :return:
        """
        # the cached noise_list is invalidated
        self.cached_noise_list = None
        self.cached_probabilities = None

        # remove from constraint index
        self.constraint_index.remove_constraint(constraint_to_remove)

        # remove from cycle_index
        self.cycle_index.remove_cycles_with_constraint(constraint_to_remove)

    def is_inconsistent(self):
        """
        :return: whether or not this certainty constraint set is inconsistent
        :rtype: bool
        """
        return self.component_tracker.is_inconsistent()

    # generate noise list

    def get_noise_list(self) -> List[Tuple[Tuple[Constraint, ...], ComponentTracker]]:
        """

        :return:
        :rtype: list[tuple[tuple[Constraint, ...], ComponentTracker]]]
        """
        if self.cached_noise_list is None:
            self.cached_noise_list = self.calculate_noise_list()
        return self.cached_noise_list

    def calculate_noise_list(self):
        """
        This function generates the noise list by calculating the noise list of order
        self.minimum_approximation_order If this fails (because there is more noise) the procedure tries higher
        approximation orders. This is repeated until the maximum approximation order is reached. At this point the
        algorithm decides that the noise is uncorrecteable

            TODO If I were to calculate the noise_list breadth first I could just keep searching untill I found a
            solution and then continue for 2 more orders

            :return: the generated noise list
            :rtype: list[tuple[tuple[Constraint, ...], ComponentTracker]]]
        """
        inconsistent_cycles = self.cycle_index.all_inconsistent_cycles
        noise_list = []
        current_order = self.minimum_approximation_order
        current_number_of_wrong_constraints = self.constraint_index.nb_wrong_constraints
        # print("START AGAIN=======================")
        while (
            len(noise_list) == 0 and current_order <= self.maximum_approximation_order
        ):
            # print("START ", current_order)
            noise_list = self.solve_all_solutions(
                list(inconsistent_cycles),
                (),
                self.component_tracker,
                current_number_of_wrong_constraints + current_order,
                set(),
                self.constraint_index,
            )
            current_order += 1
            # print("old current order=", current_order)
        if len(noise_list) == 0:
            # so this is still a problem
            raise Exception(
                "noise to complex to correct with order "
                + str(self.maximum_approximation_order)
            )
        return noise_list

    def solve_all_solutions(
        self,
        cycles_to_resolve,
        flipped_constraints,
        component_tracker: ComponentTracker,
        upper_bound,
        visited_constraints: Set[Constraint],
        component,
    ):
        """
            Complex recursive function that calculates the noise list by introducing extra noise to resolve inconsistent cycles
        :param cycles_to_resolve: these are the inconsistent cycles that still need to be resolved
            This variable contains all inconsistent basis cycles OR some of the inconsistent non basis cycles
        :type cycles_to_resolve: list[Cycle]
        :param flipped_constraints: a tuple of the constraints that are at this point considered noisy
        :type flipped_constraints: tuple[Constraint, ...]
        :param component_tracker: a datastructure that keeps track of the components (taking into account the current flipped_constraints)
        :type component_tracker:ComponentTracker
        :param upper_bound: the maximal number of noisy constraints that should be considered while building this noise list
        :type upper_bound: int
        :param visited_constraints: the constraints that already have been considered noisy and should thus not be considered noisy in subsequent steps
            This is used to ensure that the different paths in the search tree are disjunct
        :type visited_constraints: set[Constraint]
        :param component: the component for which we are calculating the noise_list
            note: this is not really used at the moment! The full constraint-index is passed in!
        :return: a list of noise_items
        :rtype: list[tuple[Constraint, ...]]
        """

        if upper_bound is not None:
            if (
                self.get_number_of_wrong_constraints_given_flipped_constraints(
                    flipped_constraints
                )
                > upper_bound
            ):
                return []

        if len(cycles_to_resolve) == 0:
            # there are no inconsistent basis cycles

            if component_tracker.is_inconsistent():
                # there are still non basis cycles that are inconsistent!
                inconsistent_non_basis_cycles = self.get_inconsistent_non_basis_cycles(
                    component_tracker
                )
                return self.solve_all_solutions(
                    inconsistent_non_basis_cycles,
                    flipped_constraints,
                    component_tracker,
                    upper_bound,
                    visited_constraints,
                    component,
                )
            else:
                # there are no inconsistent non-basis cycles
                # the current flipped_constraints is a solution
                solutions = [
                    (
                        flipped_constraints,
                        component_tracker.get_component_query_answerer(),
                    )
                ]

                # there might be longer noise items that are solutions as well
                # so consider all constraints that have not been considered noisy yet as noisy candidates
                possible_noisy_constraints = component.constraints.difference(
                    visited_constraints
                )
                constraints_not_to_consider_noisy = []
                for constraint in possible_noisy_constraints:
                    # consider the current and all previously visited constraints as not noisy
                    # to avoid having duplicate solutions
                    constraints_not_to_consider_noisy.append(constraint)
                    new_flipped_constraints = flipped_constraints + (constraint,)

                    # BEFORE: make a copy
                    # new_component_tracker = component_tracker.copy_component_tracker_with_extra_flipped_constraints(
                    #     (constraint,))
                    # NOW: just flip the constraint and undo flip later
                    component_tracker.flip_additional_constraint(constraint)
                    new_component_tracker = component_tracker
                    new_visited_constraints = visited_constraints.union(
                        constraints_not_to_consider_noisy
                    )
                    new_cycles_to_resolve = [
                        cycle
                        for cycle in self.cycle_index.get_all_cycles_for_constraint(
                            constraint
                        )
                        if self.get_number_of_cls_in_cycle_given_flipped_constraints(
                            cycle,
                            # new_flipped_constraints) == 1]
                            new_flipped_constraints,
                        )
                        == 1
                    ]
                    extra_solutions = self.solve_all_solutions(
                        new_cycles_to_resolve,
                        new_flipped_constraints,
                        new_component_tracker,
                        upper_bound,
                        new_visited_constraints,
                        component,
                    )
                    component_tracker.undo_flip_additional_constraint(constraint)
                    solutions.extend(extra_solutions)
                return solutions

        else:
            # there are still cycles_to resolve so pick the first cycle and resolve it
            cycle_to_resolve = cycles_to_resolve.pop(0)
            other_cycles_to_resolve = cycles_to_resolve

            original_nb_of_cls = self.get_number_of_cls_in_cycle_given_flipped_constraints(
                cycle_to_resolve, flipped_constraints
            )
            # this cycle is not inconsistent
            if original_nb_of_cls == 0 or original_nb_of_cls > 1:
                return self.solve_all_solutions(
                    other_cycles_to_resolve,
                    flipped_constraints,
                    component_tracker,
                    upper_bound,
                    visited_constraints,
                    component,
                )

            # find all extra noise items to get a consistent assignment in the current cycle to resolve
            # list of tuples
            possible_extra_noise_constraints = cycle_to_resolve.constraints.difference(
                visited_constraints
            )

            # for each of these extra noise_items resolve
            solutions = []
            constraints_not_to_consider_noisy = []
            for extra_noise_constraint in possible_extra_noise_constraints:
                constraints_not_to_consider_noisy.append(extra_noise_constraint)
                new_flipped_constraints = flipped_constraints + (
                    extra_noise_constraint,
                )
                # new_component_tracker = component_tracker.copy_component_tracker_with_extra_flipped_constraints(
                #     (extra_noise_constraint,))
                component_tracker.flip_additional_constraint(extra_noise_constraint)
                new_component_tracker = component_tracker
                new_visited_constraints = visited_constraints.union(
                    constraints_not_to_consider_noisy
                )
                extra_cycles_to_resolve = [
                    cycle
                    for cycle in itertools.chain(
                        self.cycle_index.get_all_cycles_for_constraint(
                            extra_noise_constraint
                        ),
                        other_cycles_to_resolve,
                    )
                    if self.get_number_of_cls_in_cycle_given_flipped_constraints(
                        cycle, new_flipped_constraints
                    )
                    == 1
                ]
                all_cycles_to_resolve = extra_cycles_to_resolve

                extra_solutions = self.solve_all_solutions(
                    all_cycles_to_resolve,
                    new_flipped_constraints,
                    new_component_tracker,
                    upper_bound,
                    new_visited_constraints,
                    component,
                )
                component_tracker.undo_flip_additional_constraint(
                    extra_noise_constraint
                )
                solutions.extend(extra_solutions)
            return solutions

    def solve_all_solutionsOLD(
        self,
        cycles_to_resolve,
        flipped_constraints,
        component_tracker: ComponentTracker,
        upper_bound,
        visited_constraints: Set[Constraint],
        component,
    ):
        """
            Complex recursive function that calculates the noise list by introducing extra noise to resolve inconsistent cycles
        :param cycles_to_resolve: these are the inconsistent cycles that still need to be resolved
            This variable contains all inconsistent basis cycles OR some of the inconsistent non basis cycles
        :type cycles_to_resolve: list[Cycle]
        :param flipped_constraints: a tuple of the constraints that are at this point considered noisy
        :type flipped_constraints: tuple[Constraint, ...]
        :param component_tracker: a datastructure that keeps track of the components (taking into account the current flipped_constraints)
        :type component_tracker:ComponentTracker
        :param upper_bound: the maximal number of noisy constraints that should be considered while building this noise list
        :type upper_bound: int
        :param visited_constraints: the constraints that already have been considered noisy and should thus not be considered noisy in subsequent steps
            This is used to ensure that the different paths in the search tree are disjunct
        :type visited_constraints: set[Constraint]
        :param component: the component for which we are calculating the noise_list
            note: this is not really used at the moment! The full constraint-index is passed in!
        :return: a list of noise_items
        :rtype: list[tuple[Constraint, ...]]
        """

        if upper_bound is not None:
            if (
                self.get_number_of_wrong_constraints_given_flipped_constraints(
                    flipped_constraints
                )
                > upper_bound
            ):
                return []

        if len(cycles_to_resolve) == 0:
            # there are no inconsistent basis cycles

            if component_tracker.is_inconsistent():
                # there are still non basis cycles that are inconsistent!
                inconsistent_non_basis_cycles = self.get_inconsistent_non_basis_cycles(
                    component_tracker
                )
                return self.solve_all_solutions(
                    inconsistent_non_basis_cycles,
                    flipped_constraints,
                    component_tracker,
                    upper_bound,
                    visited_constraints,
                    component,
                )
            else:
                # there are no inconsistent non-basis cycles
                # the current flipped_constraints is a solution
                solutions = [(flipped_constraints, component_tracker)]

                # there might be longer noise items that are solutions as well
                # so consider all constraints that have not been considered noisy yet as noisy candidates
                possible_noisy_constraints = component.constraints.difference(
                    visited_constraints
                )
                constraints_not_to_consider_noisy = []
                for constraint in possible_noisy_constraints:
                    # consider the current and all previously visited constraints as not noisy
                    # to avoid having duplicate solutions
                    constraints_not_to_consider_noisy.append(constraint)
                    new_flipped_constraints = flipped_constraints + (constraint,)
                    new_component_tracker = component_tracker.copy_component_tracker_with_extra_flipped_constraints(
                        (constraint,)
                    )
                    new_visited_constraints = visited_constraints.union(
                        constraints_not_to_consider_noisy
                    )

                    new_cycles_to_resolve = [
                        cycle
                        for cycle in self.cycle_index.get_all_cycles_for_constraint(
                            constraint
                        )
                        if self.get_number_of_cls_in_cycle_given_flipped_constraints(
                            cycle,
                            # new_flipped_constraints) == 1]
                            new_flipped_constraints,
                        )
                        == 1
                    ]

                    extra_solutions = self.solve_all_solutions(
                        new_cycles_to_resolve,
                        new_flipped_constraints,
                        new_component_tracker,
                        upper_bound,
                        new_visited_constraints,
                        component,
                    )
                    solutions.extend(extra_solutions)
                return solutions

        else:
            # there are still cycles_to resolve so pick the first cycle and resolve it
            cycle_to_resolve = cycles_to_resolve.pop(0)
            other_cycles_to_resolve = cycles_to_resolve

            original_nb_of_cls = self.get_number_of_cls_in_cycle_given_flipped_constraints(
                cycle_to_resolve, flipped_constraints
            )
            # this cycle is not inconsistent
            if original_nb_of_cls == 0 or original_nb_of_cls > 1:
                return self.solve_all_solutions(
                    other_cycles_to_resolve,
                    flipped_constraints,
                    component_tracker,
                    upper_bound,
                    visited_constraints,
                    component,
                )

            # find all extra noise items to get a consistent assignment in the current cycle to resolve
            # list of tuples
            possible_extra_noise_constraints = cycle_to_resolve.constraints.difference(
                visited_constraints
            )

            # for each of these extra noise_items resolve
            solutions = []
            constraints_not_to_consider_noisy = []
            for extra_noise_constraint in possible_extra_noise_constraints:
                constraints_not_to_consider_noisy.append(extra_noise_constraint)
                new_flipped_constraints = flipped_constraints + (
                    extra_noise_constraint,
                )
                new_component_tracker = component_tracker.copy_component_tracker_with_extra_flipped_constraints(
                    (extra_noise_constraint,)
                )
                new_visited_constraints = visited_constraints.union(
                    constraints_not_to_consider_noisy
                )
                extra_cycles_to_resolve = [
                    cycle
                    for cycle in self.cycle_index.get_all_cycles_for_constraint(
                        extra_noise_constraint
                    )
                    if self.get_number_of_cls_in_cycle_given_flipped_constraints(
                        cycle, new_flipped_constraints
                    )
                    == 1
                ]
                all_cycles_to_resolve = (
                    other_cycles_to_resolve + extra_cycles_to_resolve
                )

                extra_solutions = self.solve_all_solutions(
                    all_cycles_to_resolve,
                    new_flipped_constraints,
                    new_component_tracker,
                    upper_bound,
                    new_visited_constraints,
                    component,
                )
                solutions.extend(extra_solutions)
            return solutions

    # extra consistency check

    def get_inconsistent_non_basis_cycles(self, component_tracker):
        inconsistent_components = [
            component
            for component in component_tracker.components
            if component.is_inconsistent()
        ]
        all_inconsistent_cycles = []
        for inconsistent_component in inconsistent_components:
            inconsistent_cycles = self.get_inconsistent_cycles_from_inconsistent_component(
                inconsistent_component, component_tracker.flipped_constraints
            )
            all_inconsistent_cycles.extend(inconsistent_cycles)
        return all_inconsistent_cycles

    def get_inconsistent_cycles_from_inconsistent_component(
        self, component, flipped_constraints
    ):
        assert component.is_inconsistent()
        inter_component_cls = component.inter_component_cls
        # mls = component.ml_constraints
        # each of the inter_component_cls forms AT LEAST one inconsistent cycle with the mls
        inconsistent_cycles = []
        for cl in inter_component_cls:
            # simple solution search a ml_path between cl.i1, cl.i2 only using constraints in ml
            visited = {cl.i1}
            queue = [(cl.i1, [])]
            path = None
            while len(queue) > 0 and path is None:
                current_node, current_path = queue.pop(0)
                all_constraints = self.constraint_index.find_constraints_for_instance(
                    current_node
                )
                ml_constraints = [
                    con
                    for con in all_constraints
                    if self.get_type_of_constraint_given_flipped_constraints(
                        con, flipped_constraints
                    )
                ]
                possible_extensions = [
                    (con.get_other_instance(current_node), con)
                    for con in ml_constraints
                    if con.get_other_instance(current_node) not in visited
                ]
                for next_node, con in possible_extensions:
                    if next_node == cl.i2:
                        path = current_path + [con]
                        break
                    visited.add(next_node)
                    new_path = current_path + [con]
                    queue.append((next_node, new_path))
                if path is not None:
                    break
            # I should always find such a path!
            assert path is not None
            # the path is already based on constraints from the constraint index
            # however, the cannot-link is not!
            if cl.flip() in flipped_constraints:
                new_inconsistent_cycle = Cycle(path + [cl.flip()])
            else:
                new_inconsistent_cycle = Cycle(path + [cl])
            inconsistent_cycles.append(new_inconsistent_cycle)
        return inconsistent_cycles

    # noise list to probabilities

    def get_probability_for_single_constraint(self, prob_constraint):
        if self.cached_noise_list is None:
            self.cached_noise_list = self.calculate_noise_list()
            self.cached_probabilities = self.process_noise_list_to_probabilities(
                [noise_item for noise_item, _ in self.cached_noise_list]
            )

        if self.cached_probabilities is None:
            self.cached_probabilities = self.process_noise_list_to_probabilities(
                [noise_item for noise_item, _ in self.cached_noise_list]
            )
        if prob_constraint not in self.cached_probabilities:
            return 1 - self.cached_probabilities[prob_constraint.flip()]
        return self.cached_probabilities[prob_constraint]

    def get_probability_of_constraints_ignoring_edge_cases(self, constraints):
        """
            calculates the certainty that each constraint in constraints is satisfied
            However, this function ignores constraints that belong to the following edge case classes:
            - the constraints in a must-link component that are fully connected by must-links but are not certain enough
             (for instance an isolated must-link constraint OR 3 instances that are connected by ML's)
            - cannot-link constraints between 2 components where all possible constraints between the 2 components have been asked but are not certain enough
             (for instance a cannot-link between 2 ml-components of size 1 OR a ml-component of size 1 and a ml-component of size 2 and the 2 cl's are already queried)
            :param constraints:
            :return:
        """
        no_edge_case_constraints = self.filter_out_constraints_where_gathering_extra_certainty_is_not_possible(
            constraints
        )
        return self.get_probability_for_constraints(no_edge_case_constraints)

    def get_probability_for_constraints(self, constraints):
        satisfied_sum = 0
        unsatisfied_sum = 0
        noise_list = self.get_noise_list()
        for item, component_query_answerer in noise_list:
            all_constraints_satisfied = self.are_constraints_satisfied_given_flipped_constraints(
                constraints, item
            )
            if all_constraints_satisfied:
                satisfied_sum += self.calculate_weight_of_noise_item(item)
            else:
                unsatisfied_sum += self.calculate_weight_of_noise_item(item)
        satisfied_prob = satisfied_sum / (satisfied_sum + unsatisfied_sum)
        return satisfied_prob

    def calculate_weight_of_noise_item(self, noise_item):
        correct_u = 1 - self.noise_probability
        wrong_u = self.noise_probability
        nb_correct_user_constraints = self.constraint_index.nb_correct_constraints
        nb_wrong_user_constraints = self.constraint_index.nb_wrong_constraints
        times_correct_but_counted_as_wrong = sum(
            con.get_times_other_seen() for con in noise_item
        )
        times_wrong_but_counted_as_correct = sum(
            con.get_times_seen() for con in noise_item
        )

        total_correct = (
            nb_correct_user_constraints
            - times_wrong_but_counted_as_correct
            + times_correct_but_counted_as_wrong
        )
        total_wrong = (
            nb_wrong_user_constraints
            - times_correct_but_counted_as_wrong
            + times_wrong_but_counted_as_correct
        )

        # calculate the term
        weight = correct_u ** total_correct * wrong_u ** total_wrong
        return weight

    def filter_out_constraints_where_gathering_extra_certainty_is_not_possible(
        self, constraints
    ):
        filtered = []
        for constraint in constraints:
            if constraint.is_ML():
                # check if the component of the instance is fully connected
                component: MLComponent = self.component_tracker.get_component_of_instance(
                    constraint.i1
                )
                if component.is_fully_connected():
                    # do not count toward minimum certainty
                    continue
            else:
                # constraint is CL
                # check if all cannot-links between the 2 components exists
                comp1 = self.component_tracker.get_component_of_instance(constraint.i1)
                comp2 = self.component_tracker.get_component_of_instance(constraint.i2)
                intra_component_cls = self.component_tracker.get_intra_component_cls_between(
                    comp1, comp2
                )
                if len(intra_component_cls) == len(comp1.instances) * len(
                    comp2.instances
                ):
                    # do not count towards minimum certainty
                    continue
            filtered.append(constraint)
        return filtered

    def are_constraints_satisfied_given_flipped_constraints(
        self, constraints, noise_item
    ):
        all_constraints_satisfied = True
        for must_sat_con in constraints:
            i1, i2 = must_sat_con.get_instance_tuple()
            matching_constraints = self.constraint_index.find_constraints_between_instances(
                i1, i2
            )
            assert (
                len(matching_constraints) == 1
            ), "there should only be one matching constraint"
            matching_constraint = next(matching_constraints.__iter__())
            constraint_value_in_current_assignment = self.get_type_of_constraint_given_flipped_constraints(
                matching_constraint, noise_item
            )
            if constraint_value_in_current_assignment != must_sat_con.is_ML():
                all_constraints_satisfied = False
                break
        return all_constraints_satisfied

    def process_noise_list_to_probabilities(self, noise_list):
        probabilities = self.process_component_noise_list_to_probabilities(
            None, noise_list, []
        )
        return probabilities

    def process_component_noise_list_to_probabilities(
        self, component, noise_list, dont_cares
    ):
        correct_u = 1 - self.noise_probability
        wrong_u = self.noise_probability

        probabilities = dict()
        # handle don't cares
        for dont_care in dont_cares:
            enumerator = (
                correct_u ** dont_care.get_times_seen()
                * wrong_u ** dont_care.get_times_other_seen()
            )
            denominator = (
                enumerator
                + wrong_u ** dont_care.get_times_seen()
                * correct_u ** dont_care.get_times_other_seen()
            )
            probabilities[dont_care] = enumerator / denominator

        # handle cares
        if component is None:
            cares = self.constraint_index.constraints
        else:
            cares = component.constraints.difference(dont_cares)
        if len(noise_list) == 0 and len(cares) > 0:
            print("len(noise_list is None)!")
        nb_correct_user_constraints = sum(
            constraint.get_times_seen() for constraint in cares
        )
        nb_wrong_user_constraints = sum(
            constraint.get_times_other_seen() for constraint in cares
        )
        denominator = 0
        weight_sum_dict = defaultdict(int)

        for noisy_consistent_assignment in noise_list:

            times_correct_but_counted_as_wrong = sum(
                con.get_times_other_seen() for con in noisy_consistent_assignment
            )
            times_wrong_but_counted_as_correct = sum(
                con.get_times_seen() for con in noisy_consistent_assignment
            )

            total_correct = (
                nb_correct_user_constraints
                - times_wrong_but_counted_as_correct
                + times_correct_but_counted_as_wrong
            )
            total_wrong = (
                nb_wrong_user_constraints
                - times_correct_but_counted_as_wrong
                + times_wrong_but_counted_as_correct
            )

            # calculate the term
            weight = correct_u ** total_correct * wrong_u ** total_wrong
            denominator += weight

            for noisy_con in noisy_consistent_assignment:
                weight_sum_dict[noisy_con] += weight

        for con in cares:
            enumerator = denominator - weight_sum_dict.get(con, 0)
            probability = enumerator / denominator
            probabilities[con] = probability

        return probabilities

    def process_full_component_noise_list_to_probabilities(self, noise_list):
        """
            TODO do this again!
            This is old code that expects a full component noise list to work with (so format list[tuple[Component,tuple[noise_list, dont_cares]]])
            at the moment not used because the noise_list is not split in different components!

        :param noise_list:
        :return:
        """
        all_probabilities = dict()
        for component, (noise_list, dont_cares) in noise_list.items():
            probabilities = self.process_component_noise_list_to_probabilities(
                component, noise_list, dont_cares
            )
            all_probabilities.update(probabilities)
        return all_probabilities

    # flipped constraints util

    def get_number_of_wrong_constraints_given_flipped_constraints(
        self, flipped_constraints
    ):
        # all the constraints that are wrong in the current assignment
        nb_wrong = self.constraint_index.nb_wrong_constraints
        times_correct_but_counted_as_wrong = sum(
            con.get_times_other_seen() for con in flipped_constraints
        )
        times_wrong_but_counted_as_correct = sum(
            con.get_times_seen() for con in flipped_constraints
        )
        # total_correct = nb_correct - times_wrong_but_counted_as_correct + times_correct_but_counted_as_wrong
        total_wrong = (
            nb_wrong
            - times_correct_but_counted_as_wrong
            + times_wrong_but_counted_as_correct
        )
        return total_wrong

    @staticmethod
    def get_number_of_cls_in_cycle_given_flipped_constraints(cycle, flipped_constraint):
        original_cls = cycle.number_of_CLs
        for con in flipped_constraint:
            if con in cycle:
                if con.is_ML():
                    original_cls += 1
                else:
                    # con is CL
                    original_cls -= 1
        return original_cls

    @staticmethod
    def get_type_of_constraint_given_flipped_constraints(con, flipped_constraints):
        if con in flipped_constraints:
            return not con.is_ML()
        return con.is_ML()
