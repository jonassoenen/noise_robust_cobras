from collections import defaultdict

from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint
from noise_robust_cobras.noise_robust.datastructures.constraint_index import (
    ConstraintIndex,
)


class Cycle:
    """
        A class that represents a valid constraint cycle
        attributes:
        - constraints: a list of constraints the way they appear in the cycle (starts at a random point in the cycle)
        - sorted_constraints: a tuple of constraints that is sorted for __eq__ and __hash__
        - number_of_CLs: the number of CL constraints in this cycle
    """

    def __init__(self, constraints, composed_from=None, number_of_CLs=None):
        assert Cycle.is_valid_constraint_set_for_cycle(constraints)
        self.constraints = set(constraints)
        self.sorted_constraints = Cycle.sort_constraints(constraints)
        self.composed_from = set(composed_from) if composed_from is not None else {self}
        if number_of_CLs is None:
            self.number_of_CLs = sum(
                1 for constraint in constraints if constraint.is_CL()
            )
        else:
            self.number_of_CLs = number_of_CLs

    @staticmethod
    def compose_multiple_cycles_ordered(cycles):
        composed_cycle = cycles[0]
        for to_compose in cycles[1:]:
            composed_cycle = composed_cycle.compose_with(to_compose)
            if composed_cycle is None:
                break
        return composed_cycle

    @staticmethod
    def compose_multiple_cycles(cycles):
        composed_constraints = set(cycles[0].constraints)
        composed_from = set(cycles[0].composed_from)
        for to_compose in cycles[1:]:
            composed_constraints.symmetric_difference_update(to_compose.constraints)
            composed_from.symmetric_difference_update(to_compose.composed_from)
        if not Cycle.is_valid_constraint_set_for_cycle(composed_constraints):
            return None
        return Cycle(composed_constraints, composed_from=composed_from)

    @staticmethod
    def make_cycle_from_raw_cons(raw_constraints):
        constraints = Constraint.raw_constraints_to_constraints(raw_constraints)
        return Cycle(constraints)

    @staticmethod
    def cycle_from_instances(instances):
        instances = [int(i) for i in instances]
        raw_constraints = list(zip(instances[:-1], instances[1:])) + [
            (instances[0], instances[-1])
        ]
        return Cycle.make_cycle_from_raw_cons(raw_constraints)

    @staticmethod
    def cycle_from_instances_constraint_index(instances, constraint_index):
        instances = [int(i) for i in instances]
        raw_constraints = list(zip(instances[:-1], instances[1:])) + [
            (instances[0], instances[-1])
        ]
        return Cycle(constraint_index.instance_tuples_to_constraints(raw_constraints))

    @staticmethod
    def is_valid_constraint_set_for_cycle(constraints):
        if len(constraints) == 0:
            return False
        # check if each instance occurs twice
        count = defaultdict(lambda: 0)
        for constraint in constraints:
            count[constraint.i1] += 1
            count[constraint.i2] += 1
        for key, value in count.items():
            if value != 2:
                return False

        # check if all constraints are connected
        all_sets = []
        for constraint in constraints:
            found_sets = [
                s for s in all_sets if constraint.i1 in s or constraint.i2 in s
            ]
            if len(found_sets) == 0:
                all_sets.append({constraint.i1, constraint.i2})
            elif len(found_sets) == 1:
                found_sets[0].update(constraint.get_instance_tuple())
            elif len(found_sets) == 2:
                found_sets[0].update(found_sets[1])
                all_sets.remove(found_sets[1])
        return len(all_sets) == 1

    def is_valid_cycle(self):
        return Cycle.is_valid_constraint_set_for_cycle(self.constraints)

    def get_sorted_constraint_list(self):
        """

        :return: a list of all constraints in the order by which they appear in the cycle with an arbitrary starting constraints
        """
        all_constraints = list(self.constraints)
        start_constraint = all_constraints[0]
        temp_index = ConstraintIndex()
        for constraint in all_constraints[1:]:
            temp_index.add_constraint(constraint)

        current_list = [(start_constraint.get_instance_tuple(), start_constraint)]
        current_instance = start_constraint.i2
        while len(temp_index.constraints) > 0:
            matching_constraints = temp_index.find_constraints_for_instance(
                current_instance
            )
            if len(matching_constraints) == 1:
                matching_constraint = list(matching_constraints)[0]
            else:
                raise Exception("Not a valid cycle!")

            other_instance = matching_constraint.get_other_instance(current_instance)
            current_list.append(
                ((current_instance, other_instance), matching_constraint)
            )
            current_instance = other_instance
            temp_index.remove_constraint(matching_constraint)

        # check if the cycle is complete
        if start_constraint.i1 != current_instance:
            raise Exception("Not a valid cycle!")

        return current_list

    def compose_with(self, other_cycle):
        if len(self.constraints.intersection(other_cycle.constraints)) == 0:
            return None
        new_constraints = set(self.constraints).symmetric_difference(
            other_cycle.constraints
        )
        if len(new_constraints) == 0:
            return None
        if not Cycle.is_valid_constraint_set_for_cycle(new_constraints):
            return None
        new_cycle = Cycle(
            new_constraints,
            other_cycle.composed_from.symmetric_difference(self.composed_from),
        )
        return new_cycle

    def replace_constraint(self, old_constraint, new_constraint):
        assert old_constraint in self.constraints
        new_constraints = set(self.constraints)
        new_constraints.remove(old_constraint)
        new_constraints.add(new_constraint)
        return Cycle(new_constraints)

    @staticmethod
    def sort_constraints(constraints):
        return tuple(sorted(constraints))

    def is_useful(self):
        return self.number_of_CLs <= 2

    def is_inconsistent(self):
        return self.number_of_CLs == 1

    def __iter__(self):
        return self.constraints.__iter__()

    def __len__(self):
        return len(self.constraints)

    def __eq__(self, other):
        if other == None:
            return False
        return self.sorted_constraints == other.sorted_constraints

    def __contains__(self, item):
        return item in self.constraints

    def __hash__(self):
        return hash(self.sorted_constraints)

    def __repr__(self):
        return str(self)

    def __str__(self):
        # return ",".join([str(constraint) for constraint in self.constraints])
        return ",".join([str(con) for _, con in self.get_sorted_constraint_list()])
