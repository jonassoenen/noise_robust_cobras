from collections import defaultdict


class ConstraintComponent:
    def __init__(self, constraints=None):
        if constraints is None:
            self.constraints = []
            self.involved_instances = set()
        else:
            self.constraints = constraints
            self.involved_instances = {
                item for con in self.constraints for item in con.get_instance_tuple()
            }

    def constraint_belongs_to_component(self, constraint):
        return (
            constraint.i1 in self.involved_instances
            or constraint.i2 in self.involved_instances
        )

    def add_constraint(self, constraint):
        assert self.constraint_belongs_to_component(constraint)
        self.involved_instances.update(constraint.get_instance_tuple())
        self.constraints.append(constraint)

    def merge_with_component(self, other_component):
        assert (
            len(
                self.involved_instances.intersection(other_component.involved_instances)
            )
            > 0
        )
        self.involved_instances.update(other_component.involved_instances)
        self.constraints.extend(other_component.constraints)


class ConstraintIndex:
    """
        A class that stores constraints in a dictionary for fast retrieval
    """

    def __init__(self):
        self.constraint_index = defaultdict(set)
        self.inconsistent_pairs = []
        self.constraints = set()
        # TODO can be optimized to use a dictionary for fast lookup
        self.ml_components = []
        self.nb_correct_constraints = 0
        self.nb_wrong_constraints = 0

    def get_all_instances_with_constraint(self):
        return set(self.constraint_index.keys())

    def instance_tuples_to_constraints(self, instance_tuples):
        constraints = []
        for i1, i2 in instance_tuples:
            found_constraints = self.find_constraints_between_instances(i1, i2)
            assert len(found_constraints) == 1
            constraints.append(list(found_constraints)[0])
        return constraints

    def replace_constraint(self, old_constraint, new_constraint):
        self.remove_constraint(old_constraint)
        self.add_constraint(new_constraint)

    def get_component_of(self, constraint):
        """

        :param constraint:
        :return:
        :rtype: ConstraintComponent
        """
        matching_components = [
            component
            for component in self.ml_components
            if component.constraint_belongs_to_component(constraint)
        ]
        if len(matching_components) == 0:
            return None
        elif len(matching_components) == 1:
            return matching_components[0]
        else:
            raise Exception("components are inconsistent!")

    def add_constraint(self, constraint):
        """
            :param constraint:
            :return: whether or not the added constraint was a new constraint
            :rtype: bool
        """
        new_constraint = self.__add_constraint_to_index(constraint)
        if new_constraint:  # and constraint.is_ML():
            # add the constraint to the must-link components
            self.__add_constraint_to_ml_components(constraint)
        return new_constraint

    def __remove_constraint_from_ml_components(self, constraint):
        matching_ml_components = [
            component
            for component in self.ml_components
            if component.constraint_belongs_to_component(constraint)
        ]
        if len(matching_ml_components) == 1:
            # remove the component
            matching_component = matching_ml_components[0]
            self.ml_components.remove(matching_component)

            # read all the constraints from the component
            constraints = matching_component.constraints
            constraints.remove(constraint)

            for constraint in constraints:
                self.__add_constraint_to_ml_components(constraint)
        else:
            raise Exception(
                "should be only one component that contains the constraint to remove not {}".format(
                    len(matching_ml_components)
                )
            )

    def __add_constraint_to_ml_components(self, constraint):
        # assert constraint.is_ML()
        matching_ml_components = [
            component
            for component in self.ml_components
            if component.constraint_belongs_to_component(constraint)
        ]
        if len(matching_ml_components) == 0:
            # make a new ML-component
            new_component = ConstraintComponent([constraint])
            self.ml_components.append(new_component)
        elif len(matching_ml_components) == 1:
            # add constraint to the matching ml_component
            matching_component = matching_ml_components[0]
            matching_component.add_constraint(constraint)
        elif len(matching_ml_components) == 2:
            # merge 2 matching components and new constraint in a single new component
            matching_component1, matching_component2 = matching_ml_components
            matching_component1.add_constraint(constraint)
            matching_component1.merge_with_component(matching_component2)
            self.ml_components.remove(matching_component2)
        else:
            raise Exception("more than 2 matching components!")

    def __add_constraint_to_index(self, constraint):
        """
            :param constraint:
            :return: whether or not the just added constraint was a new constraint
        """
        # add the constraint to the constraint index
        set_to_search = self.find_constraints_between_instances(
            constraint.i1, constraint.i2
        )
        if len(set_to_search) == 0:
            self.constraints.add(constraint)
            self.constraint_index[constraint.i1].add(constraint)
            self.constraint_index[constraint.i2].add(constraint)
            self.nb_correct_constraints += constraint.get_times_seen()
            self.nb_wrong_constraints += constraint.get_times_other_seen()
            return True
        elif len(set_to_search) == 1:
            raise Exception("This should not happen!")
            # existing_constraint = next(set_to_search.__iter__())
            # existing_constraint.add_other_constraint(constraint)
            # if existing_constraint.is_ML() == constraint.is_ML():
            #     # the constraint is the same as the assignment in the constraint index
            #     self.nb_correct_constraints += constraint.get_times_seen()
            #     self.nb_wrong_constraints += constraint.get_times_other_seen()
            # else:
            #     # the constraint is wrong (based on the current constraint index)
            #     self.nb_correct_constraints += constraint.get_times_other_seen()
            #     self.nb_wrong_constraints += constraint.get_times_seen()
            # return False
        else:
            raise Exception(
                "at least two times the same constraint in constraintIndex!"
            )

    def remove_constraint(self, constraint):
        self.nb_correct_constraints -= constraint.get_times_seen()
        self.nb_wrong_constraints -= constraint.get_times_other_seen()
        to_remove = None
        for idx, (c1, c2) in enumerate(self.inconsistent_pairs):
            if c1 == constraint or c2 == constraint:
                to_remove = idx
                break
        if to_remove is not None:
            self.inconsistent_pairs.pop(to_remove)

        self.constraints.remove(constraint)
        self.constraint_index[constraint.i1].remove(constraint)
        self.constraint_index[constraint.i2].remove(constraint)

        if constraint.is_ML():
            self.__remove_constraint_from_ml_components(constraint)

    def get_ml_and_cl_tuple_lists(self):
        ml, cl = [], []
        for con in self.constraints:
            if con.is_ML():
                ml.append(con.get_instance_tuple())
            else:
                cl.append(con.get_instance_tuple())
        return ml, cl

    def get_all_mls(self):
        return [con for con in self.constraints if con.is_ML()]

    def get_all_cls(self):
        return [con for con in self.constraints if con.is_CL()]

    def get_number_of_constraints(self):
        return sum(
            con.get_times_seen() + con.get_times_other_seen()
            for con in self.constraints
        )

    def has_inconsistent_pairs(self):
        return len(self.inconsistent_pairs) > 0

    def is_constraint_part_of_inconsistent_pair(self, constraint):
        return any(constraint in pair for pair in self.inconsistent_pairs)

    def __len__(self):
        return len(self.constraints)

    def __iter__(self):
        return self.constraints.__iter__()

    def __contains__(self, constraint):
        return constraint in self.constraints

    def find_constraints_for_instance(self, instance):
        return self.constraint_index[instance]

    def does_constraint_between_instances_exist(self, i1, i2):
        return len(self.find_constraints_between_instances(i1, i2)) > 0

    def get_constraint(self, constraint):
        if constraint not in self:
            return None
        return next(
            con
            for con in self.find_constraints_between_instances(
                constraint.i1, constraint.i2
            )
            if con.is_ML() == constraint.is_ML()
        )

    def find_constraints_between_instances(self, i1, i2):
        return self.constraint_index[i1].intersection(self.constraint_index[i2])

    def find_constraints_between_instance_set(self, instances):
        resulting_constraints = set()
        for instance in instances:
            constraints_with_instance = self.find_constraints_for_instance(instance)
            for constraint in constraints_with_instance:
                other_instance = constraint.get_other_instance(instance)
                # instance < other_instance is a quick reject every constraint only needs to be added once to the
                # resulting_constraints
                if (
                    instance < other_instance
                    and constraint.get_other_instance(instance) in instances
                ):
                    resulting_constraints.add(constraint)
        return resulting_constraints
