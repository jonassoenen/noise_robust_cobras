import copy
from collections import defaultdict, deque
from enum import Enum

from noise_robust_cobras.noise_robust.datastructures.constraint_index import (
    ConstraintIndex,
)

"""
    inter-component cannot-links are cannot-links within a single component
    intra-component cannot-links are cannot-links between different components 
"""


class ConstraintValue(Enum):
    MUST_LINK = 1
    CANNOT_LINK = 2
    DONT_KNOW = 3


class MLComponent:
    """
        MLComponent is a set of instances that is connected by must-link constraints
        The ml-constraints between these instances are kept in self.ml_constraints
        The cl-constraints between these instances (these are inter-component cannot-links) are kept in self.inter_component_cls
    """

    def __init__(self, instances, ml_constraints, inter_component_cls=None):
        self.instances = set(instances)
        self.ml_constraints = set(ml_constraints)
        self.inter_component_cls = (
            [] if inter_component_cls is None else inter_component_cls
        )

    def is_fully_connected(self):
        if len(self.inter_component_cls) > 0:
            raise Exception(
                "Asking if component is fully connected while inconsistent!"
            )
        n = len(self.instances)
        return len(self.ml_constraints) == n * (n - 1) // 2

    def is_inconsistent(self):
        return len(self.inter_component_cls) > 0

    def add_ml_constraint(self, constraint):
        assert any(i in self for i in constraint.get_instance_tuple())
        for i in constraint.get_instance_tuple():
            self.instances.add(i)
        self.ml_constraints.add(constraint)

    def add_inter_component_cl(self, cl):
        assert all(i in self for i in cl.get_instance_tuple())
        self.inter_component_cls.append(cl)

    def remove_inter_component_cl(self, cl):
        self.inter_component_cls.remove(cl)

    def remove_ml_constraint(self, constraint):
        self.ml_constraints.remove(constraint)

    def __contains__(self, item):
        return item in self.instances

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        return self.instances.__iter__()

    def __deepcopy__(self, memodict={}):
        new_object = MLComponent(
            self.instances, self.ml_constraints, list(self.inter_component_cls)
        )
        return new_object


class ComponentQueryAnswerer:
    def __init__(self, instance_component_dict, intra_component_cls):
        self.instance_component_dict = instance_component_dict
        self.intra_component_cls = intra_component_cls

    def get_component_of_instance(self, instance):
        return self.instance_component_dict[instance]

    def does_intra_component_cls_exist_between(self, comp1, comp2):
        min_comp, max_comp = min(comp1, comp2), max(comp1, comp2)
        return (min_comp, max_comp) in self.intra_component_cls

    def get_constraint_value_between_instances(self, i1, i2):
        comp1 = self.get_component_of_instance(i1)
        comp2 = self.get_component_of_instance(i2)
        if comp1 == comp2:
            return ConstraintValue.MUST_LINK
        elif self.does_intra_component_cls_exist_between(comp1, comp2):
            return ConstraintValue.CANNOT_LINK
        else:
            return ConstraintValue.DONT_KNOW


class ComponentTracker:
    """
        A datastructure that keeps track of ml-components
        A ml-component is a set of instances that is connected by must-link constraints

        cannot-links between different components are called intra-component cannot-links
        cannot-links within a single component are called inter-component cannot-links
        These 2 kinds of cannot-links are also tracked

        note: that there are 2 ways constraints can be flipped
            1) the constraint is added to flipped_constraints and then the flip_constraint method is called
            2) the constraint is flipped in the constraint index and then the flip_constraint method is called

        This means that when there is a constraint con in flipped_constraints:
            - con is in the constraint_index
            - but the flipped version of con is used in the componentTracker!

        So when using the constraint_index you have to take the constraints in flipped_constraints into account

    """

    def __init__(self, constraint_index: ConstraintIndex, flipped_constraints=None):
        self.constraint_index = constraint_index
        self.flipped_constraints = (
            set(flipped_constraints) if flipped_constraints is not None else set()
        )

        # keep track of all the components
        self.component_dict = dict()
        self.components = []

        # a dictionary from a pair of components
        self.intra_component_cls = defaultdict(set)

    def print_status(self):
        print("flipped_constraints:", self.flipped_constraints)
        print("components")
        for idx, component in enumerate(self.components):
            print(
                "\t",
                idx,
                ")",
                "instances:",
                component.instances,
                component.ml_constraints,
                component.inter_component_cls,
            )
        print("intra-component cls")
        for (component1, component2), value in self.intra_component_cls.items():
            comp_idx1 = self.components.index(component1)
            comp_idx2 = self.components.index(component2)
            print(comp_idx1, comp_idx2, "->", value)

    def flip_additional_constraint(self, extra_constraint):
        # print("new flipped ", str(extra_constraint))
        # print("BEFORE")
        # self.print_status()
        self.flipped_constraints.add(extra_constraint)
        # print("removed {} added {}".format(extra_constraint, extra_constraint.flip()))
        self.flip_constraint(extra_constraint, extra_constraint.flip())
        # print("AFTER")
        # self.print_status()

    def undo_flip_additional_constraint(self, extra_constraint):
        # print("remove flipped ", str(extra_constraint))
        # print("BEFORE")
        # self.print_status()
        self.flipped_constraints.remove(extra_constraint)
        self.flip_constraint(extra_constraint.flip(), extra_constraint)
        # print("AFTER")
        # self.print_status()

    def copy_component_tracker_with_extra_flipped_constraints(self, extra_flipped):
        """
            This is used to incrementally update the component tracker whenever there is an extra flipped constraint
        :param extra_flipped:
        :return:
        """
        new_component_tracker = copy.deepcopy(self)
        for constraint in extra_flipped:
            new_component_tracker.flipped_constraints.add(constraint)
            new_component_tracker.flip_constraint(constraint, constraint.flip())
        return new_component_tracker

    def get_component_query_answerer(self):
        component_lookup = dict()
        for idx, component in enumerate(self.components):
            component_lookup[component] = idx

        new_component_dict = dict()
        for key, value in self.component_dict.items():
            new_component_dict[key] = component_lookup[value]

        new_intra_component_cls = set()
        for key, value in self.intra_component_cls.items():
            if len(value) > 0:
                comp1, comp2 = key
                comp1_int, comp2_int = component_lookup[comp1], component_lookup[comp2]
                min_comp, max_comp = (
                    min(comp1_int, comp2_int),
                    max(comp1_int, comp2_int),
                )
                new_intra_component_cls.add((min_comp, max_comp))

        return ComponentQueryAnswerer(new_component_dict, new_intra_component_cls)

    def __getitem__(self, item):
        return self.get_component_of_instance(item)

    # provided functionality for algorithms

    def is_inconsistent(self):
        """
        Checks if the current constraint set is inconsistent based on if there is an inter component cannot link in one
        of the ml componentes

        :return: whether or not the current constraint set is inconsistent
        :rtype: bool
        """
        # TODO could be optimized by extra bookkeeping
        return any(component.is_inconsistent() for component in self.components)

    def get_constraint_value_between_instances(self, i1, i2):
        comp1 = self.get_component_of_instance(i1)
        comp2 = self.get_component_of_instance(i2)
        if comp1 == comp2:
            return ConstraintValue.MUST_LINK
        elif self.does_intra_component_cls_exist_between(comp1, comp2):
            return ConstraintValue.CANNOT_LINK
        else:
            return ConstraintValue.DONT_KNOW

    #  flip constraints

    def flip_constraint(self, old_constraint, new_constraint):
        """
            Updates this componenttracker incrementally such that the constraintvalue of old_constraint is flipped
        :param old_constraint:
        :param new_constraint:
        :return:
        """
        assert old_constraint.is_ML() != new_constraint.is_ML()
        if old_constraint.is_ML():
            self.__flip_ml_to_cl(old_constraint, new_constraint)
        else:
            self.__flip_cl_to_ml(old_constraint, new_constraint)

    def __flip_ml_to_cl(self, ml, new_cl):
        """
            O(#ml constraints in component of [ml] + #cl constraints that is involved with component of [ml])
        :param ml:
        :param new_cl:
        :return:
        """
        # this ml was already in the component_structure so we expect the instances to be part of just one component
        matching_components = self.get_matching_components_for_constraint(ml)
        # print("matching_components", [self.components.index(component) for component in matching_components])
        assert len(matching_components) == 1
        matching_component = matching_components[0]
        matching_component.remove_ml_constraint(ml)
        # suppose ML(x,y) needs to be changed to CL(x,y)
        # then we need to check whether the instances x and y are still connected
        (
            instances_connected_to_i1,
            constraints_connected_to_i1,
        ) = self.get_constraints_connected_to_instance_by_mls(ml.i1)
        # print("instances connected:", instances_connected_to_i1)
        # print("constraints connected:", constraints_connected_to_i1)
        if ml.i2 in instances_connected_to_i1:
            # print("still connected")
            # the 2 instances are still connected so the component structure does not really change
            # only there is a new cannot-link in the matching_component
            matching_component.add_inter_component_cl(new_cl)
        else:
            # print("not connected")
            # the 2 instances are not connected anymore
            # this means that the matching_component is now split in 2 smaller components
            component1_instances = instances_connected_to_i1
            component2_instances = matching_component.instances.difference(
                instances_connected_to_i1
            )
            component1_constraints = constraints_connected_to_i1
            component2_constraints = matching_component.ml_constraints.difference(
                constraints_connected_to_i1
            )

            # the old cls_within_component need to be assigned correctly to either:
            # cls within component1, cls within component2 or cls between component1 and component 2
            cls_within_old_component = matching_component.inter_component_cls
            cls_within_component1 = []
            cls_within_component2 = []
            cls_between_new_components = []
            for cl_to_classify in cls_within_old_component:
                if all(
                    i in component1_instances
                    for i in cl_to_classify.get_instance_tuple()
                ):
                    cls_within_component1.append(cl_to_classify)
                elif all(
                    i in component2_instances
                    for i in cl_to_classify.get_instance_tuple()
                ):
                    cls_within_component2.append(cl_to_classify)
                else:
                    cls_between_new_components.append(cl_to_classify)

            cls_between_new_components.append(new_cl)

            # make the new components
            new_component1 = MLComponent(
                component1_instances, component1_constraints, cls_within_component1
            )
            new_component2 = MLComponent(
                component2_instances, component2_constraints, cls_within_component2
            )

            # retrieve the intra component cls that need to be reclassified
            intra_component_cls_for_old_component = self.get_intra_component_cls_for_component(
                matching_component
            )

            # add new components and remove old component
            self.remove_component(matching_component)
            self.add_new_component(new_component1)
            self.add_new_component(new_component2)

            # the cls between the matching component and another component needs to be reassigned as well
            # list of tuples (cls, (component_pair))
            new_intra_cls = []
            for cls, (comp1, comp2) in intra_component_cls_for_old_component:
                other_component = comp1 if comp1 != matching_component else comp2
                for cl in cls:
                    i1, i2 = cl.get_instance_tuple()
                    if i1 in new_component1 or i2 in new_component1:
                        new_intra_cls.append((cl, other_component, new_component1))
                    elif i1 in new_component2 or i2 in new_component2:
                        new_intra_cls.append((cl, other_component, new_component2))

            # add all of the intra component cannot-links
            for cl, comp1, comp2 in new_intra_cls:
                self.add_intra_component_cl(comp1, comp2, cl)
            for cl in cls_between_new_components:
                self.add_intra_component_cl(new_component1, new_component2, cl)

    def __flip_cl_to_ml(self, cl, new_ml):
        # there are 2 possibilities
        # or the cl was a inter component cl
        # or the cl is an intra component cl

        matching_components = self.get_matching_components_for_constraint(cl)
        # print("matching_components", [self.components.index(component) for component in matching_components])
        if len(matching_components) == 1:
            # print("------- inter-component cl")
            # it is an intercomponent cl
            # just remove the cannot-link as an inter component cl
            matching_component = matching_components[0]
            matching_component.remove_inter_component_cl(cl)
            matching_component.add_ml_constraint(new_ml)
        elif len(matching_components) == 2:
            # print("--------- intra-component cl")
            # it is an intra component cl
            component1, component2 = matching_components
            self.remove_intra_component_cl_between_components(
                component1, component2, cl
            )
            self.__merge_components(component1, component2, new_ml)

    #  add constraints

    def add_constraint(self, constraint):
        if constraint.is_ML():
            self.__add_ml_constraint(constraint)
        else:
            self.__add_cl_constraint(constraint)

    def __add_ml_constraint(self, constraint):
        matching_components = self.get_matching_components_for_constraint(constraint)
        if len(matching_components) == 0:
            new_component = MLComponent(constraint.get_instance_tuple(), (constraint,))
            self.add_new_component(new_component)
        elif len(matching_components) == 1:
            matching_component = matching_components[0]
            self.add_constraint_to_component(constraint, matching_component)
        elif len(matching_components) == 2:
            component1, component2 = matching_components
            self.__merge_components(component1, component2, constraint)
        else:
            raise Exception(
                "{} matching components for constraint {} valid values are 0,1,2".format(
                    len(matching_components), constraint
                )
            )

    def __add_cl_constraint(self, cl):
        matching_components = self.get_matching_components_for_constraint(cl)

        if len(matching_components) == 0:
            # make 2 new components with only one instance and a cannot-link between them
            new1 = MLComponent((cl.i1,), ())
            new2 = MLComponent((cl.i2,), ())
            self.add_new_component(new1)
            self.add_new_component(new2)
            self.add_intra_component_cl(new1, new2, cl)

        elif len(matching_components) == 1:
            matching_component = matching_components[0]
            if all(
                instance in matching_component for instance in cl.get_instance_tuple()
            ):
                # both instances are in the matching_component
                matching_component.add_inter_component_cl(cl)
            else:
                # only one of the instances is in the matching_component
                matching_instance = cl.i1 if cl.i1 in matching_component else cl.i2
                other_instance = cl.get_other_instance(matching_instance)
                # make a new component for the non matching instance
                new_component = MLComponent((other_instance,), ())
                self.add_new_component(new_component)
                self.add_intra_component_cl(new_component, matching_component, cl)

        elif len(matching_components) == 2:
            self.add_intra_component_cl(
                matching_components[0], matching_components[1], cl
            )
        else:
            raise Exception(
                "{} matching components for constraint {} valid values are 0,1,2".format(
                    len(matching_components), cl
                )
            )

    #  higher level operations

    def __merge_components(self, component1, component2, constraint):
        new_instances = component1.instances.union(component2.instances).union(
            constraint.get_instance_tuple()
        )
        new_ml_constraints = component1.ml_constraints.union(
            component2.ml_constraints
        ).union((constraint,))
        cls_between_components = self.get_intra_component_cls_between(
            component1, component2
        )
        cls_within_new_component = (
            list(cls_between_components)
            + component1.inter_component_cls
            + component2.inter_component_cls
        )
        new_component = MLComponent(
            new_instances, new_ml_constraints, cls_within_new_component
        )
        self.remove_intra_component_cls_between(component1, component2)

        # the intra component cls between one of the old components and another component
        # become intra component cls between the new component and the other component
        for component in [component1, component2]:
            intra_component_cls = self.get_intra_component_cls_for_component(component)
            for cls, (comp1, comp2) in intra_component_cls:
                other_component = comp1 if comp1 != component else comp2
                for cl in cls:
                    self.add_intra_component_cl(new_component, other_component, cl)

        # remove old components
        self.remove_component(component1)
        self.remove_component(component2)

        # add new components
        self.add_new_component(new_component)

    def get_constraints_connected_to_instance_by_mls(self, x):
        """
        O(#constraints in ml component)
        :param x:
        :return:
        """
        all_connected_instances = {x}
        connected_constraints = set()
        queue = deque([x])
        while len(queue) != 0:
            current = queue.popleft()
            ml_constraints = [
                con
                for con in self.constraint_index.find_constraints_for_instance(current)
                if self.__is_constraint_ml_given_flipped(con)
            ]
            connected_constraints.update(
                con if con not in self.flipped_constraints else con.flip()
                for con in ml_constraints
            )
            connected_instances = {
                con.get_other_instance(current) for con in ml_constraints
            }
            new_connected_instances = connected_instances.difference(
                all_connected_instances
            )
            all_connected_instances.update(connected_instances)

            queue.extend(new_connected_instances)
        return all_connected_instances, connected_constraints

    def __get_instances_connected_to_instance_by_mls(self, x):
        all_connected_instances = {x}
        queue = [x]
        while len(queue) != 0:
            current, queue = queue[0], queue[1:]
            ml_constraints = [
                con
                for con in self.constraint_index.find_constraints_for_instance(current)
                if self.__is_constraint_ml_given_flipped(con)
            ]
            connected_instances = {
                con.get_other_instance(current) for con in ml_constraints
            }
            new_connected_instances = connected_instances.difference(
                all_connected_instances
            )
            all_connected_instances.update(connected_instances)

            queue.extend(new_connected_instances)
        return all_connected_instances

    def __is_constraint_ml_given_flipped(self, con):
        original_type = con.is_ML()
        if con in self.flipped_constraints:
            original_type = not original_type
        return original_type

    #  intra component cls

    def does_intra_component_cls_exist_between(self, component1, component2):
        return len(self.get_intra_component_cls_between(component1, component2)) > 0

    def get_intra_component_cls_for_component(self, component):
        cls_list = []
        for (comp1, comp2), cls in self.intra_component_cls.items():
            if comp1 == component or comp2 == component:
                cls_list.append((cls, (comp1, comp2)))
        return cls_list

    def get_intra_component_cls_between(self, component1, component2):
        if (component1, component2) in self.intra_component_cls:
            return self.intra_component_cls[(component1, component2)]
        else:
            return self.intra_component_cls[(component2, component1)]

    def remove_intra_component_cl_between_components(self, component1, component2, cl):
        if (component1, component2) in self.intra_component_cls:
            self.intra_component_cls[(component1, component2)].remove(cl)
        else:
            self.intra_component_cls[(component2, component1)].remove(cl)

    def remove_intra_component_cls_with_component(self, component):
        keys_to_remove = []
        for comp1, comp2 in self.intra_component_cls.keys():
            if component == comp1 or component == comp2:
                keys_to_remove.append((comp1, comp2))
        for key in keys_to_remove:
            self.intra_component_cls.pop(key)

    def remove_intra_component_cls_between(self, component1, component2):
        self.intra_component_cls.pop((component1, component2), None)
        self.intra_component_cls.pop((component2, component1), None)

    def add_intra_component_cl(self, component1, component2, cl):
        if (component1, component2) in self.intra_component_cls:
            self.intra_component_cls[(component1, component2)].add(cl)
        else:
            self.intra_component_cls[(component2, component1)].add(cl)

    #  component dict and components

    def get_component_of_instance(self, instance):
        """

        :param instance:
        :return: the matching component
        :rtype: MLComponent
        """
        return self.component_dict.get(instance, None)

    def get_matching_components_for_constraint(self, constraint):
        components = set()
        for i in constraint.get_instance_tuple():
            component = self.get_component_of_instance(i)
            if component is not None:
                components.add(component)
        return list(components)

    def add_new_component(self, component):
        self.components.append(component)
        for instance in component:
            self.component_dict[instance] = component

    def add_constraint_to_component(self, constraint, component):
        component.add_ml_constraint(constraint)
        for i in constraint.get_instance_tuple():
            self.component_dict[i] = component

    def remove_component(self, component):
        self.components.remove(component)
        self.remove_intra_component_cls_with_component(component)
