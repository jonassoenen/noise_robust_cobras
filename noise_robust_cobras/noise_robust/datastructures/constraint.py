import functools
import json


def raw_constraints_to_constraints(raw_constraints):
    constraints = []
    for raw in raw_constraints:
        if isinstance(raw, Constraint):
            constraints.append(raw)
            continue
        if len(raw) == 2:
            i1, i2 = raw
            isML = True
        else:
            i1, i2, isML = raw
        constraints.append(Constraint(i1, i2, isML))
    return constraints


@functools.total_ordering
class Constraint:
    """
        A class that represents a single constraint
    """

    def __init__(
        self, i1, i2, isML, times_seen=1, times_other_seen=0, purpose="not specified"
    ):
        if i1 == i2:
            raise Exception("constraint between instance and itself")
        self.i1 = min(i1, i2)
        self.i2 = max(i1, i2)
        self.isML = isML
        # this can only be zero or 1 (used to be multiple)
        self.times_seen = times_seen
        self.times_other_seen = times_other_seen
        self.purpose = purpose

    def add_other_constraint(self, con):
        # can be removed
        if not con.get_instance_tuple() == self.get_instance_tuple():
            raise Exception("Trying to add constraint A to a constraint B when A != B ")

        if con.is_ML() == self.is_ML():
            self.times_seen += con.times_seen
            self.times_other_seen += con.times_other_seen
        else:
            self.times_seen += con.times_other_seen
            self.times_other_seen += con.times_seen

    def get_times_seen(self):
        return self.times_seen

    def get_times_other_seen(self):
        return self.times_other_seen

    def contains_instance(self, i):
        return i == self.i1 or i == self.i2

    def has_instance_in_common_with(self, other_constraint):
        return self.contains_instance(other_constraint.i1) or self.contains_instance(
            other_constraint.i2
        )

    def get_other_instance(self, i):
        if i == self.i2:
            return self.i1
        elif i == self.i1:
            return self.i2
        raise Exception(
            "get_other_instance with instance that is not part of the constraint"
        )

    def flip(self):
        return Constraint(
            self.i1,
            self.i2,
            not self.is_ML(),
            times_seen=self.times_other_seen,
            times_other_seen=self.times_seen,
            purpose=self.purpose,
        )

    def to_instance_set(self):
        return {self.i1, self.i2}

    def constraint_type(self):
        return self.isML

    def is_ML(self):
        return self.isML

    def is_CL(self):
        return not self.isML

    def get_instance_tuple(self):
        return (self.i1, self.i2)

    def to_tuple(self):
        return (self.i1, self.i2, self.isML)

    def __eq__(self, other):
        if other is None:
            return False
        return self.to_tuple() == other.to_tuple()

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        return hash((self.i1, self.i2, self.isML))

    def full_str(self):
        return (
            str(self)
            + " times_seen="
            + str(self.get_times_seen())
            + " times_other="
            + str(self.get_times_other_seen())
            + " for "
            + self.purpose
        )

    def __repr__(self):
        return self.__str__()  # + super().__repr__()

    def to_save_str(self):
        return json.dumps(
            (
                self.i1,
                self.i2,
                bool(self.is_ML()),
                self.times_seen,
                self.times_other_seen,
                self.purpose,
            )
        )

    @staticmethod
    def create_from_str(string):
        loaded = json.loads(string)
        if loaded is None:
            return None
        i1, i2, is_ML, times_seen, times_other_seen, purpose = loaded
        return Constraint(i1, i2, is_ML, times_seen, times_other_seen, purpose)

    def __str__(self):
        constraint_type = "ML" if self.is_ML() else "CL"
        return constraint_type + "(" + str(self.i1) + "," + str(self.i2) + ")"
