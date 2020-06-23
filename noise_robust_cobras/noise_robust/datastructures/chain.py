from collections import defaultdict
from typing import List

from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint
from noise_robust_cobras.noise_robust.datastructures.cycle import Cycle


class Chain:
    """
        A chain is a collection of constraints that can be linked together
        Is kept as a list of ordered constraints

        note: no eq or hash implementation!
    """

    def __init__(self, constraints, front=None, back=None, number_of_CLs=None):
        self.constraints = constraints
        if front is not None and back is not None:
            self.front = front
            self.back = back

        else:
            self.front, self.back = Chain.get_front_and_back(constraints)

        if self.front is None or self.back is None:
            raise Exception("illegal chain front or chain back")
        if number_of_CLs is None:
            self.number_of_CLS = sum(1 if con.is_CL() else 0 for con in constraints)
        else:
            self.number_of_CLS = number_of_CLs

    @staticmethod
    def chainify(constraints: List[Constraint]):
        all_chains = []
        # form constraints into chains
        for con in constraints:
            matching_chain = None
            for chain in all_chains:
                if chain.can_be_extended_with(con):
                    matching_chain = chain
                    break
            if matching_chain is not None:
                all_chains.remove(matching_chain)
                all_chains.append(matching_chain.extend(con))
            else:
                all_chains.append(Chain([con]))

        # check if chains can be added together
        changed_chains = True
        while changed_chains:
            changed_chains = False
            new_chains = []
            for chain in all_chains:
                matching_chain = None
                for new_chain in new_chains:
                    if new_chain.can_compose_with_chain(chain):
                        matching_chain = new_chain
                        break
                if matching_chain is not None:
                    changed_chains = True
                    new_chains.remove(matching_chain)
                    new_chains.append(matching_chain.compose_with_chain(chain))
                else:
                    new_chains.append(chain)
            all_chains = new_chains
        # print("all_chains ", [str(chain) for chain in all_chains])
        return all_chains

    @staticmethod
    def get_front_and_back(constraints):
        if len(constraints) == 1:
            front = constraints[0].i1
            back = constraints[0].i2
        else:
            count = defaultdict(lambda: 0)
            for constraint in constraints:
                count[constraint.i1] += 1
                count[constraint.i2] += 1
            occur_once = []
            for key, value in count.items():
                if value == 2:
                    continue
                elif value == 1:
                    occur_once.append(key)
                else:
                    raise Exception("invalid chain!")
            if len(occur_once) == 2:
                front, back = occur_once
            else:
                raise Exception("invalid chain!")
        return front, back

    def __len__(self):
        return len(self.constraints)

    def __contains__(self, constraint):
        return constraint in self.constraints

    def __iter__(self):
        return self.constraints.__iter__()

    def __str__(self):
        return "".join(str(constraint) for constraint in self.constraints)

    def to_cycle(self):
        if self.front != self.back:
            raise Exception("trying to make chain into invalid cycle!")
        return Cycle(self.constraints)

    def can_be_extended_with(self, con):
        return con.contains_instance(self.front) or con.contains_instance(self.back)

    def get_ordered_list(self):
        front = self.front
        chain = []
        constraints_left = set(self.constraints)
        while len(constraints_left) > 0:
            found_constraint = [
                con for con in constraints_left if con.contains_instance(front)
            ][0]
            chain.append(found_constraint)
            front = found_constraint.get_other_instance(front)
            constraints_left.remove(found_constraint)
        return chain

    def ordered_str(self):
        return "".join(str(con) for con in self.get_ordered_list())

    def goes_through_instance(self, instance):
        if len(self.constraints) < 2:
            return False
        return any(con.contains_instance(instance) for con in self.constraints[1:-1])

    def contains_cannot_link(self):
        return any(con.is_CL() for con in self.constraints)

    def can_compose_with_chain(self, other):
        return len({self.front, self.back}.intersection({other.front, other.back})) > 0

    def compose_with_chain(self, other):
        if not self.can_compose_with_chain(other):
            raise Exception(
                "cannot extend this chain {} with the given chain {}".format(
                    self, other
                )
            )
        return Chain(list(set(self.constraints).union(other.constraints)))

    def extend(self, con):
        if con.contains_instance(self.front):
            return self.extend_front(con)
        elif con.contains_instance(self.back):
            return self.extend_back(con)
        else:
            return None

    def extend_front(self, con):
        new_front = con.get_other_instance(self.front)
        new_CLs = (1 if con.is_CL() else 0) + self.number_of_CLS
        return Chain(
            [con] + self.constraints,
            front=new_front,
            back=self.back,
            number_of_CLs=new_CLs,
        )

    def extend_back(self, con):
        new_back = con.get_other_instance(self.back)
        new_CLs = (1 if con.is_CL() else 0) + self.number_of_CLS
        return Chain(
            self.constraints + [con],
            front=self.front,
            back=new_back,
            number_of_CLs=new_CLs,
        )
