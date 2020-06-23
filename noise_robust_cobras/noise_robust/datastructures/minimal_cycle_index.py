import itertools
from collections import defaultdict

from noise_robust_cobras.noise_robust import find_cycles
from noise_robust_cobras.noise_robust.datastructures.cycle import Cycle


class CycleIndex:
    """
        Cycle index is a class that keeps track of a set of cycles
        Cycles are added through add_cycle_to_index and removed with remove_cycle

        attributes:
        - cycle-index a dictionary that maps a constraint to all cycles that involve this constraint
        - all consistent cycles: all cycles in this cycle index that are consistent (#CL's != 1)
        - all inconsistent cycles: all cycles in this cycle index that are inconsistent (#CL == 1)

        Specific subclasses are provided to keep track of specific classes of cycles
    """

    def __init__(self, constraint_index):
        self.constraint_index = constraint_index
        self.cycle_index = defaultdict(CycleIndex.set_tuple)
        self.all_consistent_cycles = set()
        self.all_inconsistent_cycles = set()

    def replace_constraint(self, old_constraint, new_constraint):
        all_cycles_with_constraint = self.get_all_cycles_for_constraint(old_constraint)
        new_cycles_with_constraint = [
            cycle.replace_constraint(old_constraint, new_constraint)
            for cycle in all_cycles_with_constraint
        ]
        for cycle_to_remove in all_cycles_with_constraint:
            self.remove_cycle(cycle_to_remove)
        for cycle in new_cycles_with_constraint:
            self.add_cycle_to_index(cycle)

    @staticmethod
    def set_tuple():
        return (set(), set())

    def is_inconsistent(self):
        return len(self.all_inconsistent_cycles) > 0

    def __contains__(self, item):
        return (
            item in self.all_consistent_cycles or item in self.all_inconsistent_cycles
        )

    def all_cycles(self):
        return self.all_inconsistent_cycles.union(self.all_consistent_cycles)

    def get_all_cycles_for_constraint(self, constraint):
        con_cycles, incon_cycles = self.cycle_index[constraint]
        return con_cycles.union(incon_cycles)

    def get_inconsistent_cycles_for_constraint(self, constraint):
        _, incon_cycles = self.cycle_index[constraint]
        return incon_cycles

    def get_consistent_cycles_for_constraint(self, constraint):
        con_cycles, _ = self.cycle_index[constraint]
        return con_cycles

    def add_cycle_to_index_entry(self, cycle, constraint):
        consistent_cycles, inconsistent_cycles = self.cycle_index[constraint]
        if cycle.is_inconsistent():
            inconsistent_cycles.add(cycle)
        else:
            consistent_cycles.add(cycle)

    def add_cycle_to_index(self, cycle):
        """
        - inconsistent cycles are added to all_inconsistent_cycles and the inconsistent_cycle_index
        - consistent cycles are added to all_cycles and the cycle_index
        """

        assert cycle
        # add cycle to cycle_index
        for constraint in cycle.constraints:
            self.add_cycle_to_index_entry(cycle, constraint)

        # add cycle to all_inconsistent_cycles or all_consistent_cycles
        if cycle.is_inconsistent():
            self.all_inconsistent_cycles.add(cycle)
        else:
            # the cycle is consistent
            self.all_consistent_cycles.add(cycle)

    def remove_cycle(self, cycle_to_remove):
        self.all_consistent_cycles.discard(cycle_to_remove)
        self.all_inconsistent_cycles.discard(cycle_to_remove)

        for con in cycle_to_remove:
            consistent, inconsistent = self.cycle_index[con]
            consistent.discard(cycle_to_remove)
            inconsistent.discard(cycle_to_remove)

    def remove_cycles_with_constraint(self, constraint_to_remove):
        con_cycles, incon_cycles = self.cycle_index[constraint_to_remove]
        self.all_consistent_cycles.difference_update(con_cycles)
        self.all_inconsistent_cycles.difference_update(incon_cycles)
        self.cycle_index.pop(constraint_to_remove)


class MinimalCycleIndex(CycleIndex):
    """
        Through add constraint keeps track of all the minimal cycles in the graph
        (for each constraint only the cycles are kept with the minimal length)

        note: old cycles that are not minimal are not removed from this datastructure

        constraints should be added through add_constraint  and removed through remove_cycles_with_constraint to ensure consistency of the data structure
    """

    def __init__(self, constraint_index):
        super().__init__(constraint_index)
        # minimal cycles dict is a dictionary from a constraint to a set of cycles
        # it keeps the cycles that need to be retained for this constraint
        self.minimal_cycles_dict = defaultdict(set)

    def add_constraint(self, constraint):
        all_cycles = find_cycles.find_all_cycles_with_minimal_length(
            self.constraint_index, constraint
        )
        if all_cycles is not None:
            self.add_minimal_cycles_to_index(all_cycles)

    def add_minimal_cycles_to_index(self, cycles):
        minimal_length = len(cycles[0])
        assert all(len(cycle) == minimal_length for cycle in cycles)
        # add the cycles to the index
        for cycle in cycles:
            self.add_cycle_to_index(cycle)

        minimal_cycle: Cycle = cycles[0]
        # remove longer cycles and add smaller minimal cycles
        # this does nothing!
        constraints_that_occur_in_short_cycle = minimal_cycle.constraints

        cycles_to_check = {
            cycle
            for con in constraints_that_occur_in_short_cycle
            for cycle in self.get_all_cycles_for_constraint(con)
            if len(cycle) > minimal_length
        }
        for old_cycle in cycles_to_check:
            composition = minimal_cycle.compose_with(old_cycle)
            if composition is not None and len(composition) < len(old_cycle):
                if composition not in self:
                    self.add_cycle_to_index(composition)

    def add_cycle_to_index(self, cycle):
        super(MinimalCycleIndex, self).add_cycle_to_index(cycle)
        self.add_cycle_to_minimal_cycle_dict(cycle)

    def check_cycles_for_removal(self, cycles):
        for cycle in cycles:
            if not self.is_minimal_cycle(cycle):
                self.remove_cycle(cycle)

    def add_cycle_to_minimal_cycle_dict(self, cycle):
        for constraint in cycle.constraints:
            existing_entry = self.minimal_cycles_dict[constraint]
            if len(existing_entry) == 0:
                self.minimal_cycles_dict[constraint].add(cycle)
            else:
                # you should keep the old cycle to ensure you have an inconsistent cycle
                some_cycle = list(existing_entry)[0]
                old_length = len(some_cycle)
                new_length = len(cycle)
                if new_length < old_length:
                    old_cycles = self.minimal_cycles_dict[constraint]
                    self.minimal_cycles_dict[constraint] = {cycle}
                    self.check_cycles_for_removal(old_cycles)
                elif new_length == old_length:
                    self.minimal_cycles_dict[constraint].add(cycle)
                else:
                    # new_length > old_length
                    pass

    def is_minimal_cycle(self, cycle):
        for constraint in cycle.constraints:
            if cycle in self.minimal_cycles_dict[constraint]:
                return True
        return False

    def remove_cycles_with_constraint(self, constraint_to_remove):
        involved_cycles = self.get_all_cycles_for_constraint(constraint_to_remove)
        new_cycles = []
        for cycle1, cycle2 in itertools.combinations(involved_cycles, 2):
            new_cycle = cycle1.compose_with(cycle2)
            if new_cycle is None:
                continue
            new_cycles.append(new_cycle)

        for cycle in involved_cycles:
            self.remove_cycle(cycle)

        for new_cycle in new_cycles:
            self.add_cycle_to_index(new_cycle)

        self.cycle_index.pop(constraint_to_remove)

    def remove_cycle(self, cycle_to_remove):
        super(MinimalCycleIndex, self).remove_cycle(cycle_to_remove)
        self.remove_cycle_from_minimal_cycle_dict(cycle_to_remove)

    def remove_cycle_from_minimal_cycle_dict(self, cycle_to_remove):
        for con in cycle_to_remove:
            entry = self.minimal_cycles_dict[con]
            entry.discard(cycle_to_remove)
