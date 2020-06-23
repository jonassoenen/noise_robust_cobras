from noise_robust_cobras.noise_robust.datastructures.chain import Chain


#####general algorithms######
def find_all_cycles_with_minimal_length(constraint_index, constraint):
    queue = [Chain([constraint])]
    visited = set()
    while not len(queue) == 0:
        next, queue = queue[0], queue[1:]
        front = next.front
        visited.add(front)
        front_extensions = [
            next.extend_front(constraint)
            for constraint in constraint_index.find_constraints_for_instance(front)
            if constraint not in next
            and constraint.get_other_instance(front) not in visited
        ]
        for new in front_extensions:
            if new.front == new.back:
                found_cycle = new.to_cycle()
                return find_other_cycles_with_same_length_as(
                    constraint_index, queue, visited, found_cycle
                )
        queue.extend(front_extensions)  # search breadth first
    return None


def find_other_cycles_with_same_length_as(constraint_index, queue, visited, cycle):
    """
        This function will check if there are other cycles with the same length as an already found cycle
        It utilizes the remaining queue after finding the given cycle
        This queue has to be constructed in a breath first fashion for this function to work correctly
        :param constraint_index:
        :param queue:
        :param visited:
        :param cycle:
        :return:
    """
    max_length = len(cycle)
    cycles = [cycle]
    for next in queue:
        if len(next) >= max_length:
            continue
        front = next.front
        visited.add(front)
        front_extensions = [
            next.extend_front(constraint)
            for constraint in constraint_index.find_constraints_for_instance(front)
            if constraint not in next
            and constraint.get_other_instance(front) not in visited
        ]
        for new in front_extensions:
            if new.front == new.back:
                cycles.append(new.to_cycle())
    return cycles


def find_cycle(constraint_index, constraint):
    queue = [Chain([constraint])]
    visited = set()
    while not len(queue) == 0:
        next, queue = queue[0], queue[1:]
        front = next.front
        visited.add(front)
        front_extensions = [
            next.extend_front(constraint)
            for constraint in constraint_index.find_constraints_for_instance(front)
            if constraint not in next
            and constraint.get_other_instance(front) not in visited
        ]
        new_filtered = list(filter(lambda x: x.number_of_CLS <= 2, front_extensions))
        for new in new_filtered:
            if new.front == new.back:
                return new.to_cycle()
        queue.extend(new_filtered)  # search breadth first
    return None
