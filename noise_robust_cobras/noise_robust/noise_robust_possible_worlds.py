import itertools

from noise_robust_cobras.cobras_logger import ClusteringLogger
from noise_robust_cobras.noise_robust.datastructures.certainty_constraint_set import (
    NewCertaintyConstraintSet,
)
from noise_robust_cobras.noise_robust.datastructures.componenttracker import (
    ConstraintValue,
)


def gather_extra_evidence(
    certainty_constraint_set: NewCertaintyConstraintSet,
    relevant_constraints,
    certainty_threshold,
    querier,
    logger: ClusteringLogger,
):
    """
        Ask redundant constraints until the confidence in the corrected relevant constraints is > certainty_threshold
        The query selection strategy used evaluates all possible constraint candidates based on the possible worlds and asks the candidate that has the highest impact

        The constraints are corrected in place after the execution of this method certainty_constraint_set contains the corrected constraints

        :param relevant_constraints: the constraints that are relevant and thus need to be corrected with reasonable confidence
        :return: a boolean indicating whether or not there was noise detected
    """
    logger.log_entering_phase("evidence")
    noise_detected = False

    # if the constraints are inconsistent launch the find noise phase
    if certainty_constraint_set.is_inconsistent():
        detected_noise = find_and_correct_noise(
            certainty_constraint_set, relevant_constraints, querier, logger
        )
        logger.log_entering_phase("evidence")
        detected_noise_dict = dict(detected_noise)
        relevant_constraints = [
            detected_noise_dict[con] if con in detected_noise_dict else con
            for con in relevant_constraints
        ]
        noise_detected = True

    # update the certainty of the relevant constraints
    current_certainty = certainty_constraint_set.get_probability_of_constraints_ignoring_edge_cases(
        relevant_constraints
    )

    # while the certainty of the relevant constraints is lower than certainty_threshold ask redundant queries
    while current_certainty < certainty_threshold:
        # select the best query to pose
        best_query = best_query_to_pose_gather_extra_evidence(
            certainty_constraint_set, relevant_constraints
        )

        # query the constraint
        new_constraint = querier.query(*best_query, "gather extra evidence")
        certainty_constraint_set.add_constraint(new_constraint)

        # if the constraints are inconsistent launch the find noise phase
        if certainty_constraint_set.is_inconsistent():
            detected_noise = find_and_correct_noise(
                certainty_constraint_set, relevant_constraints, querier, logger
            )
            logger.log_entering_phase("evidence")
            detected_noise_dict = dict(detected_noise)
            relevant_constraints = [
                detected_noise_dict[con] if con in detected_noise_dict else con
                for con in relevant_constraints
            ]
            noise_detected = True

        # update the certainty of the relevant constraints
        current_certainty = certainty_constraint_set.get_probability_of_constraints_ignoring_edge_cases(
            relevant_constraints
        )

    return noise_detected


def find_and_correct_noise(
    certainty_constraint_set, relevant_constraints, querier, logger
):
    """
        Given an inconsistent certainty_constraint_set this function will find and correct constraints such that the
        certainty_constraint_set becomes inconsistent again.

        :return:
        :note:  can optimize running over the noise_list twice at the beginning and the end of the while loop
    """
    logger.log_entering_phase("correction")
    # some utility functions
    nb_wrong = certainty_constraint_set.constraint_index.nb_wrong_constraints

    def get_nb_of_wrong_constraints(flipped_constraints):
        times_correct_but_counted_as_wrong = sum(
            con.get_times_other_seen() for con in flipped_constraints
        )
        times_wrong_but_counted_as_correct = sum(
            con.get_times_seen() for con in flipped_constraints
        )
        total_wrong = (
            nb_wrong
            - times_correct_but_counted_as_wrong
            + times_wrong_but_counted_as_correct
        )
        return total_wrong

    def get_flipped_relevant_tuples(noise_list):
        all_relevant_flipped_tuples = set()
        for flipped, component_tracker in noise_list:
            flipped_relevant = tuple(
                sorted(set(flipped).intersection(relevant_constraints))
            )
            all_relevant_flipped_tuples.add(flipped_relevant)
        return all_relevant_flipped_tuples

    # retrieve the current noise_list
    noise_list = certainty_constraint_set.get_noise_list()

    # calculate the most likely assignments to the relevant constraints
    minimum_length_noise_item = min(
        get_nb_of_wrong_constraints(x) for x, _ in noise_list
    )
    most_likely_options = [
        (x, y)
        for x, y in noise_list
        if get_nb_of_wrong_constraints(x) == minimum_length_noise_item
    ]
    flipped_relevant_tuples = get_flipped_relevant_tuples(most_likely_options)

    # while there are multiple most likely assignments ask a redundant constraints to reduce these to a single most likely one
    while len(flipped_relevant_tuples) > 1:

        # select the best query to pose
        query = best_query_to_pose_find_and_correct_noise(
            certainty_constraint_set, relevant_constraints, len(flipped_relevant_tuples)
        )
        # if there is no query possible anymore
        if query is None:
            break

        # pose the query
        new_constraint = querier.query(*query, "find noise")
        certainty_constraint_set.add_constraint(new_constraint)

        # get the updated noise list
        noise_list = certainty_constraint_set.get_noise_list()

        # calculate the most likely assignments to the relevant constraints
        minimum_length_noise_item = min(
            get_nb_of_wrong_constraints(x) for x, _ in noise_list
        )
        most_likely_options = [
            (x, y)
            for x, y in noise_list
            if get_nb_of_wrong_constraints(x) == minimum_length_noise_item
        ]
        flipped_relevant_tuples = get_flipped_relevant_tuples(most_likely_options)

    # the most likely consistent assignments all have the same value for the relevant constraints
    # OR there are no useful constraints left to query!
    # in both these cases just take one of the most likely assignments as the currently correct one
    most_likely_option_flipped, most_likely_component_tracker = most_likely_options[0]
    detected_noise = []
    for constraint_to_flip in most_likely_option_flipped:
        flipped_constraint = certainty_constraint_set.flip_constraint(
            constraint_to_flip
        )
        detected_noise.append((constraint_to_flip, flipped_constraint))

    logger.log_detected_noisy_constraints([x for x, _ in detected_noise])

    return detected_noise


def best_query_to_pose_gather_extra_evidence(
    certainty_constraint_set, relevant_constraints
):
    """
        For each candidate this method counts the number of lowest order possible worlds that are eliminated
        ties are broken by also taking the higher order constraints into account

    :return:
    """
    # get the current noise list
    noise_list = certainty_constraint_set.get_noise_list()

    # get lowest order possible worlds where the relevant constraints are not satisfied
    noise_list_relevant_not_satisfied = [
        (flipped, component_tracker)
        for flipped, component_tracker in noise_list
        if len(set(flipped).intersection(relevant_constraints)) != 0
    ]
    lowest_order = min(
        certainty_constraint_set.get_number_of_wrong_constraints_given_flipped_constraints(
            x
        )
        for x, _ in noise_list_relevant_not_satisfied
    )
    lowest_order_noise_list = [
        (flipped, component_tracker)
        for flipped, component_tracker in noise_list_relevant_not_satisfied
        if certainty_constraint_set.get_number_of_wrong_constraints_given_flipped_constraints(
            flipped
        )
        == lowest_order
    ]

    # generate all query candidates that are not known already
    all_instances = (
        certainty_constraint_set.constraint_index.get_all_instances_with_constraint()
    )
    all_query_candidates = {
        (i1, i2)
        for i1, i2 in itertools.combinations(all_instances, 2)
        if not certainty_constraint_set.constraint_index.does_constraint_between_instances_exist(
            i1, i2
        )
    }

    # run over all the candidates and select the ones with the highest score
    best_candidates = []
    best_score = None
    for candidate in all_query_candidates:
        # try to eliminate as much of the lowest_order_noise_list possible worlds
        # assuming that the current most likely world is correct
        answer_in_current = certainty_constraint_set.component_tracker.get_constraint_value_between_instances(
            *candidate
        )
        if answer_in_current == ConstraintValue.DONT_KNOW:
            # if the answer is don't know in the current assignment
            # this query will not tell us anything usefull
            continue
        (
            ml_count,
            dk_count,
            cl_count,
        ) = get_weight_sum_per_query_answer_for_all_possible_worlds(
            certainty_constraint_set, lowest_order_noise_list, candidate
        )
        if answer_in_current == ConstraintValue.MUST_LINK:
            eliminated_weight = cl_count
        elif answer_in_current == ConstraintValue.CANNOT_LINK:
            eliminated_weight = ml_count
        else:
            raise Exception("Unkown constraint value {}".format(answer_in_current))
        if best_score is None or eliminated_weight >= best_score:
            if eliminated_weight == best_score:
                best_candidates.append(candidate)
            else:
                best_candidates = [candidate]
                best_score = eliminated_weight

    # break the tie between the best candidates by also evaluating higher order noise items
    best_candidates2 = []
    best_score = None
    for candidate in best_candidates:
        answer_in_current = certainty_constraint_set.component_tracker.get_constraint_value_between_instances(
            *candidate
        )
        if answer_in_current == ConstraintValue.DONT_KNOW:
            continue
        (
            ml_count,
            dk_count,
            cl_count,
        ) = get_weight_sum_per_query_answer_for_all_possible_worlds(
            certainty_constraint_set, noise_list_relevant_not_satisfied, candidate
        )
        if answer_in_current == ConstraintValue.MUST_LINK:
            eliminated_weight = cl_count
        elif answer_in_current == ConstraintValue.CANNOT_LINK:
            eliminated_weight = ml_count
        else:
            raise Exception("Unkown constraint value {}".format(answer_in_current))
        if best_score is None or eliminated_weight >= best_score:
            if eliminated_weight == best_score:
                best_candidates2.append(candidate)
            else:
                best_candidates2 = [candidate]
                best_score = eliminated_weight

    return best_candidates2[0]


def best_query_to_pose_find_and_correct_noise(
    certainty_constraint_set, relevant_constraints, nb_current_flipped_relevants
):
    """
    Finds the best query to pose when we are trying to find and correct noisy constraints
    A query is asked such that approximately half of the most likely assignments implies this is a must-link
     while the other half implies it is a cannot-link

    :param certainty_constraint_set:
    :param relevant_constraints:
    :param nb_current_flipped_relevants:
    :return:
    """
    # get the current noise list
    noise_list = certainty_constraint_set.get_noise_list()

    # gets the lowest order noise list items
    lowest_order = min(
        certainty_constraint_set.get_number_of_wrong_constraints_given_flipped_constraints(
            x
        )
        for x, _ in noise_list
    )
    lowest_order_noise_list = [
        (flipped, component_query_answerer)
        for flipped, component_query_answerer in noise_list
        if certainty_constraint_set.get_number_of_wrong_constraints_given_flipped_constraints(
            flipped
        )
        == lowest_order
    ]
    # note: I don't have to remove the current assignment because it is inconsistent

    # construct all the query candidates
    all_instances = (
        certainty_constraint_set.constraint_index.get_all_instances_with_constraint()
    )
    all_query_candidates = {
        (i1, i2)
        for i1, i2 in itertools.combinations(all_instances, 2)
        if not certainty_constraint_set.constraint_index.does_constraint_between_instances_exist(
            i1, i2
        )
    }

    # evaluate each candidate and keep the candidates with the best score
    best_candidates = []
    best_score = None
    for candidate in all_query_candidates:
        (
            ml_count,
            ml_flipped_relevants,
            dk_count,
            dk_flipped_constraints,
            cl_count,
            cl_flipped_relevants,
        ) = get_weight_sum_per_query_answer_and_flipped_relevant_constraints_for_all_possible_worlds(
            lowest_order_noise_list,
            certainty_constraint_set,
            candidate,
            relevant_constraints,
        )
        if (
            len(ml_flipped_relevants) == nb_current_flipped_relevants
            and len(cl_flipped_relevants) == nb_current_flipped_relevants
        ):
            # this query is not relevant! It does not restrict the assignments to the relevant constraints
            continue
        # it could be that some of the possible worlds return don't know
        # take this into account!
        if ml_count == 0 and cl_count == 0:
            # ignore constraints that are don't know in all cases
            continue
        difference_between_scores = abs(ml_count - cl_count)
        if best_score is None or difference_between_scores <= best_score:
            if difference_between_scores == best_score:
                best_candidates.append(candidate)
            else:
                best_candidates = [candidate]
                best_score = difference_between_scores

    # from the best candidates select the best one based on higher orders
    best_candidates2 = []
    best_score = None
    for candidate in best_candidates:
        (
            ml_count,
            dk_count,
            cl_count,
        ) = get_weight_sum_per_query_answer_for_all_possible_worlds(
            certainty_constraint_set, noise_list, candidate
        )
        # it could be that some of the possible worlds return don't know
        # take this into account!
        difference_between_scores = abs(ml_count - cl_count)
        if best_score is None or difference_between_scores <= best_score:
            if difference_between_scores == best_score:
                best_candidates2.append(candidate)
            else:
                best_candidates2 = [candidate]
                best_score = difference_between_scores

    if len(best_candidates2) != 0:
        return best_candidates2[0]
    # there is no good constraint left!
    return None


def get_weight_sum_per_query_answer_and_flipped_relevant_constraints_for_all_possible_worlds(
    noise_list, certainty_constraint_set, query, relevant_constraints
):
    """

            :param noise_list:
            :param query:
            :return:
    """
    ml_weight_sum, ml_flipped_relevants = 0, set()
    cl_weight_sum, cl_flipped_relevants = 0, set()
    dk_weight_sum, dk_flipped_relevants = 0, set()
    i1, i2 = query
    for flipped_constraints, component_query_answerer in noise_list:
        query_answer = component_query_answerer.get_constraint_value_between_instances(
            i1, i2
        )
        assignment_weight = get_weight_of_noisy_consistent_assignment(
            flipped_constraints, certainty_constraint_set
        )
        relevant_flipped_constraints = tuple(
            sorted(set(flipped_constraints).intersection(relevant_constraints))
        )
        if query_answer == ConstraintValue.MUST_LINK:
            ml_weight_sum += assignment_weight
            ml_flipped_relevants.add(relevant_flipped_constraints)
        elif query_answer == ConstraintValue.CANNOT_LINK:
            cl_weight_sum += assignment_weight
            cl_flipped_relevants.add(relevant_flipped_constraints)
        elif query_answer == ConstraintValue.DONT_KNOW:
            dk_weight_sum += assignment_weight
            dk_flipped_relevants.add(relevant_flipped_constraints)
        else:
            raise Exception("unknown ConstraintValue: " + str(query_answer))

    return (
        ml_weight_sum,
        ml_flipped_relevants,
        dk_weight_sum,
        dk_flipped_relevants,
        cl_weight_sum,
        cl_flipped_relevants,
    )


def get_weight_sum_per_query_answer_for_all_possible_worlds(
    certainty_constraint_set: NewCertaintyConstraintSet, noise_list_subset, query
):
    """

    :param noise_list:
    :param query:
    :return:
    """
    ml_weight_sum = 0
    cl_weight_sum = 0
    dk_weight_sum = 0
    i1, i2 = query
    for flipped_constraints, component_tracker in noise_list_subset:
        query_answer = component_tracker.get_constraint_value_between_instances(i1, i2)
        if query_answer == ConstraintValue.MUST_LINK:
            ml_weight_sum += get_weight_of_noisy_consistent_assignment(
                flipped_constraints, certainty_constraint_set
            )
        elif query_answer == ConstraintValue.CANNOT_LINK:
            cl_weight_sum += get_weight_of_noisy_consistent_assignment(
                flipped_constraints, certainty_constraint_set
            )
        elif query_answer == ConstraintValue.DONT_KNOW:
            dk_weight_sum += get_weight_of_noisy_consistent_assignment(
                flipped_constraints, certainty_constraint_set
            )
        else:
            raise Exception("unknown ConstraintValue: " + str(query_answer))

    return ml_weight_sum, dk_weight_sum, cl_weight_sum


def get_weight_of_noisy_consistent_assignment(
    flipped_constraints, certainty_constraint_set: NewCertaintyConstraintSet
):
    """
        Gets the weight of a noisy consistent assignment P(R|U)
    :param flipped_constraints:
    :param certainty_constraint_set:
    :return:
    """
    noise_prob = certainty_constraint_set.noise_probability
    correct_u = 1 - noise_prob
    wrong_u = noise_prob

    nb_correct_user_constraints = sum(
        constraint.get_times_seen()
        for constraint in certainty_constraint_set.constraint_index
    )
    nb_wrong_user_constraints = sum(
        constraint.get_times_other_seen()
        for constraint in certainty_constraint_set.constraint_index
    )
    times_correct_but_counted_as_wrong = sum(
        con.get_times_other_seen() for con in flipped_constraints
    )
    times_wrong_but_counted_as_correct = sum(
        con.get_times_seen() for con in flipped_constraints
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
