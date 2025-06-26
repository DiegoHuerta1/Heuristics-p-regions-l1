from igraph import Graph
from numpy.typing import NDArray
import numpy as np
import time
import random

from Heuristics.utils.utils import compute_distance_matrix, obj_f_with_dist_matrix
from Heuristics.utils.utils import get_frontier_set, get_movable_nodes_set
from Heuristics.utils.utils import get_possible_moves, obj_f_move
from Heuristics.utils.utils import make_move, update_frontier, update_movable_nodes

def local_search_from_partition(graph: Graph, P: dict,
                                dist_matrix: None | NDArray = None,
                                max_time: float = np.inf) -> dict:
    """Run Local search from a given partition

    Args:
        graph (Graph): Graph instance
        P (dict): Initial solution
        dist_matrix (None | NDArray, optional): Distance Matrix. Defaults to None.
        max_time (float, optional): Maximum Time. Defaults to np.inf.

    Returns:
        dict: Improved Partition via LS
    """

    # count time
    start_time = time.time()

    # define a distance matrix
    if dist_matrix is None:
        dist_matrix = compute_distance_matrix(graph)

    # important variables
    local_optimum = False
    record_f = []

    # initialize sets
    frontier_set = get_frontier_set(graph, P)
    movable_nodes_set = get_movable_nodes_set(graph, P, frontier_set)

    # obj function
    f_P = obj_f_with_dist_matrix(P, dist_matrix)
    record_f.append(f_P)

    # explore unit local optimum
    explore = True
    while explore:

        # the best is to do nothing
        best_move = None
        f_best_N = f_P

        # explore posible moves, random order
        possible_moves = get_possible_moves(frontier_set, movable_nodes_set)
        while len(possible_moves) > 0:
            random_index = random.randint(0, len(possible_moves) - 1)
            move = possible_moves.pop(random_index)

            # evaluate it
            f_N = obj_f_move(P, f_P, move, dist_matrix)
            if f_N < f_best_N:
                f_best_N = f_N
                best_move = move

        # if a local optimum is reached
        if best_move is None:
            local_optimum = True
            explore = False 

        # there is a better solution
        else:

            # update sets
            frontier_set = update_frontier(graph, P, frontier_set, best_move)
            movable_nodes_set = update_movable_nodes(graph, P, frontier_set, movable_nodes_set, best_move)

            # apply move
            P = make_move(P, best_move)
            f_P = f_best_N
            record_f.append(f_P)

        # if maximum time is reached
        if time.time() - start_time > max_time:
            explore = False

    # return results
    elapsed_time = time.time() - start_time
    results = {
        "P": P,
        "f_P": f_P,
        "time": elapsed_time,
        "record_f": record_f,
        "local_optimum": local_optimum,
    }
    return results
