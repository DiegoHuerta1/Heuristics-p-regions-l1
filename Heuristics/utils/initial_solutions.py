from igraph import Graph
from numpy.typing import NDArray
import numpy as np
import random

from Heuristics.utils.utils import compute_distance_matrix
from Heuristics.utils.utils_sg import get_feasible_elements, get_RLC, update_feasible


def greedy_algorithm(graph: Graph, num_regions: int,
                     dist_matrix: NDArray) -> dict:
    """
    Greedy algorithm for generating initial solutions for LS


    Args:
        graph (Graph): Graph instance
        num_regions (int): Number of regions
        dist_matrix (NDArray): Distance Matrix

    Returns:
        dict: Partition
    """

    # select seeds, randomly
    list_nodes = np.arange(graph.vcount())
    seeds = np.random.choice(list_nodes, size = num_regions, replace = False)

    # Only define weights if not yet defined
    if "w" not in graph.es.attributes():
        graph.es["w"] = [dist_matrix[v, u] for v, u in graph.get_edgelist()]

    # compute distances, from each seed
    dist_from_seeds = graph.distances(source = seeds,
                                      weights = graph.es["w"],
                                      algorithm = "dijkstra")
    
    # assign each node to closest seed
    P0 = {i:[] for i in range(1, num_regions + 1)}
    for v in range(graph.vcount()):
        dist_seeds_from_v = {idx+1: dist_from_seeds[idx][v] for idx, _ in enumerate(seeds)}
        k_star =  min(dist_seeds_from_v, key = lambda i: (dist_seeds_from_v[i], i))
        P0[k_star].append(v)

    return P0

    

def adaptative_semi_greedy(graph: Graph, num_regions: int,
                           dist_matrix: None | NDArray = None,
                           alpha: float = 0.1) -> dict:
    """Adaptative Semi Greedy for initial solutions in GRASP

    Args:
        graph (Graph): Graph instance
        num_regions (int): Number of regions
        dist_matrix (None | NDArray, optional): Distance Matrix. Defaults to None.
        alpha (float, optional): Alpha value. Defaults to 0.1.

    Returns:
        dict: Partition
    """

    # define a distance matrix
    if dist_matrix is None:
        dist_matrix = compute_distance_matrix(graph)

    # start with seeds
    list_nodes = np.arange(graph.vcount())
    seeds = np.random.choice(list_nodes, size = num_regions, replace = False)
    P_sg = {(idx+1): [s] for idx, s in enumerate(seeds)}

    # get feasible elements, iterate while there are elements
    dict_feasible = get_feasible_elements(graph, P_sg, dist_matrix)
    while len(dict_feasible) > 0:

        # get RLC, and select element
        RLC_list = get_RLC(dict_feasible, alpha)
        if alpha == 0:
            star_element = RLC_list[0]
        else:
            star_element = random.choice(RLC_list)
        v_star = star_element[0]
        k_star = star_element[1]

        # add to region
        P_sg[k_star].append(v_star)

        # update feasible elements
        dict_feasible = update_feasible(graph, P_sg, dist_matrix, dict_feasible, star_element)

    return P_sg


