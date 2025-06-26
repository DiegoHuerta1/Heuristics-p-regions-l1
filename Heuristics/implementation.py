import numpy as np
import random
import time
import warnings

from igraph import Graph
from numpy.typing import NDArray

from Heuristics.utils.initial_solutions import greedy_algorithm
from Heuristics.utils.initial_solutions import adaptative_semi_greedy
from Heuristics.utils.local_search import local_search_from_partition
from Heuristics.utils.utils import compute_distance_matrix


def Multi_Start_LS(graph: Graph, num_regions: int,
                   n_iter: int = 100,
                   dist_matrix: None | NDArray = None,
                   max_time: float = np.inf) -> dict:
    """
    Multi Start Local Search

    Args:
        graph (Graph): Graph instance
        num_regions (int): Number of regions K
        n_iter (int, optional): Number of local seach iterations. Defaults to 100.
        dist_matrix (None | NDArray, optional): Distance Matrix, if available. Defaults to None.
        max_time (float, optional): Maximum execution time. Defaults to np.inf.

    Returns:
        dict: Dictionary with keys:
        "P"
        "f_P"
        "Total Time"
        "Time P0"
        "Time: LS"
        "record_f"
    """
        
    # count execution time, and for diferent concepts
    start_time_general = time.time()
    time_P0 = 0
    time_LS = 0
    
    # define a distance matrix
    if dist_matrix is None:
        dist_matrix = compute_distance_matrix(graph)

    # save the evolution of f for each execution
    f_record = []

    # the best found solution
    P_best = None
    f_P_best = float("inf")

    # make all iterations
    for _ in range(n_iter):
        
        # get initial solution, count time
        start_time_P0 = time.time()
        P_0 = greedy_algorithm(graph, num_regions)       
        time_P0_it = time.time() - start_time_P0
        
        # get the remaining available time
        elapsed_time = time.time() - start_time_general
        available_time = max_time - elapsed_time

        # use local search, count time
        start_time_ls = time.time()
        ls_results = local_search_from_partition(graph, P_0, dist_matrix, available_time)
        P_it = ls_results["P"]
        f_P_it = ls_results["f_P"]
        hist_it = ls_results["record_f"]
        time_LS_it = time.time() - start_time_ls
        
        # add execution times 
        time_P0 += time_P0_it
        time_LS += time_LS_it
        
        # update based on this iteration
        f_record.append(hist_it)
        if f_P_it < f_P_best:
            P_best = P_it
            f_P_best = f_P_it
            
        # if the last ls did not reach a local optimim, the time is up
        if not ls_results["local_optimum"]:
            break

    # return results
    results = {
        "P": P_best,
        "f_P": f_P_best,
        "Total Time": time.time() - start_time_general,
        "Time: P0": time_P0,
        "Time: LS": time_LS,
        "record_f": f_record
    }    
    return results


def GRASP(graph: Graph, num_regions: int,
          n_iter: int = 100,
          alpha: int | float | list | tuple = 0.1,
          dist_matrix: None | NDArray = None,
          max_time: float = np.inf) -> dict:
    """
    GRASP 

    Args:
        graph (Graph): Graph instance
        num_regions (int): Number of regions K 
        alpha (float | list | tuple, optional): Parameter alpha for the semi greedy
        if float, it is constant in all iterations,
        if a list, it must contain the [min, max] alpha values. Defaults to 0.1.
        n_iter (int, optional): Number of local seach iterations. Defaults to 100.
        dist_matrix (None | NDArray, optional): Distance Matrix. Defaults to None.
        max_time (float, optional): Maximum execution time. Defaults to np.inf.

    Returns:
        dict: Dictionary with keys:
        "P"
        "f_P"
        "Total Time"
        "Time P0"
        "Time: LS"
        "record_f"
    """

    # count execution time, and for diferent concepts
    start_time_general = time.time()
    time_P0 = 0
    time_LS = 0
    
    # define a distance matrix
    if dist_matrix is None:
        dist_matrix = compute_distance_matrix(graph)

    # save the record of f, for each execution
    f_record = []

    # the best found solution
    P_best = None
    f_P_best = float("inf")
    
    # define all alpha values to explore
    if isinstance(alpha, float): 
        alpha_iterations = [alpha] * n_iter
    elif isinstance(alpha, list) or isinstance(alpha, tuple): 
        min_alpha = alpha[0]
        max_alpha = alpha[-1]
        alpha_iterations = np.linspace(min_alpha, max_alpha, num = n_iter)
    else:
        warnings.warn("alpha value not valid!", UserWarning)
        return dict()
            
    # make all iterations
    for alpha_iter in alpha_iterations:

        # get initial solution, count time
        start_time_P0 = time.time()
        P0 = adaptative_semi_greedy(graph, num_regions,
                                            dist_matrix= dist_matrix,
                                            alpha= alpha_iter)
        time_P0_it = time.time() - start_time_P0
        
        # get the remaining available time
        elapsed_time = time.time() - start_time_general
        available_time = max_time - elapsed_time

        # use local search, count time
        start_time_ls = time.time()
        ls_results = local_search_from_partition(graph, P0, dist_matrix, available_time)
        P_it = ls_results["P"]
        f_P_it = ls_results["f_P"]
        hist_it = ls_results["record_f"]
        time_LS_it = time.time() - start_time_ls
                
        # add execution times 
        time_P0 += time_P0_it
        time_LS += time_LS_it
        
        # update based on this iteration
        f_record.append(hist_it)
        if f_P_it < f_P_best:
            P_best = P_it
            f_P_best = f_P_it
            
        # if the last ls did not reach a local optimim, the time is up
        if not ls_results["local_optimum"]:
            break
    
    # return results
    results = {
        "P": P_best,
        "f_P": f_P_best,
        "Total Time": time.time() - start_time_general,
        "Time: P0": time_P0,
        "Time: LS": time_LS,
        "record_f": f_record
    }    
    return results

