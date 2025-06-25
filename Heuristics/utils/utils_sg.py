from igraph import Graph
from numpy.typing import NDArray
import numpy as np
from Heuristics.utils.utils import function_a


def get_feasible_elements(graph: Graph, P, dist_matrix):
    dict_feasible = dict()

    # get assigned nodes
    assigned_nodes = []
    for nodes_region in P.values():
        assigned_nodes.extend(nodes_region)

    # iterate in assigned nodes, and their region
    for u_assigned in assigned_nodes:
        assigned_region = function_a(u_assigned, P)

        # iterate in their neighbors
        for w in graph.neighbors(u_assigned):

            # w is unnasigned, that is a feasible element
            if w not in assigned_nodes:
                f_element = (w, assigned_region)

                # if it is a duplicate, ignore, if not, add
                if f_element in dict_feasible:
                    continue
                dict_feasible[f_element] = greedy_function_element(P, dist_matrix, f_element)

    return dict_feasible


def greedy_function_element(P, dist_matrix, feasible_element):
    v = feasible_element[0]
    k = feasible_element[1]
    # compute distances, return sum
    distances = [dist_matrix[v, u] for u in P[k]]
    return np.sum(distances)


def get_RLC(dict_feasible, alpha):
    
    # take max and min
    greedy_evaluations = list(dict_feasible.values())
    max_greedy = max(greedy_evaluations)
    min_greedy = min(greedy_evaluations)

    # filter
    RLC_elements = [element 
                    for element, greedy_element
                    in dict_feasible.items()
                    if min_greedy <= greedy_element and
                    greedy_element <= min_greedy + alpha * (max_greedy - min_greedy)]
    return RLC_elements


def update_feasible(graph, P_sg, dist_matrix, dict_feasible_old, star_element):
    # star element
    v_star = star_element[0]
    k_star = star_element[1]
    
    # v star can be part of a feasible element
    dict_feasible = {element: greedy_element
                     for element, greedy_element 
                     in dict_feasible_old.items()
                     if element[0] != v_star}
        
    
    # re evaluate greedy function
    for element in dict_feasible_old.keys():
        u = element[0]
        h = element[1]
        
        # only if u != v star
        if u == v_star:
            continue
        
        # assign u to k_star, update greedy functiom     
        if h == k_star:
            dict_feasible[element] = dict_feasible_old[element] + dist_matrix[u, v_star]

    # assigned nodes
    assigned_nodes = []
    for nodos_region in P_sg.values():
        assigned_nodes.extend(nodos_region)
    
    # unasigned neighbors of v star
    for w in graph.neighbors(v_star):
        if w not in assigned_nodes:
        
            # potential new element: (w, k_star)
            new_element = (w, k_star)
            if new_element in dict_feasible:
                continue
            # add it
            dict_feasible[new_element] = greedy_function_element(P_sg, dist_matrix, new_element)
    
    return dict_feasible


