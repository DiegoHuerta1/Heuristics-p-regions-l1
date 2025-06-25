import numpy as np
import itertools


def compute_distance_matrix(G):

    # initialize as a 0 matrix
    num_nodos = G.vcount()
    dist_matrix = np.zeros((num_nodos, num_nodos))

    # for each node, take its vector
    for i in range(num_nodos):
        vector_i = G.vs[i]['x']
        # for each other node, do not repeat pairs, take its vector
        for j in range(i, num_nodos):
            vector_j = G.vs[j]['x']

            # compute the l1 norm if the diference
            dist_matrix[i, j] = np.sum(np.abs(vector_i - vector_j))

    # make symetric
    dist_matrix = dist_matrix + dist_matrix.T
    return dist_matrix


def function_a(element, P):
    for idx_region, region in P.items():
        if element in region:
            return idx_region
    raise Exception(f"Node {element} not found")



def get_frontier_node(v, P, graph):
    a_N = set([function_a(u, P) for u in graph.neighbors(v)])
    a_v = function_a(v, P)
    return a_N - {a_v}

def get_frontier_set(grafo, P):
    frontier_set = {node_v: get_frontier_node(node_v, P, grafo)
                        for node_v in range(grafo.vcount())}
    return frontier_set


def update_frontier(graph, P, frontier_set, move):
    # get neigbhor and node
    v = move[1]
    P_prime = make_move(P, move)

    # start with the same
    new_frontier = frontier_set.copy()

    # update
    new_frontier[v] = get_frontier_node(v, P_prime, graph)
    for u in graph.neighbors(v):
        new_frontier[u] = get_frontier_node(u, P_prime, graph)

    return new_frontier



def can_be_removed(graph, P, idx, node):
    # subgraph of region
    subgraph_reg = graph.subgraph(P[idx])

    # remove node from subgraph
    node_name = graph.vs[node]['name']
    node_subgraph = subgraph_reg.vs.find(name= node_name).index
    subgraph_reg.delete_vertices(node_subgraph)

    # subgraph infeasible
    if subgraph_reg.vcount() == 0 or not subgraph_reg.is_connected():
        return False
    return True


def movable_nodes_reg(graph, P, idx, frontier_set):
    # filter with non empty frontier
    mov_nodes_preliminar = [v for v in P[idx] if len(frontier_set[v]) > 0]
    # only if it can be removed
    mov_nodes = [v for v in mov_nodes_preliminar if can_be_removed(graph, P, idx, v)]
    return mov_nodes

def get_movable_nodes_set(graph, P, frontier_set):
    mov_nodes = {idx_i: movable_nodes_reg(graph, P, idx_i, frontier_set) for idx_i in P.keys()}
    return mov_nodes

def update_movable_nodes(graph, P, frontier_set, movable_nodes_set, move):
    # get elements, and neighbot
    k1 = move[0]
    k2 = move[2]
    P_prime = make_move(P, move)

    # start with the same
    new_movable_nodes = movable_nodes_set.copy()

    # update
    new_movable_nodes[k1] = movable_nodes_reg(graph, P_prime, k1, frontier_set)
    new_movable_nodes[k2] = movable_nodes_reg(graph, P_prime, k2, frontier_set)

    return new_movable_nodes



def get_possible_moves(frontier_set, movable_nodes_set):
    possible_moves = [(k1, v, k2)
                      for k1, movable_nodes_reg in movable_nodes_set.items()
                      for v in movable_nodes_reg
                      for k2 in frontier_set[v]]
    return possible_moves


def make_move(P, move):
    # get elements
    k1 = move[0]
    v = move[1]
    k2 = move[2]

    # start the same
    P_prime = P.copy()

    # make move
    P_prime[k1] = [u for u in P[k1] if u != v]
    P_prime[k2] = P[k2] + [v]
    return P_prime


def obj_f_with_dist_matrix(P, dist_matrix):
    sum = 0

    # for each pair of nodes
    for P_i in P.values():
        pairs = list(itertools.combinations(P_i, 2))
        for i, j in pairs:

            # add distance
            sum += dist_matrix[i, j]

    return sum

def obj_f_move(P, f_P, move, dist_matrix):
    k1 = move[0]
    v = move[1]
    k2 = move[2]
    # compute based on the actual
    excendetes = [dist_matrix[v, u] for u in P[k1]]
    faltantes =  [dist_matrix[v, u] for u in P[k2]]
    return f_P - np.sum(excendetes) + np.sum(faltantes)
