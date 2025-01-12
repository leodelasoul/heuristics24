from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import numpy as np

import util as util


class v2_MWCCPInstance:
    """MWCC problem instance.

            The goal is to minimize the number of crossings for v' of a bipartite Graph G = (U ∪ V, E), while the constraint is satisfied. Constraint C is a set of n-tuples
            C = { (v,v')_n, .... }.

            A probleminstance is represented as so:
            - head: instance head, information for the whole instance
            - constraints: each row is a element of the constraints set, represented as a tuple
            - edges: each row is a element of the edge set, represented as a triple. Like so E = ( u, v, w) , whereas w stands for edge weight

            """
    file_name = ""
    n = 0
    edges = None

    def __init__(self, file_name):
        self.file_name: str = file_name
        self.instance: Dict[str, numpy.ndarray] = {}

        self.set_problem_instance()

       
    def get_instance(self):
        return self.instance

    def set_problem_instance(self):
        with open(self.file_name, 'r') as f:
            lines = f.readlines()

        # Parse the first line to get sizes
        first_line = lines[0].strip()
        U_size, V_size, num_constraints, num_edges = map(int, first_line.split())

        # Skip to the #edges section
        edges_section = False
        edges = []
        constraint_selection = False
        constraints = []

        for line in lines[1:]:
            line = line.strip()
            if line == '#edges':
                edges_section = True
                constraint_selection = False
                continue
            if edges_section:
                if line == '':
                    continue  # Skip empty lines
                # Parse edge data
                i, j, w = map(int, line.split())
                edges.append((i, j, w))

            if line == '#constraints':
                constraint_selection = True
                continue
            if constraint_selection:
                if line == '':
                    continue
                # Parse constraint data
                i, j = map(int, line.split())
                constraints.append((i, j))

        # Create a dictionary to store constraints for efficient lookup
        constraint_dict = {}
        for i, j in constraints:
            if i not in constraint_dict:
                constraint_dict[i] = []
            constraint_dict[i].append(j)
            
        edges = list(edges)
        constraints = np.array(constraints)

        # Collect vertex numbers for U and V
        U_vertices = set()
        V_vertices = set()

        #max_vertex = max(max(U_vertices, default=0), max(V_vertices, default=0))
        #size = max_vertex + 1  # Adjusting for 1-based indexing
        size = U_size + V_size + 1
        weight_matrix = [[0] * size for _ in range(size)] 
        edge_matrix = [[0] * size for _ in range(size)]


        adjacency_from_V = {}

        # Fill the weight matrix symmetrically
        for i, j, w in edges:
            U_vertices.add(i)
            V_vertices.add(j)

            weight_matrix[i][j] = w
            weight_matrix[j][i] = w  # Symmetry

            edge_matrix[i][j] = 1
            edge_matrix[j][i] = 1  # Symmetry

            # Create adjacency list for V
            if j not in adjacency_from_V:
                adjacency_from_V[j] = []
            adjacency_from_V[j].append(i)


        # Create U_vector and V_vector with the specified sizes
        U_vector = np.fromiter(U_vertices, int)
        V_vector = np.fromiter(V_vertices, int)

        self.n = U_size

        crossing_contribution = self.preprocess_crossing_contributions(U_vector, V_vector, edges, weight_matrix)

        self.instance = {"u": U_vector, "v": V_vector, "c": constraint_dict, "crossing_contrib": crossing_contribution, "w": weight_matrix, "edges": edges, "c_tup": constraints, "adj_v": adjacency_from_V, "n": U_size}

    
    def preprocess_crossing_contributions(self, vertices_u, vertices_v, edges, weights):
        # Map nodes in V to their edges
        edges_by_v = defaultdict(list)
        for (u, v, w) in edges:
            edges_by_v[v].append((u, weights[u][v]))

        # Initialize the crossing contribution dictionary
        contribution = {}

        # Precompute contributions for all pairs of nodes in V
        for i, v1 in enumerate(vertices_v):
            for j, v2 in enumerate(vertices_v):
                if i == j:
                    continue  # Skip duplicate or self-pairs

                # Edges for v1 and v2
                edges_v1 = edges_by_v[v1]
                edges_v2 = edges_by_v[v2]

                # Compute crossing contributions for (v1, v2)
                contrib_v1_left = 0
                contrib_v2_left = 0

                for u1, w1 in edges_v1:
                    for u2, w2 in edges_v2:
                        if u1 == u2:
                            continue  # Edges ending in the same U node do not cross

                        contrib = w1 + w2  # Crossing contribution for this edge pair
                        # Check if the edges (v1, u1) and (v2, u2) cross
                        if u1 > u2:
                            contrib_v1_left += contrib  # When v1 is left of v2
                        elif u1 < u2:
                            contrib_v2_left += contrib  # When v2 is left of v1

                # Store contributions
                contribution[(v1, v2)] = contrib_v1_left
                contribution[(v2, v1)] = contrib_v2_left

        return contribution