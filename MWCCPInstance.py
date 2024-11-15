from typing import Dict

import numpy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MWCCPInstance:
    """MWCC problem instance.

            The goal is to minimize the number of crossings for v' of a bipartite Graph G = (U ∪ V, E), while the constraint is satisfied. Constraint C is a set of n-tuples
            C = { (v,v')_n, .... }.

            A probleminstance is represented as so:
            - head: instance head, information for the whole instance
            - constraints: each row is a element of the constraints set, represented as a tuple
            - edges: each row is a element of the edge set, represented as a triple. Like so E = ( u, v, w) , whereas w stands for edge weight

            """
    file_name = ""


    def __int__(self, file_name):
        self.file_name: str = file_name
        self.instance: Dict[str, numpy.ndarray] = {}

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
        for line in lines[1:]:
            line = line.strip()
            if line == '#edges':
                edges_section = True
                continue
            if edges_section:
                if line == '':
                    continue  # Skip empty lines
                # Parse edge data
                i, j, w = map(int, line.split())
                edges.append((i, j, w))

        # Collect vertex numbers for U and V
        U_vertices = set()
        V_vertices = set()
        for i, j, w in edges:
            U_vertices.add(i)
            V_vertices.add(j)

        # Create U_vector and V_vector with the specified sizes
        U_vector = np.fromiter(U_vertices, int)
        V_vector = np.fromiter(V_vertices, int)

        max_vertex = max(max(U_vertices, default=0), max(V_vertices, default=0))
        size = max_vertex + 1  # Adjusting for 1-based indexing

        weight_matrix = [[0] * size for _ in range(size)]

        # Fill the weight matrix symmetrically
        for i, j, w in edges:
            weight_matrix[i][j] = w
            #weight_matrix[j][i] = w  # Symmetry

        self.instance = {"u": U_vector, "v": V_vector, "w": weight_matrix}

    def draw_instance(self):
        graph = nx.Graph()

        self.set_problem_instance()
        graph.add_nodes_from(self.instance["u"], bipartite=0)  # Top nodes
        graph.add_nodes_from(self.instance["v"], bipartite=1)  # Bottom nodes

        edges = []

        for u,v in zip(np.nditer(self.instance["u"]),np.nditer(self.instance["v"])):
            edges = edges.append((u, v))

        graph.add_edges_from(edges)

