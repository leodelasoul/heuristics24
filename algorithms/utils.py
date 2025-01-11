import numpy as np
from typing import Dict, List, Tuple

class MWCCPInstance:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.U: List[int] = []
        self.V: List[int] = []
        self.constraints: List[Tuple[int, int]] = []
        self.edges: List[Tuple[int, int, int]] = []
        self.crossing_matrix = None

        # Automatically parse the instance file
        self._parse_instance()
        self._precompute_crossings()

    def _parse_instance(self):
        """
        Parse the instance file to extract U, V, constraints, and edges.
        """
        with open(self.file_name, 'r') as f:
            lines = f.readlines()

        # Parse the first line to get sizes
        first_line = lines[0].strip()
        U_size, V_size, num_constraints, num_edges = map(int, first_line.split())

        # Initialize U and V sets
        self.U = list(range(1, U_size + 1))
        self.V = list(range(U_size + 1, U_size + V_size + 1))

        # Parse constraints and edges
        parsing_constraints = False
        parsing_edges = False

        for line in lines[1:]:
            line = line.strip()

            if line == "#constraints":
                parsing_constraints = True
                parsing_edges = False
                continue

            if line == "#edges":
                parsing_edges = True
                parsing_constraints = False
                continue

            if parsing_constraints and line:
                # Parse constraints as tuples
                v1, v2 = map(int, line.split())
                self.constraints.append((v1, v2))

            if parsing_edges and line:
                # Parse edges as triples (u, v, weight)
                u, v, weight = map(int, line.split())
                self.edges.append((u, v, weight))

    def _precompute_crossings(self):
        # Create a mapping of edges for each node in V
        edges_by_v = {v: [] for v in self.V}
        for u, v, weight in self.edges:
            edges_by_v[v].append((u, weight))

        # Initialize crossing matrix
        self.crossing_matrix = np.zeros((len(self.V), len(self.V)))

        # Compute crossing contributions for each pair of V nodes
        for i, v1 in enumerate(self.V):
            for j, v2 in enumerate(self.V):
                if i >= j:
                    continue  # Avoid duplicate pairs

                for (u1, w1) in edges_by_v[v1]:
                    for (u2, w2) in edges_by_v[v2]:
                        contribution = 0
                        if u1 != u2:  # Edges must connect different U nodes to cross
                            contribution += w1 + w2
                        if u1 > u2:            
                            self.crossing_matrix[i, j] += contribution
                        elif u1 < u2:
                            self.crossing_matrix[j, i] += contribution

        

    def get_crossing_contribution(self, v1, v2):
        i, j = self.V.index(v1), self.V.index(v2)
        return self.crossing_matrix[i, j]
    

    def get_instance(self) -> Dict[str, np.ndarray]:
        """
        Get the parsed problem instance as a dictionary.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary containing U, V, constraints, and edges.
        """
        return {
            "U": np.array(self.U),
            "V": np.array(self.V),
            "constraints": np.array(self.constraints),
            "edges": np.array(self.edges),
        }

    def __repr__(self):
        """
        String representation of the instance for debugging purposes.
        """
        return (
            f"MWCCPInstance(\n"
            f"  U: {self.U},\n"
            f"  V: {self.V},\n"
            f"  Constraints: {self.constraints},\n"
            f"  Edges: {self.edges}\n"
            f")"
        )
