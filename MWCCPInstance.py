from typing import Dict

import numpy
import numpy as np


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
        with open(self.file_name, "r") as file:
            current_key = ""
            for line_idx, line in enumerate(file):
                if line_idx == 0:
                    self.instance["head"] = np.array([int(l) for l in line.split(" ")], dtype=int)
                    continue
                if "#" in line:
                    current_key = line[1:-1]
                    self.instance[current_key] = np.empty((0, 3 if current_key == "edges" else 2), dtype=int)
                    continue
                line = np.array([int(l) for l in line.split()], dtype=int).reshape(1, -1)
                self.instance[current_key] = np.vstack((self.instance[current_key], line))