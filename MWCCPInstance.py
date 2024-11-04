import numpy as np
class MWCCPInstace():
    """MWCC problem instance.

        The goal is to minimize the number of crossings for v' of a bipartite Graph G = (U ∪ V, E), while the constraint is satisfied. Constraint C is a set of n-tuples
        C = { (v,v')_n, .... }.

        A probleminstance is represented as so:
        - head: instance head, information for the whole instance
        - constraints: each row is a element of the constraints set, represented as a tuple
        - edges: each row is a element of the edge set, represented as a triple. Like so E = ( u, v, w) , whereas w stands for edge weight

        """
    def __int__(self, file_name):
        with open(file_name, "r") as file:
            for line in file:
                self.instance = (
                    "head", line[0]
                )
                break
    def get_instance(self):
        return self.instance

