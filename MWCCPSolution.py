from pymhlib.permutation_solution import PermutationSolution
import numpy as np

from MWCCPInstance import MWCCPInstance


class MWCCPSolution(PermutationSolution):
    
    def __init__(self, inst: MWCCPInstance, init=True):
        """Initializes the solution withthe nodes in set V if init is set."""
        super().__init__(len(inst.get_instance()["u"]), init = False, inst=inst.get_instance())
        
        if init:
            self.x = np.array(list(inst.get_instance()["v"]))
    
        self.obj_val_valid = False

    def copy(self):
        sol = MWCCPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        distance = 0
        for i in range(self.inst.n - 1):
            distance += self.inst.distances[self.x[i]][self.x[i + 1]]
        distance += self.inst.distances[self.x[-1]][self.x[0]]
        return distance
    
        
    def initialize(self, k):
        #super().initialize(k) is with random construction
        super().initialize(k)
        self.invalidate()
    

    def construct(self):
        """
        Construct a solution using a greedy heuristic.
    
        :return: A feasible permutation of V.
        """
        #Step 1: compute average position of each node in V
        
        V = self.inst.get_instance()["v"]
        U = self.inst.get_instance()["u"]
        averages = {}
        for v in V:
            edges_to_v = self.inst.get_instance()["adj_v"][v]
            if edges_to_v:
                total_position = sum((U.index(u)+1) for u in edges_to_v)
                averages[v] = total_position / len(edges_to_v)
            else:
                averages[v] = 0

        #Step 2: sort V by average position
        sorted_V = sorted(V, key=lambda v: averages[v])

        #Step 3: resolve constraint violations
        for v, v_prime in self.inst.get_instance()["c"].items():
            if sorted_V.index(v) >= sorted_V.index(v_prime):
                #swap v and v_prime
                i = sorted_V.index(v)
                j = sorted_V.index(v_prime)
                sorted_V[i], sorted_V[j] = sorted_V[j], sorted_V[i]
        
        self.x = np.array(sorted_V)
        self.invalidate()


    def construct_random(self):
        np.random.shuffle(self.x)

    def check(self):
        """
        Check if valid solution. All constraints must be satisfied.

        :raises ValueError: if problem detected.
        """
        for node, constraint_nodes in self.inst.get_instance()["c"].items():
            for v_prime in constraint_nodes:
                if list(self.x).index(node) >= list(self.x).index(v_prime):
                    raise ValueError(f"Constraint {node} {v_prime} violated.")


        
