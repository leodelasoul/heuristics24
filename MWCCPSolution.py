from pymhlib.permutation_solution import PermutationSolution
import numpy as np

from MWCCPInstance import MWCCPInstance


class MWCCPSolution(PermutationSolution):
    instance_w: list[list[int]]
    instance_u: list[int]
    instance_v: list[int]
    instance_c: list[(int, int)]
    instance_adj_v: dict[int, set[int]]
    instance_edges: list[(int, int, int)]

    def __init__(self, inst: MWCCPInstance):
        super().__init__(inst.n, inst=inst)
        self.obj_val_valid = False

    def __init__(self, inst: MWCCPInstance, init=True, **kwargs):
        super().__init__(len(inst.get_instance()["u"]), init = False, inst=inst)
        self.instance_w = inst.get_instance()["w"]
        self.instance_u = inst.get_instance()["u"]
        self.instance_v = inst.get_instance()["v"]
        self.instance_adj_v = inst.get_instance()["adj_v"]
        self.instance_c = inst.get_instance()["c"]
        inst.n = len(self.instance_v)

        edges = []
        for u, i in enumerate(self.instance_w[1:]):
            for v_index, weight in enumerate(i):
                if weight > 0:
                    edges.append((u + 1, v_index, weight))
        self.instance_edges = edges
        if init:
            self.x = np.array(list(self.instance_v))

        self.obj_val_valid = False

    def copy(self):
        sol = MWCCPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        total_crossings = 0
        objective_value= 0
        position = {v: i for i, v in enumerate(self.x)}
        for (u1, v1, weight) in self.instance_edges:
            for (u2, v2, weights) in self.instance_edges:
                if u1 < u2 and position[v1] > position[v2]:
                    total_crossings += 1
                    objective_value += weight + weights
        #return (total_crossings, objective_value)
        return objective_value

    def initialize(self, k):
        #super().initialize(k) is with random construction
        super().initialize(k)
        self.invalidate()

    def construct_pymlib(self, par, _result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)


    def construct(self, par, _result):
        """
        Construct a solution using a greedy heuristic.

        :return: A feasible permutation of V.
        """
        #Step 1: compute average position of each node in V

        averages = {}
        V = list(self.instance_v)
        for v in V:
            edges_to_v = self.instance_adj_v[v]
            if edges_to_v:
                total_position = sum((u) for u in edges_to_v)
                averages[v] = total_position / len(edges_to_v)
            else:
                averages[v] = 0

        #Step 2: sort V by average position
        sorted_V = sorted(V, key=lambda v: averages[v])

        #Step 3: resolve constraint violations
        for v, v_prime in self.instance_c.items():
            print(f"Checking constraint {v} {v_prime}")
            for v_prime in self.instance_c[v]:
                if sorted_V.index(v) >= sorted_V.index(v_prime):
                    #swap v and v_prime
                    print()
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



