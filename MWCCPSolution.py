from pymhlib.permutation_solution import PermutationSolution
import numpy as np
from pymhlib.solution import TObj

from MWCCPInstance import MWCCPInstance


class MWCCPSolution(PermutationSolution):
    instance_w: list[list[int]]
    instance_u: list[int]
    instance_v: list[int]
    instance_c: dict[(int, int)]
    instance_adj_v: dict[int, set[int]]
    instance_edges: list[(int, int, int)]
    to_maximize = False
    def __init__(self, inst: MWCCPInstance):
        super().__init__(inst.n, inst=inst)
        self.obj_val_valid = False

    def __init__(self, inst: MWCCPInstance, init=True, **kwargs):
        super().__init__(len(inst.get_instance()["u"]), init=False, inst=inst)
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
        objective_value = 0
        position = {v: i for i, v in enumerate(self.x)}
        for (u1, v1, weight) in self.instance_edges:
            for (u2, v2, weights) in self.instance_edges:
                if u1 < u2 and position[v1] > position[v2]:
                    total_crossings += 1
                    objective_value += weight + weights
        # return (total_crossings, objective_value)
        return objective_value

    def initialize(self, k):
        # super().initialize(k) is with random construction
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
        # Step 1: compute average position of each node in V

        averages = {}
        V = list(self.instance_v)
        for v in V:
            edges_to_v = self.instance_adj_v[v]
            if edges_to_v:
                total_position = sum((u) for u in edges_to_v)
                averages[v] = total_position / len(edges_to_v)
            else:
                averages[v] = 0

        # Step 2: sort V by average position
        sorted_V = sorted(V, key=lambda v: averages[v])

        # Step 3: resolve constraint violations
        for v, v_prime in self.instance_c.items():
            for v_prime in self.instance_c[v]:
                if sorted_V.index(v) >= sorted_V.index(v_prime):
                    # swap v and v_prime
                    i = sorted_V.index(v)
                    j = sorted_V.index(v_prime)
                    sorted_V[i], sorted_V[j] = sorted_V[j], sorted_V[i]

        self.x = np.array(sorted_V)
        self.invalidate()

    def construct_random(self):
        np.random.shuffle(self.x)

    def local_improve(self, _par, _result):
        self.two_opt_neighborhood_search(True)

    def two_exchange_move_delta_eval(self, p1: int, p2: int) -> TObj:
        """Return delta value in objective when exchanging positions p1 and p2 in self.x.

        The solution is not changed.
        This is a helper function for delta-evaluating solutions when searching a neighborhood that should
        be overloaded with a more efficient implementation for a concrete problem.
        Here we perform the move, calculate the objective value from scratch and revert the move.

        :param p1: first position
        :param p2: second position
        """
        obj = self.obj()
        x = self.x
        x[p1], x[p2] = x[p2], x[p1]
        self.invalidate()
        if x[p2] in self.instance_c.keys():
            c = self.instance_c[x[p2]]
            if x[p1] in c:
                delta = np.inf
            else:
                delta = self.obj() - obj
        else:
            delta = self.obj() - obj

        x[p1], x[p2] = x[p2], x[p1]
        self.obj_val = obj
        return delta


    def two_opt_neighborhood_search(self, best: bool):
        n = self.inst.n
        best_delta = 0
        best_p1 = None
        best_p2 = None
        order = np.arange(n)
        np.random.shuffle(order)
        for idx, p1 in enumerate(order[:n - 1]):
            for p2 in order[idx + 1:]:
                # consider exchange of positions p1 and p2
                delta = self.two_exchange_move_delta_eval(p1, p2)
                # obj_val = self.calc_objective()
                if self.is_better_obj(delta, best_delta):
                    if not best:
                        self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
                        self.obj_val += delta
                        return True
                    best_delta = delta
                    best_p1 = p1
                    best_p2 = p2
        if best_p1:
            self.x[best_p1], self.x[best_p2] = self.x[best_p2], self.x[best_p1]
            self.obj_val += best_delta
            return True
        return False

    def check(self):
        """
        Check if valid solution. All constraints must be satisfied.

        :raises ValueError: if problem detected.
        """
        for node, constraint_nodes in self.inst.get_instance()["c"].items():
            for v_prime in constraint_nodes:
                if list(self.x).index(node) >= list(self.x).index(v_prime):
                    raise ValueError(f"Constraint {node} {v_prime} violated.")
