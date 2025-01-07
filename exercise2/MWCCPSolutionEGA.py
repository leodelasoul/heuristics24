from abc import ABC

from pymhlib.permutation_solution import PermutationSolution
from exercise1.v2_MWCCPInstance import v2_MWCCPInstance


class MWCCPSolutionEGA(PermutationSolution):
    to_maximize = False

    def __init__(self, inst: v2_MWCCPInstance):
        super().__init__(len(inst.get_instance()["u"]), init=False, inst=inst)
        self.instance_c = None
        self.instance_edges = None

    def copy(self):
        sol = MWCCPSolutionEGA(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        objective_value = 0
        position = {v: i for i, v in enumerate(self.x)}
        for (u1, v1, weight) in self.instance_edges:
            for (u2, v2, weights) in self.instance_edges:
                if u1 < u2 and position[v1] > position[v2]:
                    objective_value += weight + weights
        return objective_value

    def construct(self):
        print("execute construct")
        pass

    def crossover(self):
        print("execute crossover")

        pass

    def shaking(self):
        print("execute shaking")

        pass

    def local_improve(self):
        print("execute local_improve")

        pass

    def check(self, *args):
        x = self.x
        a = None
        for a in args:
            if len(x) > 0:
                x = a
        for node, constraint_nodes in self.instance_c.items():
            for v_prime in self.instance_c[node]:
                if list(x).index(node) >= list(x).index(v_prime):
                    if a is not None:
                        return False
                    else:
                        raise ValueError(f"Constraint {node} {v_prime} violated.")


