from pymhlib.permutation_solution import PermutationSolution

from MWCCPInstance import MWCCPInstance


class MWCCPSolution(PermutationSolution):
    instance_w: list[list[int]]
    instance_u: list[int]
    instance_v: list[int]
    instance_c: list[(int, int)]
    instance_edges: list[(int, int, int)]

    def __init__(self, inst: MWCCPInstance, length: int, **kwargs):
        super().__init__(len(inst.get_instance()["u"]), inst=inst.get_instance())
        self.instance_w = inst.get_instance()["w"]
        self.instance_u = inst.get_instance()["u"]
        self.instance_v = inst.get_instance()["v"]
        self.instance_c = inst.get_instance()["c"]

        edges = []
        for u, i in enumerate(self.instance_w[1:]):
            for v_index, weight in enumerate(i):
                if weight > 0:
                    edges.append((u + 1, v_index, weight))
        self.instance_edges = edges

    def copy(self):
        sol = MWCCPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        total_crossings = 0
        objective_value= 0
        position = {v: i for i, v in enumerate(self.instance_v)}
        for (u1, v1, weight) in self.instance_edges:
            for (u2, v2, weights) in self.instance_edges:
                if u1 < u2 and position[v1] > position[v2]:
                    total_crossings += 1
                    objective_value += weight + weights
        return (total_crossings, objective_value)
