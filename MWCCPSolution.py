from pymhlib.permutation_solution import PermutationSolution
from MWCCPInstance import MWCCPInstance


class MWCCPSolution(PermutationSolution):

    def __init__(self, inst: MWCCPInstance, length: int, **kwargs):
        super().__init__(len(inst.get_instance()["u"]), inst=inst.get_instance())

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
