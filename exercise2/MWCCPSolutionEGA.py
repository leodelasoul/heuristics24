from abc import ABC

import numpy as np
from pymhlib.permutation_solution import PermutationSolution
from exercise1.v2_MWCCPInstance import v2_MWCCPInstance
import sys


class MWCCPSolutionEGA(PermutationSolution, ABC):
    to_maximize = False

    def __init__(self, inst: v2_MWCCPInstance):
        super().__init__(len(inst.get_instance()["u"]), init=False, inst=inst)
        self.instance_c_tup = inst.get_instance()["c_tup"]
        self.prior_obj_val = sys.maxsize
        self.instance_c = inst.get_instance()["c"]
        self.instance_edges = inst.get_instance()["edges"]

    def copy(self):
        sol = MWCCPSolutionEGA(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        objective_value = 0
        for (u1, v1, weight1) in self.instance_edges:
            for (u2, v2, weight2) in self.instance_edges:
                if u1 < u2 and self.x[abs(v1 - self.inst.n) - 1] > self.x[
                    abs(v2 - self.inst.n) - 1]:  # offset(-1) needed because instance_edges are counted from 1 instead of 0
                    objective_value += weight1 + weight2
        return objective_value

    def construct(self, _sol, _par, _res): #GA INIT STEP
        order = np.arange(self.inst.n)
        np.random.shuffle(order)
        x = self.check_order_constraints(order, construct=True)
        self.x = x
        current_obj_val = self.calc_objective()
        if self.is_better_obj(current_obj_val, self.prior_obj_val):
            self.obj_val = current_obj_val
        self.prior_obj_val = self.calc_objective()
        print(self.prior_obj_val)

    def crossover(self, _par1, _par2): #GA RECOMBINE STEP
        pass

    def shaking(self, _sol, _par, _res):
        pass

    def local_improve(self, _sol, _par, _res): #GA REPLACE STEP
        pass


    def check_order_constraints(self, order, construct):
        # Repeat until all constraints are satisfied
        swapped = True
        constraint_pairs = self.instance_c_tup
        arr = np.array([self.x[i] for i in np.nditer(order)])
        while swapped:
            swapped = False
            for a, b in constraint_pairs:
                indices_a = np.where(arr == a)[0]
                indices_b = np.where(arr == b)[0]
                if indices_a.size == 0 or indices_b.size == 0:
                    continue
                index_a = indices_a[0]
                index_b = indices_b[0]
                if index_a > index_b:
                    arr[index_a], arr[index_b] = arr[index_b], arr[index_a]
                    order[index_a], order[index_b] = order[index_b], order[index_a]
                    swapped = True
        x = np.array([self.x[i] for i in np.nditer(order)])
        if construct:
            return x
        return order

    def check(self):
        for node, constraint_nodes in self.instance_c.items():
            for v_prime in self.instance_c[node]:
                if list(self.x).index(node) >= list(self.x).index(v_prime):
                    return False
                else:
                    raise ValueError(f"Constraint {node} {v_prime} violated.")