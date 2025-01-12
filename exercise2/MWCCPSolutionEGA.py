import random
from abc import ABC

import numpy as np
from pymhlib.permutation_solution import PermutationSolution
from exercise1.v2_MWCCPInstance import v2_MWCCPInstance
import sys


class MWCCPSolutionEGA(PermutationSolution, ABC):
    to_maximize = False

    x = []
    def __init__(self, inst: v2_MWCCPInstance):
        super().__init__(len(inst.get_instance()["v"]), init=True, inst=inst)
        self.x = inst.get_instance()["v"]
        self.instance_c_tup = inst.get_instance()["c_tup"]
        self.prior_obj_val = sys.maxsize
        self.instance_c = inst.get_instance()["c"]
        self.instance_edges = inst.get_instance()["edges"]

    def copy(self):
        sol = MWCCPSolutionEGA(self.inst)
        # sol.x = self.x
        sol.copy_from(self)
        return sol

    def copy_from(self, sol):
        super().copy_from(sol)
        #self.x = sol.x


    def calc_objective(self):
        cost = 0
        for i in range(len(self.x)):
            for j in range(i + 1, len(self.x)):
                v1, v2 = self.x[i], self.x[j]
                cost += self.inst.get_crossing_contribution(v1, v2)
        return cost

    def construct(self, _sol, _par, _res): #GA INIT STEP
        order = np.arange(self.inst.n)
        np.random.shuffle(order)
        x = self.check_order_constraints(order, construct=True)
        self.x = x
        current_obj_val = self.calc_objective()
        if self.is_better_obj(current_obj_val, self.prior_obj_val):
            self.obj_val = current_obj_val
        self.prior_obj_val = self.calc_objective()
        return x

    def crossover(self, parent1, parent2, _par): #recombine
        crossover_point = None
        if(_par != None): #parametertuning
            crossover_point = int(_par * self.inst.n)
        else:
            crossover_point = random.randint(0, self.inst.n)
        child1 = np.concatenate((parent1.x[:crossover_point], parent2.x[crossover_point:]))
        child2 = np.concatenate((parent2.x[:crossover_point], parent1.x[crossover_point:]))
        def replace_duplicates(child, parent):
            unique_values = set(child)
            missing_values = [val for val in parent if val not in unique_values]
            seen = set()
            for i in range(len(child)):
                if child[i] in seen:
                    # Replace duplicate with a missing value
                    child[i] = missing_values.pop(0)
                seen.add(child[i])
            return child

        child1 = replace_duplicates(child1, parent1.x)
        child2 = replace_duplicates(child2, parent2.x) # if child2  is needed
        child1 = self.check_order_constraints(child1, construct=True)
        self.x = child1
        return self.x

    def shaking(self, _sol, _par, _res): # GA mutate step
        for _ in range(_par):
            a = random.randint(0, self.inst.n - 1)
            b = random.randint(0, self.inst.n - 1)
            if not self.is_constraint_valid(a,b):
                continue
            else:
                self.x[a], self.x[b] = self.x[b], self.x[a]
                return self.x


    def local_improve(self, _sol, _par, _res): #GA Local Search // optional

        pass


    def check_order_constraints(self, order, construct):
        constraint_pairs = self.instance_c_tup
        arr = np.array([self.x[abs(i - self.inst.n) - 1] for i in np.nditer(order)])
        invalid_constraints = []

        # Collect all invalid constraints
        for a, b in constraint_pairs:
            indices_a = np.where(arr == a)[0]
            indices_b = np.where(arr == b)[0]
            if indices_a.size == 0 or indices_b.size == 0:
                continue
            index_a = indices_a[0]
            index_b = indices_b[0]
            if index_a > index_b:
                invalid_constraints.append((index_a, index_b))

        # Resolve all invalid constraints
        while invalid_constraints:
            swapped = False
            for index_a, index_b in invalid_constraints:
                arr[index_a], arr[index_b] = arr[index_b], arr[index_a]
                order[index_a], order[index_b] = order[index_b], order[index_a]
                swapped = True

            # Re-check constraints after swapping
            invalid_constraints = []
            for a, b in constraint_pairs:
                indices_a = np.where(arr == a)[0]
                indices_b = np.where(arr == b)[0]
                if indices_a.size == 0 or indices_b.size == 0:
                    continue
                index_a = indices_a[0]
                index_b = indices_b[0]
                if index_a > index_b:
                    invalid_constraints.append((index_a, index_b))

            if not swapped:
                break

        x = np.array([self.x[abs(i - self.inst.n) - 1] for i in np.nditer(order)])
        if construct:
            return x
        return order

    def check(self):
        x = self.x
        for node, constraint_nodes in self.instance_c.items():
            for v_prime in self.instance_c[node]:
                if list(x).index(node) >= list(x).index(v_prime):
                    raise ValueError(f"Constraint {node} {v_prime} violated.")


    def is_constraint_valid(self, p1, p2):
        x = self.x
        first = x[p1]
        second = x[p2]
        if bool(second in self.instance_c.keys()):
            if bool(first in self.instance_c[second]):
                return False
        return True

