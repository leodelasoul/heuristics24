import random
from typing import Any

import numpy as np
from pymhlib.permutation_solution import PermutationSolution
from pymhlib.solution import TObj
import sys

from MWCCPInstance import MWCCPInstance


class MWCCPSolution(PermutationSolution):
    instance_w: list[list[int]]
    instance_u: list[int]
    instance_v: list[int]
    instance_c: dict[int, list[int]]
    instance_c_tup: list[(int, int)]
    instance_adj_v: dict[int, set[int]]
    instance_edges: list[(int, int, int)]

    to_maximize = False
    random_order = False
    x = None
    w = list[list[int]]
    prior_obj_val = sys.maxsize

    # def __init__(self, inst: MWCCPInstance):
    #     super().__init__(inst.n, inst=inst)
    #     self.obj_val_valid = False

    def __init__(self, inst: MWCCPInstance, init=True, **kwargs):
        super().__init__(len(inst.get_instance()["u"]), init=False, inst=inst)
        self.instance_w = inst.get_instance()["w"]
        self.instance_u = inst.get_instance()["u"]
        self.instance_v = inst.get_instance()["v"]
        self.instance_adj_v = inst.get_instance()["adj_v"]
        self.instance_c = inst.get_instance()["c"]
        self.instance_c_tup = inst.get_instance()["c_tup"]
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

    def construct(self, _par, _result):
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

    def is_constraint_valid(self, p1, p2):
        x = self.x
        first = x[p1]
        second = x[p2]
        if bool(second in self.instance_c.keys()):
            if bool(first in self.instance_c[second]):
                return False
        return True

    def check_order_constraints(self, order, construct):
        '''
        Used to construct a valid order after a move operator is applied, does not take delta evaluation into
        account, needs to be extended maybe
        :param order:
        :param construct:
        :return:
        '''
        # Repeat until all constraints are satisfied
        swapped = True
        #constraint_pairs = {(a, b) for a, values in self.instance_c.items() for b in
        #                    (values if isinstance(values, list) else [values])}
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

    def construct_random(self, _par, _result):
        '''
        construction heuristic that iteratively generates random orders of v
        :param _par:
        :param _result:
        :return:
        '''
        order = np.arange(self.inst.n)
        np.random.shuffle(order)
        x = self.check_order_constraints(order, construct=True)
        self.x = x
        current_obj_val = self.calc_objective()
        if self.is_better_obj(current_obj_val, self.prior_obj_val):
            self.obj_val = current_obj_val
        self.prior_obj_val = self.calc_objective()

    def local_improve(self, _par, _result):
        '''
        Scheduler Method for local search
        :param _par: 
        :param _result: 
        :return: returns True if a an improved solution is found
        '''''
        self.two_opt_neighborhood_search(False)

    def shaking(self, _par: Any, _result):
        '''
        Scheduler method for shaking picks random vertices and shuffles them unless they are within the constrain dict
        :param _par:
        :param _result:
        :return:
        '''
        for vert in range(_par):
            a = random.randint(0, self.inst.n - 1)
            b = random.randint(0, self.inst.n - 1)
            if self.is_constraint_valid(a,b):
                continue
            else:
                self.x[a], self.x[b] = self.x[b], self.x[a]
                self.invalidate()


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
        delta = self.obj() - obj
        x[p1], x[p2] = x[p2], x[p1]
        self.obj_val = obj
        return delta

    def one_opt_neighborhood_search(self, best: bool):
        x = self.x
        best_p1 = self.instance_edges
        for i in self.instance_v:
            if self.check(x):
                pass


    def two_opt_neighborhood_search(self, best: bool):
        n = self.inst.n
        best_sol = self.obj_val
        order = np.arange(n)
        x = self.x
        for idx, p1 in enumerate(order[:n - 1]):
            for p2 in order[idx + 1:]:
                order[p1], order[p2] = order[p2], order[p1]  # 2 opt move
                order = self.check_order_constraints(order,
                                                     construct=False)  # check if it was valid, if not rearrange invalids
                x = np.array([self.x[i] for i in np.nditer(order)])  # construct x out of our ordered indices
                self.x = x  # set current solution
                current_obj_val = self.calc_objective()
                if best and best_sol > current_obj_val:  # local search best improvement
                    best_sol = current_obj_val
                    self.obj_val = best_sol
                elif not best:
                    if self.prior_obj_val > current_obj_val:
                        self.obj_val = current_obj_val
                self.prior_obj_val = self.calc_objective()

        return False
    # def two_opt_neighborhood_search(self, best: bool):
    #     n = self.inst.n
    #     best_delta = 0
    #     best_p1 = None
    #     best_p2 = None
    #     order = np.arange(n)
    #     np.random.shuffle(order)
    #     order = self.check_order_constraints(order,construct=False)
    #     for idx, p1 in enumerate(order[:n - 1]):
    #         for p2 in order[idx + 1:]:
    #             order[p1], order[p2] = order[p2], order[p1]  # 2 opt move
    #             order = self.check_order_constraints(order,
    #                                                  construct=False)  # check if it was valid, if not rearrange invalids
    #             x = np.array([self.x[i] for i in np.nditer(order)])  # construct x out of our ordered indices
    #
    #             # consider exchange of positions p1 and p2
    #             if self.is_constraint_valid(p1, p2):
    #                 delta = self.two_exchange_move_delta_eval(p1, p2)
    #                 if self.is_better_obj(delta, best_delta):
    #                     if not best:
    #                         self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
    #                         self.obj_val += delta
    #                         return True
    #                     best_delta = delta
    #                     best_p1 = p1
    #                     best_p2 = p2
    #     if best_p1:
    #         self.x[best_p1], self.x[best_p2] = self.x[best_p2], self.x[best_p1]
    #         self.obj_val += best_delta
    #         return True
    #     return False

    def grasp(self):
        pass

    def check(self, *args):
        """
        Check if valid solution. All constraints must be satisfied.

        :raises ValueError: if problem detected.
        """
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
