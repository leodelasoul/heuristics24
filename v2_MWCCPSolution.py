import random
from typing import Any

import numpy as np
from pymhlib.permutation_solution import PermutationSolution
from pymhlib.solution import TObj
import sys

from v2_MWCCPInstance import v2_MWCCPInstance

class v2_MWCCPSolution(PermutationSolution):
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

    #neighborhood index for VND
    vnd_neighborhood_index = 0

    def __init__(self, inst: v2_MWCCPInstance, init=True, **kwargs):
        super().__init__(len(inst.get_instance()["u"]), init=False, inst=inst)
        self.instance_w = inst.get_instance()["w"]
        self.instance_u = inst.get_instance()["u"]
        self.instance_v = inst.get_instance()["v"]
        self.instance_adj_v = inst.get_instance()["adj_v"]
        self.instance_c = inst.get_instance()["c"]
        self.instance_c_tup = inst.get_instance()["c_tup"]
        inst.n = len(self.instance_v)

        self.instance_edges = inst.get_instance()["edges"]

        if init:
            self.x = np.array(list(self.instance_v))

        self.obj_val_valid = False

    #copy solution instance to copy_solution
    def copy(self):
        sol = v2_MWCCPSolution(self.inst)
        sol.copy_from(self)
        return sol

    #calculate objective value
    def calc_objective(self):
        objective_value = 0
        position = {v: i for i, v in enumerate(self.x)}
        for (u1, v1, weight) in self.instance_edges:
            for (u2, v2, weights) in self.instance_edges:
                if u1 < u2 and position[v1] > position[v2]:
                    objective_value += weight + weights
        return objective_value
    
    #deterministic construction heuristic
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

    # random construction heuristic
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

    # check if new order is valid
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
    
    #return true if new order of x is valid
    def check_order_constraints_bool(self, order):
        constraint_pairs = self.instance_c_tup
        arr = np.array([self.x[i] for i in np.nditer(order)])
        position = {node: idx for idx, node in enumerate(arr)}
        for v, v_prime in constraint_pairs:
            if position[v] > position[v_prime]:
                return False
        return True

    #local search two exchange neighborhood
    def ls_two_swap_best(self, _par, _result):
        self.local_improve_swap("best")

    def ls_two_swap_first(self, _par, _result):
        self.local_improve_swap("first")

    def ls_two_swap_random(self, _par, _result):
        self.local_improve_swap("random")

    #local search shift neighborhood

    def ls_shift_best(self, _par, _result):
        self.local_improve_shift("best")

    def ls_shift_first(self, _par, _result):
        self.local_improve_shift("first")

    def ls_shift_random(self, _par, _result):
        self.local_improve_shift("random")

    #local search reverse subinterval neighborhood

    def ls_reverse_best(self, _par, _result):
        pass

    def ls_reverse_first(self, _par, _result):
        pass

    def ls_reverse_random(self, _par, _result):
        pass
    
    # local search for two exchange neighborhood
    def local_improve_swap(self, step):
        sol = self.copy()
        n = self.inst.n
        order = np.arange(n) #to swap every possible position
        x = sol.x # working on copy
        current_obj_val = sol.obj_val

        if step == "random":
            #search through neighborhood randomly
            #stop search after 10*n iterations
            is_better_obj = False
            counter = 0

            while not is_better_obj:
                p1, p2 = np.random.choice(order, 2, replace=False)
                order[p1], order[p2] = order[p2], order[p1] #swap positions
                while self.check_order_constraints_bool(order) == False:
                    #swap back to original positions
                    order[p1], order[p2] = order[p2], order[p1]
                    # get two new positions
                    p1, p2 = np.random.choice(order, 2, replace=False) #get two random positions
                    order[p1], order[p2] = order[p2], order[p1]

                #change order of copy x
                x[p1], x[p2] = x[p2], x[p1]
                #check if new order is better
                delta_obj = self.calc_delta_obj_swap(p1, p2, sol)
                if delta_obj < current_obj_val:
                    #change actual solution
                    current_obj_val = delta_obj
                    self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
                    self.obj_val = current_obj_val
                    is_better_obj = True     
                else:
                    #change copy back to original order
                    x[p1], x[p2] = x[p2], x[p1]
                    #change order back to original order
                    order[p1], order[p2] = order[p2], order[p1]
                    counter += 1
                    #and do random swap again
                
                if counter > 10*n:
                    break

            return is_better_obj

        else:
            best_sol = sol.obj_val
            best_p1 = None
            best_p2 = None
            if step == "first":
                best = False
            else:
                best = True
            #search systematically

            for idx, p1 in enumerate(order[:n - 1]):
                for p2 in order[idx + 1:]:
                    #change order
                    order[p1], order[p2] = order[p2], order[p1]
                    if self.check_order_constraints_bool(order) == False:
                        #order not valid -> swap back and continue
                        order[p1], order[p2] = order[p2], order[p1]
                        continue
                    else:
                        #order valid -> check if new order is better
                        #change order of copy x
                        x[p1], x[p2] = x[p2], x[p1]
            
                        delta_sol = self.calc_delta_obj_swap(p1, p2, sol)
                        if self.is_better_obj(delta_sol, best_sol):

                            current_obj_val = delta_sol
                            if best and best_sol > current_obj_val:  # best improvement
                                best_sol = current_obj_val
                                best_p1 = p1
                                best_p2 = p2
                            else: # first improvement
                                #self.copy_from(sol)
                                self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
                                self.obj_val = current_obj_val
                                return True
                    
                    #change copy x and order back to original order and continue
                    x[p1], x[p2] = x[p2], x[p1]
                    order[p1], order[p2] = order[p2], order[p1]
                                                
            if best_p1:
                self.x[best_p1], self.x[best_p2] = self.x[best_p2], self.x[best_p1]
                self.obj_val = best_sol
                return True
            

        return False
    
    # local search for two exchange neighborhood
    def local_improve_shift(self, step):
        sol = self.copy()
        n = self.inst.n
        order = np.arange(n) #to shift every possible position
        x = sol.x # working on copy
        current_obj_val = sol.obj_val

        if step == "random":
            #search through neighborhood randomly
            #stop search after 10*n iterations
            is_better_obj = False
            counter = 0

            while not is_better_obj:
                p1, p2 = np.random.choice(order, 2, replace=False)
                original_order = order
                node_pos = original_order[p1]
                order = np.delete(order, p1) # Remove node from position p1
                order = np.insert(order, p2, node_pos) # Insert node at position p2
                while self.check_order_constraints_bool(order) == False:
                    #swap back to original positions
                    order = original_order
                    # get two new positions
                    p1, p2 = np.random.choice(order, 2, replace=False) #get two random positions
                    node_pos = original_order[p1]
                    order = np.delete(order, p1) 
                    order = np.insert(order, p2, node_pos) 

                #change order of copy x
                x = np.array([self.x[idx] for idx in np.nditer(order)])
                
                #check if new order is better
                delta_obj = self.calc_delta_obj_shift(p1, p2, sol)
                if delta_obj < current_obj_val:
                    #change actual solution
                    current_obj_val = delta_obj
                    self.x = x
                    self.obj_val = current_obj_val
                    is_better_obj = True     
                else:
                    #change copy back to original order
                    x = np.array([self.x[idx] for idx in np.nditer(original_order)])
                    #change order back to original order
                    order = original_order
                    counter += 1
                    #and do random swap again
                
                if counter > 10*n:
                    break

            return is_better_obj

        else:
            best_sol = sol.obj_val
            best_p1 = None
            best_p2 = None
            if step == "first":
                best = False
            else:
                best = True
            #search systematically

            for p1 in order:
                for p2 in range(n):
                    shifted_order = order
                    # Create a shifted order
                    shifted_order = np.delete(shifted_order, p1)  # Remove node at position p1
                    shifted_order = np.insert(shifted_order, p2, p1)  # Insert node at position p2

                    if self.check_order_constraints_bool(shifted_order) == False:
                        continue
                    else:
                        #order valid -> check if new order is better
                        #change order of copy x
                        x = np.array([self.x[idx] for idx in np.nditer(shifted_order)])
        
            
                        delta_sol = self.calc_delta_obj_shift(p1, p2, sol)
                        if self.is_better_obj(delta_sol, best_sol):

                            current_obj_val = delta_sol
                            if best and best_sol > current_obj_val:  # best improvement
                                best_sol = current_obj_val
                                best_p1 = p1
                                best_p2 = p2
                            else: # first improvement
                                #self.copy_from(sol)
                                self.x = x
                                self.obj_val = current_obj_val
                                return True
                    
                    #change copy x and order back to original order and continue
                    x = np.array([self.x[idx] for idx in np.nditer(order)])
                                                
            if best_p1:
                shifted_order = order
                shifted_order = np.delete(shifted_order, best_p1)  # Remove node at position p1
                shifted_order = np.insert(shifted_order, best_p2, best_p1)  # Insert node at position p2
                self.x = np.array([self.x[idx] for idx in np.nditer(shifted_order)])
                self.obj_val = best_sol
                return True
            

        return False  

    #calculate delta objective value for swap move
    def calc_delta_obj_swap(self, p1, p2, sol):
        delta_old = 0
        delta_new = 0
        obj = sol.obj()

        if p1 > p2:
            p1, p2 = p2, p1

        sublist_V = sol.x[p1:p2+1]
        sublist_U = np.arange(p1 + 1, p2 + 2) #nodes "on top of" the nodes in sublist

        if len(sublist_V) != len(sublist_U):
            raise ValueError("Sublists are not of the same length")
        
        #positions in self.x of the nodes in the sublist
        position = {v: i for i, v in enumerate(self.x)}

        affected_edges = [(u1, v1, weight) for (u1, v1, weight) in self.instance_edges if v1 in sublist_V or u1 in sublist_U]

        for (u1, v1, w1) in affected_edges:
            for (u2, v2, w2) in affected_edges:
                #edges where at most least one node is in the sublist
                if u1 < u2 and position[v1] > position[v2]:    
                        delta_old += w1 + w2

        #change positons of the nodes in the sublist (delta evaluation)
        temp = position[sublist_V[-1]]
        position[sublist_V[-1]] = position[sublist_V[0]]
        position[sublist_V[0]] = temp

        for (u1, v1, w1) in affected_edges:
            for (u2, v2, w2) in affected_edges:
                #edges where at most least one node is in the sublist
                if u1 < u2 and position[v1] > position[v2]:    
                        delta_new += w1 + w2

        new_obj = obj + delta_new - delta_old
        if new_obj < obj:
            pass
        return new_obj if new_obj < obj else obj
    
    def calc_delta_obj_shift(self, p1, p2, sol):
        delta_old = 0
        delta_new = 0
        obj = sol.obj()  # Current objective value
        x = sol.x  # Current order

        # Identify the node to be shifted and its neighbors
        node_to_shift = x[p1]
        old_neighbors = []
        if p1 > 0:
            old_neighbors.append((x[p1 - 1], node_to_shift))  # Left neighbor
        if p1 < len(x) - 1:
            old_neighbors.append((node_to_shift, x[p1 + 1]))  # Right neighbor

        # Compute delta_old: crossings caused by edges involving the node at its old position
        position = {v: i for i, v in enumerate(self.x)}  # Positions of all nodes in self.x
        for u1, v1, w1 in self.instance_edges:
            for u2, v2, w2 in self.instance_edges:
                # Check for crossings where at least one edge involves the shifted node
                if v1 == node_to_shift or v2 == node_to_shift:
                    if u1 < u2 and position[v1] > position[v2]:
                        delta_old += w1 + w2

        # Simulate the shift by removing the node and inserting it at the new position
        temp_x = list(x)  # Copy of the current order
        node = temp_x.pop(p1)  # Remove the node from position p1
        temp_x.insert(p2, node)  # Insert the node at position p2

        # Update positions after the shift
        new_position = {v: i for i, v in enumerate(temp_x)}

        # Compute delta_new: crossings caused by edges involving the node at its new position
        for u1, v1, w1 in self.instance_edges:
            for u2, v2, w2 in self.instance_edges:
                if v1 == node_to_shift or v2 == node_to_shift:
                    if u1 < u2 and new_position[v1] > new_position[v2]:
                        delta_new += w1 + w2

        # Calculate the new objective value
        new_obj = obj + delta_new - delta_old

        return new_obj if new_obj < obj else obj

    
    #just take initial order as first construction heuristic (is not really used in grasp because construction is in loop)
    def construct_grasp(self, _par, _result):
        self.x = np.array(list(self.instance_v))

    def random_greedy_construction(self, _par, _result):
        self.construct_random(_par, _result) #should be implemented that it constructs solution with candidate list

    #use local search always on random solution
    def grasp(self, _par, _result):
        self.random_greedy_construction(_par, _result) # random (but greedy part is missing)
        #best improvement used for local search (lecture)
        self.ls_two_swap_best(_par, _result) # local search


    #check if solution is valid
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

    
    


