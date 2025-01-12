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
    instance_crossing_contrib: dict[(int, int), int]

    to_maximize = False
    random_order = False
    x = None
    w = list[list[int]]
    prior_obj_val = sys.maxsize
    best_sol_found = None

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
        self.instance_crossing_contrib = inst.get_instance()["crossing_contrib"]
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
        vertices = list(self.instance_v)
        print(vertices)
        for v1 in self.instance_v:
            for v2 in self.instance_v:
                pos_v1 = position[v1]
                pos_v2 = position[v2]
                if pos_v1 == pos_v2:
                    continue
                if pos_v1 < pos_v2: #only check edges once, we iterate over all edges
                    objective_value += self.instance_crossing_contrib[(v1,v2)]

        # Iterate over all unique pairs (upper triangular traversal)
        # for i in range(len(vertices)):
        #     v1 = vertices[i]
        #     for j in range(i + 1, len(vertices)):  # Start the inner loop after the current index
        #         v2 = vertices[j]
        #         print(v1, v2)
        #         pos_v1 = position[v1]
        #         pos_v2 = position[v2]

        #         if pos_v1 < pos_v2:
        #             objective_value += self.instance_edge_crossing[(v1, v2)]
        #         else:
        #             objective_value += self.instance_edge_crossing[(v2, v1)]

        return objective_value
    
    def construct(self, _par, _result):
        """
        Deterministic construction heuristic that adjusts node positions
        based on edge weights and constraints.

        Args:
            _par: Additional parameters (if any).
            _result: Result object for storing intermediate results (if any).
        """
        # Step 1: Compute weighted average position of each node in V
        averages = {}
        V = list(self.instance_v)
        for v in V:
            edges_to_v = self.instance_adj_v[v]  # Get edges connected to v
            if edges_to_v:
                # Calculate weighted average position for node v
                total_position = 0
                total_weight = 0
                highest_weight = float('-inf')
                highest_position = None

                for u in edges_to_v:
                    weight = self.instance_w[u][v]
                    total_position += u * weight
                    total_weight += weight

                    # Track the highest weight and corresponding position
                    if weight > highest_weight:
                        highest_weight = weight
                        highest_position = u

                # Weighted average with priority for the highest weight edge
                if total_weight > 0:
                    averages[v] = (total_position / total_weight) * 0.8 + highest_position * 0.2
                else:
                    averages[v] = 0
            else:
                averages[v] = 0

        # Step 2: Sort V by weighted average position
        sorted_V = sorted(V, key=lambda v: averages[v])

        # Step 3: Resolve constraint violations
        sorted_V = np.array(list(sorted_V)) - (self.inst.n + 1)
        sorted_V = self.check_order_constraints(sorted_V, construct=True)

        # Store the sorted solution
        self.x = sorted_V
        self.invalidate()
        
    # random construction heuristic
    def construct_random(self, _par, _result):
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
        constraint_pairs = self.instance_c_tup
        arr = np.array([self.x[i] for i in np.nditer(order)])
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
    
    #check if change of positions is valid
    def is_constraint_valid(self, p1, p2):
        x = self.x
        first = x[p1]
        second = x[p2]
        if bool(second in self.instance_c.keys()):
            if bool(first in self.instance_c[second]):
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
                # shift node from p1 to position p2
                p1, p2 = np.random.choice(order, 2, replace=False)
                original_order = order
                node_pos = original_order[p1]
                order = np.delete(order, p1) # Remove node from position p1
                order = np.insert(order, p2, node_pos) # Insert node at position p2
                while self.check_order_constraints_bool(order) == False:
                    #swap back to original positions
                    order = original_order
                    # shift node from p1 to position p2
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

        # Get the sublist of nodes between p1 and p2
        sublist_V = sol.x[p1:p2+1]

        # Get the positions of the nodes in the sublist
        position = {v: i for i, v in enumerate(self.x)}

        # Calculate delta_old using the crossing_contrib matrix
        for i in range(len(sublist_V)):
            for j in range(i + 1, len(sublist_V)):
                v1 = sublist_V[i]
                v2 = sublist_V[j]
                if position[v1] < position[v2]:
                    delta_old += self.instance_crossing_contrib[(v1, v2)]
                else:
                    delta_old += self.instance_crossing_contrib[(v2, v1)]

        # Swap the positions of the nodes in the sublist
        temp = position[sublist_V[-1]]
        position[sublist_V[-1]] = position[sublist_V[0]]
        position[sublist_V[0]] = temp

        # Calculate delta_new using the crossing_contrib matrix
        for i in range(len(sublist_V)):
            for j in range(i + 1, len(sublist_V)):
                v1 = sublist_V[i]
                v2 = sublist_V[j]
                if position[v1] < position[v2]:
                    delta_new += self.instance_crossing_contrib[(v1, v2)]
                else:
                    delta_new += self.instance_crossing_contrib[(v2, v1)]

        new_obj = obj + delta_new - delta_old
       
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
        for u1, v1 in self.instance_edges:
            for u2, v2 in self.instance_edges:
                # Check for crossings where at least one edge involves the shifted node
                if v1 == node_to_shift or v2 == node_to_shift:
                    if position[u1] < position[u2] and position[v1] > position[v2]:
                        delta_old += self.instance_crossing_contrib[(u1, v1)] + self.instance_crossing_contrib[(u2, v2)]

        # Simulate the shift by removing the node and inserting it at the new position
        temp_x = list(x)  # Copy of the current order
        node = temp_x.pop(p1)  # Remove the node from position p1
        temp_x.insert(p2, node)  # Insert the node at position p2

        # Update positions after the shift
        new_position = {v: i for i, v in enumerate(temp_x)}

        # Compute delta_new: crossings caused by edges involving the node at its new position
        for u1, v1 in self.instance_edges:
            for u2, v2 in self.instance_edges:
                if v1 == node_to_shift or v2 == node_to_shift:
                    if new_position[u1] < new_position[u2] and new_position[v1] > new_position[v2]:
                        delta_new += self.instance_crossing_contrib[(u1, v1)] + self.instance_crossing_contrib[(u2, v2)]

        # Calculate the new objective value
        new_obj = obj + delta_new - delta_old

        return new_obj if new_obj < obj else obj

    # shaking methode for gvns
    def shaking_swap(self, par: Any, _result):
        for _ in range(par):
            a = random.randint(0, self.inst.n - 1)
            b = random.randint(0, self.inst.n - 1)
            if not self.is_constraint_valid(a,b):
                continue
            else:
                self.x[a], self.x[b] = self.x[b], self.x[a]

        self.invalidate()

    # shaking methode for gvns
    def shaking_shift(self, par: Any, _result):
        for _ in range(par):
            order = np.arange(self.inst.n)
            p1, p2 = np.random.choice(order, 2, replace=False)
            original_order = order
            node_pos = original_order[p1]
            order = np.delete(order, p1) # Remove node from position p1
            order = np.insert(order, p2, node_pos) # Insert node at position p2
            while self.check_order_constraints_bool(order) == False:
                #swap back to original positions
                order = original_order
                # shift node from p1 to position p2
                p1, p2 = np.random.choice(order, 2, replace=False) #get two random positions
                node_pos = original_order[p1]
                order = np.delete(order, p1) 
                order = np.insert(order, p2, node_pos) 

            #change order of copy x
            self.x = np.array([self.x[idx] for idx in np.nditer(order)])
        self.invalidate()
    
    #just take initial order as first construction heuristic (is not really used in grasp because construction is in loop)
    def construct_grasp(self, _par, _result):
        self.x = np.array(list(self.instance_v))

    def random_greedy_construction(self, alpha):
        solution = []  # Final solution (permutation of V)
        remaining_nodes = set(self.instance_v)  # Nodes to be added to the solution

        while remaining_nodes:
            # Step 1: Build the candidate list (CL)
            candidate_list = list(remaining_nodes)

            # Compute greedy values for all candidates (like deterministic construction: based on average position)
            greedy_values = {}
            for node in candidate_list:
                edges_to_node = self.instance_adj_v[node]
                if edges_to_node:
                    total_position = sum(u for u in edges_to_node)
                    greedy_values[node] = total_position / len(edges_to_node)
                else:
                    greedy_values[node] = float('inf')  # Less desirable if no edges

            # Step 2: Determine threshold for the RCL
            min_value = min(greedy_values.values())
            max_value = max(greedy_values.values())
            threshold = min_value + alpha * (max_value - min_value)

            # Step 3: Build the RCL
            restricted_candidate_list = [
                node for node in candidate_list if greedy_values[node] <= threshold
            ]

            # Step 4: Select a candidate randomly from the RCL
            selected_node = random.choice(restricted_candidate_list)

            # Step 5: Add the selected node to the solution and update remaining nodes
            solution.append(selected_node)
            remaining_nodes.remove(selected_node)

            # Step 6: Resolve any constraints (optional, based on your problem's requirements)
            for constraint in self.instance_c.get(selected_node, []):
                if constraint in remaining_nodes:
                    remaining_nodes.remove(constraint)
                    solution.append(constraint)

        # Step 7: Update the solution
        self.x = np.array(solution)
        self.invalidate()
        


    #use local search always on random solution
    def grasp(self, _par, _result):
        self.random_greedy_construction(_par, _result) # random (but greedy part is missing)
        #best improvement used for local search (lecture)
        self.ls_two_swap_best(_par, _result) # local search


    #check if solution is valid
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

    
    


