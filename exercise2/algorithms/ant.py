import random
import numpy as np

class Ant:
    def __init__(self, instance, pheromones, heuristics, alpha, beta):
        """
        Initialize the Ant with problem instance data.
        """
        self.instance = instance
        self.pheromones = pheromones  # n x n matrix
        self.heuristics = heuristics  # n x n matrix
        self.alpha = alpha
        self.beta = beta
        self.solution = []
        self.cost = float('inf')

    def construct_solution(self):
        """
        Construct a solution by iteratively selecting nodes from V.
        """
        available_nodes = list(self.instance.V)
        self.solution = []

        #start at random node
        chosen_first = random.choice(available_nodes)
        self.solution.append(chosen_first)
        available_nodes.remove(chosen_first)


        # Start constructing the solution
        while available_nodes:
            probabilities = self._calculate_probabilities(available_nodes)
            chosen = random.choices(available_nodes, probabilities)[0]
            self.solution.append(chosen)
            available_nodes.remove(chosen)

        # Check feasibility and repair if necessary
        if not self._is_feasible():
            self._repair_solution()
            if not self._is_feasible():
                self.cost = float('inf')  # Mark solution as invalid
            else:
                self.cost = self._calculate_cost()
        else:
            self.cost = self._calculate_cost()

    def _calculate_probabilities(self, available_nodes):
        """
        Calculate the transition probabilities for selecting the next node.
        """
        probabilities = []
        current_idx = self.instance.V.index(self.solution[-1]) if self.solution else None

        for v in available_nodes:
            v_idx = self.instance.V.index(v)

            # If no current node (start), use uniform pheromone; otherwise, use the matrices
            pheromone = self.pheromones[current_idx, v_idx] if current_idx is not None else 1
            heuristic = self.heuristics[current_idx, v_idx] if current_idx is not None else 1
            probabilities.append((pheromone ** self.alpha) * (heuristic ** self.beta))


        probabilities = np.array(probabilities) / sum(probabilities)
  
        return probabilities

    def _is_feasible(self):
        """
        Check if the current solution satisfies all constraints.
        """
        pos = {v: i for i, v in enumerate(self.solution)}
        return all(pos[v1] < pos[v2] for v1, v2 in self.instance.constraints)

    def _repair_solution(self):
        """
        Repair the solution to make it feasible by satisfying all constraints.
        """
        pos = {v: i for i, v in enumerate(self.solution)}
        for v1, v2 in self.instance.constraints:
            if pos[v1] >= pos[v2]:
                self.solution.remove(v1)
                index_v2 = self.solution.index(v2)
                self.solution.insert(index_v2, v1)
                pos = {v: i for i, v in enumerate(self.solution)}

    def _calculate_cost(self):
        """
        Calculate the cost of the current solution based on crossing contributions.
        """
        cost = 0
        for i in range(len(self.solution)):
            for j in range(i + 1, len(self.solution)):
                v1, v2 = self.solution[i], self.solution[j]
                cost += self.instance.get_crossing_contribution(v1, v2)
        return cost
