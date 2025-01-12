import numpy as np
import random
from .ant import Ant
import logging

# Configure logging to write only to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filename="mmas.log",
    filemode="w"
)

class MMAS:
    def __init__(self, instance, params):
        """
        Initialize the MMAS algorithm.
        """
        self.instance = instance
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.rho = params["rho"]
        self.num_ants = params["num_ants"]
        self.num_iterations = params["num_iterations"]
        self.p = params.get("p", 0.05)  # Tuning parameter for tau_min
        self.initial_tau = params["initial_tau"]
        self.reinit_threshold = params.get("reinit_threshold", 20)

        self.best_cost = float('inf')
        self.best_solution = None
        self.stagnation_counter = 0

        # Initialize pheromone and heuristic matrices
        n = len(self.instance.V)
        self.pheromones = np.full((n, n), self.initial_tau)  # Initialize pheromones uniformly
        self.heuristics = self._compute_heuristic_matrix()

    def _compute_heuristic_matrix(self):
        """
        Compute the heuristic matrix η
        """
        n = len(self.instance.V)
        heuristics = np.zeros((n, n))

        #for i, v in enumerate(self.instance.V):
            # for j, v2 in enumerate(self.instance.V):
            #     if i != j:
            #         # Compute heuristic as inverse of crossing contribution
            #         weight = self.instance.get_crossing_contribution(v, v2)
            #         heuristics[i, j] = 1 / weight if weight > 0 else 0

        # for i, v in enumerate(self.instance.V):
        #     heuristics[i, :] = 1 / (1 + self.instance.in_degree[v])


        # Step 1: Compute average position of each node in V
        instance = self.instance
        averages = {}
        for v in instance.V:
            edges_to_v = instance.adjacent_v[v]  # List of edges connected to v (u, weight)
            if edges_to_v:
                total_position = sum(u for u, _ in edges_to_v)  # Sum of positions of nodes in U
                averages[v] = total_position / len(edges_to_v)  # Average position
            else:
                averages[v] = 0  # If no edges, set average to 0

        # Step 2: Fill the heuristic matrix based on relative positions
        for i, v1 in enumerate(instance.V):
            for j, v2 in enumerate(instance.V):
                if i != j:
                    # Compute heuristic as inverse of position difference + 1
                    heuristics[i, j] = 1 / (abs(averages[v1] - averages[v2]) + 1)

        

        return heuristics

    def _calculate_tau_bounds(self):
        """
        Dynamically calculate tau_min and tau_max based on the best cost.
        """
        tau_max = 1 / ((1 - self.rho) * self.best_cost)
        tau_min = tau_max * (1 - self.p ** (1 / len(self.instance.V))) / ((len(self.instance.V) / 2 - 1) * self.p ** (1 / len(self.instance.V)))
        return tau_min, tau_max

    def _calculate_probabilities(self, current_solution, available_nodes):
        """
        Calculate the transition probabilities for the current ant.
        """
        probabilities = []
        current_idx = self.instance.V.index(current_solution[-1])

        for v in available_nodes:
            v_idx = self.instance.V.index(v)
            pheromone = self.pheromones[current_idx, v_idx] ** self.alpha
            heuristic = self.heuristics[current_idx, v_idx] ** self.beta
            probabilities.append(pheromone * heuristic)

        total = sum(probabilities)
        if total == 0:
            return [1 / len(available_nodes)] * len(available_nodes)  # Equal probabilities if no guidance
        return [p / total for p in probabilities]

    def _update_pheromones(self, iteration_best_solution, iteration_best_cost):
        """
        Update pheromone levels using the iteration-best solution.
        """
        tau_min, tau_max = self._calculate_tau_bounds()

        # Evaporate pheromones
        self.pheromones *= (1 - self.rho)
        self.pheromones = np.maximum(self.pheromones, tau_min)

        # Deposit pheromones for the iteration-best solution
        for i in range(len(iteration_best_solution) - 1):
            v1 = self.instance.V.index(iteration_best_solution[i])
            v2 = self.instance.V.index(iteration_best_solution[i + 1])
            self.pheromones[v1, v2] += 1 / iteration_best_cost
            self.pheromones[v1, v2] = min(self.pheromones[v1, v2], tau_max)

    def _reinitialize_pheromones(self):
        """
        Reinitialize pheromones to tau_max in case of stagnation.
        """
        tau_max = 1 / ((1 - self.rho) * self.best_cost)
        self.pheromones.fill(tau_max)
        logging.info("Pheromones reinitialized due to stagnation.")

    def run(self):
        """
        Run the MMAS algorithm.
        """
        for iteration in range(self.num_iterations):
            ants = [Ant(self.instance, self.pheromones, self.heuristics, self.alpha, self.beta) for _ in range(self.num_ants)]

            iteration_best_solution = None
            iteration_best_cost = float('inf')

            for ant in ants:
                ant.construct_solution()
                if ant.cost < iteration_best_cost and ant.cost != float('inf'):
                    iteration_best_solution = ant.solution
                    iteration_best_cost = ant.cost

            # Update global best if a better solution is found
            if iteration_best_cost < self.best_cost:
                self.best_cost = iteration_best_cost
                self.best_solution = iteration_best_solution
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # Update pheromones if a feasible solution exists
            if iteration_best_solution is not None:
                self._update_pheromones(iteration_best_solution, iteration_best_cost)

            # Reinitialize pheromones if stagnation occurs
            if self.stagnation_counter >= self.reinit_threshold:
                self._reinitialize_pheromones()
                self.stagnation_counter = 0

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                logging.info(f"Iteration {iteration + 1}: Current Best Sol: {iteration_best_cost}, Best cost: {self.best_cost}")

        return self.best_solution, self.best_cost
