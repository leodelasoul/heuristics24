import random
from .ant import Ant

import logging

# Configure logging to write only to a file
logging.basicConfig(
    level=logging.INFO,                # Set the logging level
    format="%(asctime)s - %(message)s",  # Format the log message with a timestamp
    filename="mmas.log",                # Specify the log file name
    filemode="w"                        # Overwrite the file each time the program runs
)

class MMAS:
    def __init__(self, U, V, constraints, edges, params):
        self.U = U
        self.V = V
        self.constraints = constraints
        self.edges = edges
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.rho = params["rho"]
        self.num_ants = params["num_ants"]
        self.num_iterations = params["num_iterations"]
        self.tau_min = params["tau_min"]
        self.tau_max = params["tau_max"]
        self.pheromones = {edge: self.tau_max for edge in edges}

    def run(self):
        global_best_solution = None
        global_best_cost = float('inf')

        for iteration in range(self.num_iterations):
            
            ants = [Ant(self.U, self.V, self.constraints, self.edges, self.pheromones, self.alpha, self.beta) for _ in range(self.num_ants)]

            iteration_best_solution = None
            iteration_best_cost = float('inf')
            costs = []

            for ant in ants:
                ant.construct_solution()
                costs.append(ant.cost)
                if ant.cost < iteration_best_cost and ant.cost != float('inf'):
                    iteration_best_solution = ant.solution
                    iteration_best_cost = ant.cost

            if iteration_best_solution is not None and iteration_best_cost < global_best_cost:
                global_best_solution = iteration_best_solution
                global_best_cost = iteration_best_cost

            if global_best_solution is not None:
                self._update_pheromones(global_best_solution, global_best_cost)

            avg_cost = sum(costs) / len(costs)
            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                logging.info(f"Iteration {iteration + 1}/{self.num_iterations}")
                logging.info(
                    f"Iteration {iteration + 1}: Best cost: {iteration_best_cost}, "
                    f"Avg cost: {avg_cost}, Global best cost: {global_best_cost}"
                )

        return global_best_solution, global_best_cost

    def _update_pheromones(self, best_solution, best_cost):
        # Evaporation
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)
            self.pheromones[edge] = max(self.pheromones[edge], self.tau_min)

        # Deposit pheromones for the best solution
        for v in best_solution:
            for u in self.U:
                if (u, v) in self.edges:
                    self.pheromones[(u, v)] += 1 / best_cost
                    self.pheromones[(u, v)] = min(self.pheromones[(u, v)], self.tau_max)
