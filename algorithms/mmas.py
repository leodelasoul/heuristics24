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
    def __init__(self, instance, params):
        self.instance = instance
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.rho = params["rho"]
        self.num_ants = params["num_ants"]
        self.num_iterations = params["num_iterations"]
        self.p = params.get("p", 0.05)  # Tuning parameter for tau_min
        self.reinit_threshold = params.get("reinit_threshold", 20)
        #pheromone level on edge (i,j) (slides)
        self.pheromones = {edge: 1.0 for edge in instance.edges}
        self.best_cost = float('inf')
        self.best_solution = None
        self.stagnation_counter = 0

    def _calculate_tau_bounds(self):
        # Update tau_max and tau_min dynamically
        tau_max = 1 / ((1 - self.rho) * self.best_cost)
        tau_min = tau_max * (1 - self.p ** (1 / len(self.instance.V))) / ((len(self.instance.V) / 2 - 1) * self.p ** (1 / len(self.instance.V)))
        return tau_min, tau_max

    def _update_pheromones(self, iteration_best_solution, iteration_best_cost):
        tau_min, tau_max = self._calculate_tau_bounds()

        # Evaporate pheromones
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.rho)
            self.pheromones[edge] = max(self.pheromones[edge], tau_min)

        # Deposit pheromones for the iteration-best solution
        for v in iteration_best_solution:
            for u in self.instance.U:
                if (u, v) in self.pheromones:
                    self.pheromones[(u, v)] += 1 / iteration_best_cost
                    self.pheromones[(u, v)] = min(self.pheromones[(u, v)], tau_max)

    def _reinitialize_pheromones(self):
        tau_max = 1 / ((1 - self.rho) * self.best_cost)
        self.pheromones = {edge: tau_max for edge in self.instance.edges}
        logging.info("Pheromones reinitialized due to stagnation.")

    def run(self):
        for iteration in range(self.num_iterations):
            ants = [Ant(self.instance, self.pheromones, self.alpha, self.beta) for _ in range(self.num_ants)]
            
            iteration_best_solution = None
            iteration_best_cost = float('inf')
        
            for ant in ants:
                ant.construct_solution()
                if ant.cost < iteration_best_cost and ant.cost != float('inf'):
                    iteration_best_solution = ant.solution
                    iteration_best_cost = ant.cost

            # Update global best if better solution is found
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