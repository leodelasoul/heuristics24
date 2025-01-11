import random

class Ant:
    def __init__(self, instance, pheromones, alpha, beta):
        """
        Initialize the Ant with problem instance data.

        Args:
            instance (MWCCPInstance): Problem instance object.
            pheromones (dict): Current pheromone levels.
            alpha (float): Influence of pheromone.
            beta (float): Influence of heuristic.
        """
        self.instance = instance
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.solution = []
        self.cost = float('inf')

    def construct_solution(self):
        """
        Constructs a solution by iteratively selecting nodes from V based on pheromone
        and heuristic information.
        """
        available_nodes = list(self.instance.V)
        self.solution = []

        while available_nodes:
            probabilities = self._calculate_probabilities(available_nodes)
            chosen = random.choices(available_nodes, probabilities)[0]
            self.solution.append(chosen)
            available_nodes.remove(chosen)

        

        if self._is_feasible():
            self.cost = self._calculate_cost()

    def _calculate_probabilities(self, available_nodes):
        probabilities = []
        for v in available_nodes:
            prob = sum(
                (self.pheromones.get((u, v), 0) ** self.alpha) * (1 / self.instance.edges[(u, v)]) ** self.beta
                for u in self.instance.U if (u, v) in self.pheromones
            )
            probabilities.append(prob)
        
        total = sum(probabilities)
        if total == 0:
            # Assign equal probability if all pheromone and heuristic values are zero
            return [1 / len(available_nodes)] * len(available_nodes)
        
        # Normalize probabilities
        return [p / total for p in probabilities]


    def _is_feasible(self):
        pos = {v: i for i, v in enumerate(self.solution)}
        return all(pos[v1] < pos[v2] for v1, v2 in self.instance.constraints)

    def _calculate_cost(self):
        cost = 0
        for i in range(len(self.solution)):
            for j in range(i + 1, len(self.solution)):
                v1, v2 = self.solution[i], self.solution[j]
                cost += self.instance.get_crossing_contribution(v1, v2)
        return cost