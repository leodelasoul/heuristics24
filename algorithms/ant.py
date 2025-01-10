import random

class Ant:
    def __init__(self, U, V, constraints, edges, pheromones, alpha, beta):
        self.U = U
        self.V = V
        self.constraints = constraints
        self.edges = edges
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.solution = []
        self.cost = float('inf')

    def construct_solution(self):
        available_nodes = list(self.V)
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
                (self.pheromones[(u, v)] ** self.alpha) * (1 / self.edges[(u, v)]) ** self.beta
                for u in self.U if (u, v) in self.edges
            )
            probabilities.append(prob)
        total = sum(probabilities)
        return [p / total for p in probabilities]

    def _is_feasible(self):
        pos = {v: i for i, v in enumerate(self.solution)}
        return all(pos[v1] < pos[v2] for v1, v2 in self.constraints)

    def _calculate_cost(self):
        cost = 0
        edges_list = list(self.edges.keys())
        for i in range(len(edges_list)):
            for j in range(i + 1, len(edges_list)):
                (u1, v1) = edges_list[i]
                (u2, v2) = edges_list[j]
                if u1 < u2 and self.solution.index(v1) > self.solution.index(v2):
                    cost += self.edges[(u1, v1)] + self.edges[(u2, v2)]
        return cost
