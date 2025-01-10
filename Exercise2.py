import sys
import os
from algorithms.mmas import MMAS
from algorithms.utils import parse_input

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, 'test_instances/small/inst_50_4_00001')
FILENAME_COMPET_1: str = os.path.join(DIRNAME, 'competition_instances/inst_50_4_00001')
FILENAME_COMPET_2: str = os.path.join(DIRNAME, 'competition_instances/inst_200_20_00001')
FILENAME_COMPET_3: str = os.path.join(DIRNAME, 'competition_instances/inst_500_40_00003')
FILENAME_COMPET_4: str = os.path.join(DIRNAME, 'competition_instances/inst_500_40_00012')
FILENAME_COMPET_5: str = os.path.join(DIRNAME, 'competition_instances/inst_500_40_00021')

def main(input_file, output_file):
    # Parse the problem instance
    U, V, constraints, edges = parse_input(input_file)

    # Parameters for MMAS
    params = {
        "alpha": 1.0,          # Influence of pheromone
        "beta": 2.0,           # Influence of heuristic
        "rho": 0.3,            # Evaporation rate
        "num_ants": 10,        # Number of ants
        "num_iterations": 1000, # Maximum iterations
        "tau_min": 0.1,        # Minimum pheromone
        "tau_max": 5.0,        # Maximum pheromone
    }

    # Initialize MMAS solver
    solver = MMAS(U, V, constraints, edges, params)

    # Solve the MWCCP instance
    best_solution, best_cost = solver.run()

    # Output the solution
    with open(output_file, "w") as f:
        f.write(os.path.basename(input_file) + "\n")
        f.write(" ".join(map(str, best_solution)) + "\n")
        f.write(f"Cost: {best_cost}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        main(FILENAME_COMPET_1, "mmas_out")
        #print("Usage: python main.py <input_file> <output_file>")
    else:
        main(sys.argv[1], sys.argv[2])
