import sys
import os
from algorithms.mmas import MMAS
from algorithms.instance import MWCCPInstance
#from algorithms.smac_tuning import run_smac_tuning

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00001')
FILENAME_COMPET_1: str = os.path.join(DIRNAME, '../competition_instances/inst_50_4_00001')
FILENAME_COMPET_2: str = os.path.join(DIRNAME, '../competition_instances/inst_200_20_00001')
FILENAME_COMPET_3: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
FILENAME_COMPET_4: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00012')
FILENAME_COMPET_5: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00021')

def main(input_file, output_file, input_size="small"):
        

    # Parse the problem instance
    instance = MWCCPInstance(input_file)

    # Parameters for MMAS
    params = {
        "alpha": 2.0,          # Influence of pheromone
        "beta": 1.0,           # Influence of heuristic
        "rho": 0.3,            # Evaporation rate
        "num_ants": 100,        # Number of ants
        "num_iterations": 1000, # Max iterations
        "initial_tau": 5.0,    # Initial pheromone level
        "reinit_threshold": 40 # Stagnation threshold
    }

    if input_size == "small":
        params  = {
            "alpha": 1.8419192896947711,
            "beta": 2.4520290182502418,
            "rho": 0.10771225749644961,
            "num_ants": 97,
            "num_iterations": 842,
            "initial_tau": 8,
            "reinit_threshold": 42
        }
    else:
        params = {
            "alpha": 2.2414470214334323,
            "beta": 2.5174274204112432,
            "rho": 0.22972339690248644,
            "num_ants": 100,
            "num_iterations": 83,
            "initial_tau": 7,
            "reinit_threshold": 14
        }



    # Initialize MMAS solver
    solver = MMAS(instance, params)

    # Solve the MWCCP instance
    best_solution, best_cost = solver.run()

    # Output the solution
    with open(output_file, "w") as f:
        f.write(os.path.basename(input_file) + "\n")
        f.write(" ".join(map(str, best_solution)) + "\n")
        f.write(f"Cost: {best_cost}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        input_size = "small"
        main(FILENAME_COMPET_1, "mmas_out", input_size)
        #print("Usage: python main.py <input_file> <output_file>")
    else:
        main(sys.argv[1], sys.argv[2])
