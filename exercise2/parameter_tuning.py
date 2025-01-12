import os
from skopt import gp_minimize
from skopt.space import Real, Integer
from algorithms.instance import MWCCPInstance
from algorithms.mmas import *


DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00001')

def objective(params, instance_file):
    """
    Objective function for Scikit-Optimize.
    
    Args:
        params (list): List of parameter values [alpha, beta, rho, num_ants, num_iterations].
        instance_file (str): Path to the instance file.

    Returns:
        float: Best cost (to be minimized).
    """
    alpha, beta, rho, num_ants, num_iterations, initial_tau, reinit_threshold = params

    # Define MMAS parameters
    mmas_params = {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "num_ants": int(num_ants),
        "num_iterations": int(num_iterations),
        "initial_tau": int(initial_tau),  # Example initial pheromone level
        "reinit_threshold": int(reinit_threshold)  # Example stagnation threshold
    }

    # Load the instance
    instance = MWCCPInstance(instance_file)

    # Run MMAS
    solver = MMAS(instance, mmas_params)
    _, best_cost = solver.run()

    return best_cost

def tune_and_save_results(input_folder, output_file):
    """
    Perform tuning for all instances in a folder and save results to a file.

    Args:
        input_folder (str): Path to the folder containing instance files.
        output_file (str): Path to the output file for saving results.
    """
    # Define the parameter space for tuning
    param_space = [
        Real(0.5, 2.0, name="alpha"),          # Influence of pheromones
        Real(1.0, 3.0, name="beta"),           # Influence of heuristic
        Real(0.05, 0.5, name="rho"),           # Evaporation rate
        Integer(10, 100, name="num_ants"),     # Number of ants
        Integer(50, 500, name="num_iterations"),  # Number of iterations
        Integer(1, 10, name="initial_tau"),     # Initial pheromone level
        Integer(10, 100, name="reinit_threshold")  # Stagnation threshold
    ]

    results = []

    # Iterate through all instance files in the folder
    for instance_file in sorted(os.listdir(input_folder)):
        if instance_file.startswith("inst"):  # Assuming instance files have .txt extension
            full_path = os.path.join(input_folder, instance_file)
            print(f"Tuning parameters for instance: {full_path}")

            # Run the tuning process
            result = gp_minimize(
                func=lambda params: objective(params, full_path),
                dimensions=param_space,
                n_calls=50,
                random_state=42
            )

            # Store results
            results.append({
                "instance": instance_file,
                "best_params": result.x,
                "best_cost": result.fun
            })

    # Save all results to the output file
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Instance: {result['instance']}\n")
            f.write(f"Best Parameters: {result['best_params']}\n")
            f.write(f"Best Cost: {result['best_cost']}\n")
            f.write("\n")  # Separate entries with a newline

    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_folder = os.path.join(DIRNAME, '../tuning_instances/small') # Change this to the desired folder
    print(input_folder)
    output_file = os.path.join(DIRNAME, 'tuning_sol/tuning_results_small.txt')  # Change this to the desired output file
    tune_and_save_results(input_folder, output_file)
