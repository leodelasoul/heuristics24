import os
# https://optuna.org/
import optuna
from optuna.trial import TrialState
from algorithms.instance import MWCCPInstance
from algorithms.mmas import *

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00001')

def objective(trial, instance_files):
    """
    Objective function for Optuna.
    """
    # Suggest parameters
    alpha = trial.suggest_float("alpha", 1.0, 3.0)          # Influence of pheromones
    beta = trial.suggest_float("beta", 1.0, 3.0)            # Influence of heuristic
    rho = trial.suggest_float("rho", 0.05, 0.9)             # Evaporation rate
    num_ants = trial.suggest_int("num_ants", 20, 100)       # Number of ants
    num_iterations = trial.suggest_int("num_iterations", 50, 250)  # Number of iterations
    initial_tau = trial.suggest_int("initial_tau", 1, 10)   # Initial pheromone level
    reinit_threshold = trial.suggest_int("reinit_threshold", 10, 50)  # Stagnation threshold

    # Define MMAS parameters
    mmas_params = {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "num_ants": num_ants,
        "num_iterations": num_iterations,
        "initial_tau": initial_tau,
        "reinit_threshold": reinit_threshold
    }

    # Aggregate results across all instances
    total_cost = 0
    for instance_file in instance_files:
        # Load the instance
        instance = MWCCPInstance(instance_file)

        # Run MMAS
        solver = MMAS(instance, mmas_params)
        _, best_cost = solver.run()
        total_cost += best_cost

    # Return the average cost across all instances
    return total_cost / len(instance_files)

def tune_and_save_results(input_folder, output_file, n_trials=20):
    """
    Perform tuning for all instances in a folder and save results to a file.
    """
    results = []
    average_params = {}

    # Collect all instance files in the folder
    instance_files = [
        os.path.join(input_folder, file)
        for file in sorted(os.listdir(input_folder))
        if file.startswith("inst")  # Assuming instance files start with "inst"
    ]

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize parameters
    study.optimize(lambda trial: objective(trial, instance_files), n_trials=n_trials)

    # Get the best parameters and cost
    best_params = study.best_params
    best_cost = study.best_value


    # Save the results
    with open(output_file, "a") as f:
        f.write(os.path.basename(output_file) + "\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Best Average Cost: {best_cost}\n")

    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_folder = os.path.join(DIRNAME, '../tuning_instances/medium')  # Change this to the desired folder
    print(input_folder)
    output_file = os.path.join(DIRNAME, 'tuning_sol/tuning_results_medium')  # Change this to the desired output file
    tune_and_save_results(input_folder, output_file)
