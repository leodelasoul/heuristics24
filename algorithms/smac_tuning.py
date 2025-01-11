import os
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from mmas import MMAS
from instance import MWCCPInstance
import time

def define_parameter_space():
    """
    Define the parameter space for SMAC3 optimization.
    """
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformFloatHyperparameter("alpha", 0.5, 3.0, default_value=1.0),
        UniformFloatHyperparameter("beta", 0.5, 3.0, default_value=2.0),
        UniformFloatHyperparameter("rho", 0.05, 0.5, default_value=0.1),
        UniformIntegerHyperparameter("num_ants", 10, 100, default_value=50),
        UniformIntegerHyperparameter("num_iterations", 50, 500, default_value=100),
        UniformIntegerHyperparameter("reinit_threshold", 10, 50, default_value=20),
    ])
    return cs

def smac_objective(config, instance_file, max_time_per_run=60):
    """
    Objective function for SMAC3 with a time limit per MMAS run.
    """
    alpha = config["alpha"]
    beta = config["beta"]
    rho = config["rho"]
    num_ants = config["num_ants"]
    num_iterations = config["num_iterations"]
    reinit_threshold = config["reinit_threshold"]

    instance = MWCCPInstance(instance_file)
    params = {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "num_ants": num_ants,
        "num_iterations": num_iterations,
        "p": 0.05,
        "reinit_threshold": reinit_threshold,
    }

    # Run MMAS with a timer
    start_time = time.time()
    solver = MMAS(instance, params)
    _, best_cost = solver.run()
    elapsed_time = time.time() - start_time

    # If the time exceeds the limit, return a large penalty
    if elapsed_time > max_time_per_run:
        print(f"Run exceeded time limit ({elapsed_time:.2f}s > {max_time_per_run}s). Penalizing.")
        return float('inf')

    return best_cost

def run_smac_tuning(base_dir, subfolders, time_limit_per_instance=600, time_limit_per_run=60):
    """
    Run SMAC3 tuning for all instance files in the specified subfolders with a global time limit.
    """
    results = {}
    cs = define_parameter_space()

    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        instance_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        for instance_file in instance_files:
            print(f"Tuning parameters for instance: {instance_file}")

            scenario = Scenario({
                "run_obj": "quality",       # We optimize the quality (cost)
                "runcount-limit": 100,      # Limit the number of evaluations per instance
                "cs": cs,                   # Configuration space
                "deterministic": "true",
                "wallclock-limit": time_limit_per_instance,  # Limit total tuning time per instance
            })

            def objective_function(config):
                return smac_objective(config, instance_file, max_time_per_run=time_limit_per_run)

            smac = SMAC(scenario=scenario, tae_runner=objective_function)
            best_config = smac.optimize()

            results[instance_file] = best_config
            print(f"Best configuration for {instance_file}: {best_config}")

    return results
