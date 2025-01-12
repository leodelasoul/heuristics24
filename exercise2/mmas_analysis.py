import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.mmas import MMAS
from algorithms.instance import MWCCPInstance

DIRNAME = os.path.dirname(__file__)
OUTPUT_FOLDER = os.path.join(DIRNAME, 'analysis/analysis_outputs')
PLOTS_FOLDER = os.path.join(DIRNAME, 'analysis/analysis_plots')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

FILENAME_COMPET_1: str = os.path.join(DIRNAME, '../competition_instances/inst_50_4_00001')
FILENAME_COMPET_2: str = os.path.join(DIRNAME, '../competition_instances/inst_200_20_00001')
FILENAME_COMPET_3: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
FILENAME_COMPET_4: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00012')
FILENAME_COMPET_5: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00021')

def run_experiment(instance_file, params, output_file, plot_prefix):
    # Parse the instance
    instance = MWCCPInstance(instance_file)


    # Metrics to track
    best_costs = []
    iteration_list = []
    runtimes = []
    if instance_name == "inst_50_4_00001":
        iteration_amount = 10
    else:
        iteration_amount = 5
    params["num_iterations"] = 5

    # Run the algorithm
    for iteration in range(iteration_amount):

        iteration_list.append(params["num_iterations"])
        # Initialize the MMAS solver
        solver = MMAS(instance, params)

        start_time = time.time()
        best_solution, best_cost = solver.run()
        runtimes.append(time.time() - start_time)

        # Record metrics
        best_costs.append(best_cost)
        #avg_cost = sum(ant.cost for ant in ants) / len(ants)
        #average_costs.append(avg_cost)
        params["num_iterations"] += 50

    # Save metrics to CSV
    metrics = pd.DataFrame({
        "Iteration": iteration_list,
        "BestCost": best_costs,
        "Runtime": runtimes,
    })
    metrics.to_csv(output_file, index=False)

    # Plot: Best and Average Costs Over Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["Iteration"], metrics["BestCost"], label="Best Cost", color="blue")
    #plt.plot(metrics["Iteration"], metrics["AverageCost"], label="Average Cost", color="orange")
    plt.title("Best Costs Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plot_file = os.path.join(PLOTS_FOLDER, f"{plot_prefix}_costs.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Cost plot saved to {plot_file}")

    # Plot: Runtime Per Iteration
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["Iteration"], metrics["Runtime"], label="Runtime", color="green")
    plt.title("Runtime Per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    plot_file = os.path.join(PLOTS_FOLDER, f"{plot_prefix}_runtime.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Runtime plot saved to {plot_file}")

# Example usage
if __name__ == "__main__":

    # Example instances
    instance_files = [
        FILENAME_COMPET_1,
        FILENAME_COMPET_2,
        FILENAME_COMPET_3,
        FILENAME_COMPET_4,
        FILENAME_COMPET_5
    ]

    # Run experiments on each instance
    for instance_file in instance_files:
        instance_name = os.path.basename(instance_file).split(".")[0]
        output_file = os.path.join(OUTPUT_FOLDER, f"{instance_name}_metrics.csv")
        plot_prefix = instance_name
        if instance_name == "inst_50_4_00001":
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
        run_experiment(instance_file, params, output_file, plot_prefix)
