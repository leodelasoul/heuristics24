import os
import sys
import csv
import time
from algorithms.mmas import MMAS
from algorithms.instance import MWCCPInstance

DIRNAME = os.path.dirname(__file__)
TEST_FOLDERS = {
    "small": os.path.join(DIRNAME, '../test_instances/small'),
    "medium": os.path.join(DIRNAME, '../test_instances/medium'),
    "medium_large": os.path.join(DIRNAME, '../test_instances/medium_large'),
    "large": os.path.join(DIRNAME, '../test_instances/large'),
}
OUTPUT_BASE = os.path.join(DIRNAME, 'test_compare')

# Predefined parameters for MMAS based on instance size
PARAMS = {
    "small": {
        "alpha": 1.8419192896947711,
        "beta": 2.4520290182502418,
        "rho": 0.10771225749644961,
        "num_ants": 97,
        "num_iterations": 842,
        "initial_tau": 8,
        "reinit_threshold": 42,
    },
    "default": {
        "alpha": 2.6,
        "beta": 1.6,
        "rho": 0.8,
        "num_ants": 36,
        "num_iterations": 229,
        "initial_tau": 4,
        "reinit_threshold": 20,
    },
}

def run_mmas_on_instances(folder, size, num_runs, output_folder, amount_of_files = 3):
    """
    Run MMAS on all instances in the folder and save results in CSV files.
    """
    counter = 0
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for instance_file in sorted(os.listdir(folder)):
        if not instance_file.startswith("inst"):  # Skip non-instance files
            continue
        
        instance_path = os.path.join(folder, instance_file)
        print(f"Processing instance: {instance_path}")
        instance = MWCCPInstance(instance_path)

        # Select parameters based on size
        params = PARAMS.get(size, PARAMS["default"])

        # Prepare CSV output
        csv_file = os.path.join(output_folder, f"{os.path.basename(instance_file)}.csv")
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["mmas", "mmas_time"])  # Add more columns if needed

            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                start_time = time.time()

                # Initialize and run MMAS solver
                solver = MMAS(instance, params)
                _, best_cost = solver.run()

                elapsed_time = time.time() - start_time

                # Write the results to CSV
                writer.writerow([best_cost, elapsed_time])
        
        if counter == amount_of_files:
            break
        counter += 1
            

        print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    
    size = "small"  # Change this to "medium", "medium_large", or "large" as needed

    if size not in TEST_FOLDERS:
        print(f"Invalid size '{size}'. Choose from: small, medium, medium_large, large")
        sys.exit(1)

    input_folder = TEST_FOLDERS[size]
    output_folder = os.path.join(OUTPUT_BASE, size)

    num_runs = 10 if size == "small" else 5

    run_mmas_on_instances(input_folder, size, num_runs, output_folder)

