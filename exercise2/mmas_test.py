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
OUTPUT_BASE = os.path.join(DIRNAME, 'test_compare_mmas')

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
        "alpha": 2.2414470214334323,
        "beta": 2.5174274204112432,
        "rho": 0.22972339690248644,
        "num_ants": 100,
        "num_iterations": 83,
        "initial_tau": 7,
        "reinit_threshold": 14
    }
}

def run_mmas_on_instances(folder, size, num_runs, output_folder, amount_of_files = 2, start_with_file = 0):
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

        if start_with_file < 0:
            start_with_file += 1
            continue

        # Select parameters based on size
        params = PARAMS.get(size, PARAMS["default"])

        # Prepare CSV output
        csv_file = os.path.join(output_folder, f"{os.path.basename(instance_file)}.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            if not file_exists:
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
    
    size = "medium"  # Change this to "medium", "medium_large", or "large" as needed

    if size not in TEST_FOLDERS:
        print(f"Invalid size '{size}'. Choose from: small, medium, medium_large, large")
        sys.exit(1)

    input_folder = TEST_FOLDERS[size]
    output_folder = os.path.join(OUTPUT_BASE, size)

    num_runs = 10 if size == "small" else 10

    run_mmas_on_instances(input_folder, size, num_runs, output_folder)
