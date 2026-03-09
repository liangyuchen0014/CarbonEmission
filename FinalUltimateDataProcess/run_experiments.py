import json
import os
import subprocess
import sys
import time


def separate_vehicle_prediction():
    # Configuration
    data_dir = "data"
    models = [
        "linear",
        "ridge",
        "lasso",
        "rf",
        "xgboost",
        "lightgbm",
        "polynomial",
        "gam",
        "woa_gam",
    ]

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    # Get all CSV files
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    files.sort()

    if not files:
        print(f"No CSV files found in '{data_dir}'.")
        return

    total_tasks = len(files) * len(models)
    print(f"Found {len(files)} CSV files.")
    print(f"Models to run: {', '.join(models)}")
    print(f"Total experiments to run: {total_tasks}")
    print("-" * 50)

    start_time = time.time()
    current_task = 0

    for i, filename in enumerate(files):
        # if filename not in ["LZGJLG843PX029028.csv", "LZGJL4849PX027735.csv"]:
        #     continue

        file_path = os.path.join(data_dir, filename)
        print(f"\nProcessing file {i + 1}/{len(files)}: {filename}")

        for model in models:
            current_task += 1
            print(
                f"  [{current_task}/{total_tasks}] Running model: {model} ... ",
                end="",
                flush=True,
            )

            cmd = [
                sys.executable,
                "src/main.py",
                "--source",
                file_path,
                "--model",
                model,
                "--save_result",
                "--save_plot",
                # "--random_seed",
                # "21",
            ]
            if model == "linear":
                cmd.append("--save_predict_result")
            try:
                # Run the command
                # capture_output=True to suppress stdout unless there's an error,
                # but main.py uses logging which goes to stderr/stdout.
                # Let's keep it visible or redirect to a log file?
                # User probably wants to see progress, but main.py logs might be noisy.
                # Let's suppress stdout/stderr for cleaner batch output, but print error if fails.
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print("Done.")
                else:
                    print("Failed!")
                    print(f"    Error output:\n{result.stderr}")

            except Exception as e:
                print(f"Error executing command: {e}")

    elapsed_time = time.time() - start_time
    print("-" * 50)
    print(f"All tasks completed in {elapsed_time:.2f} seconds.")


def category_weight_aggregated_prediction():
    models = [
        "linear",
        "ridge",
        "lasso",
        "rf",
        "xgboost",
        "lightgbm",
        "polynomial",
        "gam",
        "woa_gam",
    ]

    if not os.path.exists("data/category.json"):
        print("Error: 'data/category.json' not found.")
        return

    with open("data/category.json", "r") as f:
        category_weights = json.load(f)

        total_tasks = 0
        for category, weights in category_weights.items():
            total_tasks += len(weights) * len(models)

        print(f"Total aggregated experiments to run: {total_tasks}")
        current_task = 0

        for category, weights in category_weights.items():
            for weight, PINs in weights.items():
                print(
                    f"Category: {category}, Weight: {weight}, PINs count: {len(PINs)}"
                )

                # Construct comma-separated paths
                # Assuming PINs are filenames without extension, located in 'data/'
                pin_paths = [
                    os.path.join("data", f"{filename}.csv") for filename in PINs
                ]
                source_arg = ",".join(pin_paths)

                output_name = f"{category}_{weight}"

                for model in models:
                    current_task += 1
                    print(
                        f"  [{current_task}/{total_tasks}] Running model: {model} for {output_name} ... ",
                        end="",
                        flush=True,
                    )

                    cmd = [
                        sys.executable,
                        "src/main.py",
                        "--source",
                        source_arg,
                        "--model",
                        model,
                        "--save_result",
                        "--save_plot",
                        "--output_name",
                        output_name,
                    ]

                    if model == "linear":
                        cmd.append("--save_predict_result")

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)

                        if result.returncode == 0:
                            print("Done.")
                        else:
                            print("Failed!")
                            print(f"    Error output:\n{result.stderr}")

                    except Exception as e:
                        print(f"Error executing command: {e}")


if __name__ == "__main__":
    separate_vehicle_prediction()
    category_weight_aggregated_prediction()
