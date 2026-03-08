import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np


def generate_group_size_histogram():
    # Configuration
    # Use path relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "weights_histogram")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all JSON files starting with LZG
    pattern = os.path.join(results_dir, "LZG*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} files.")

    for file_path in files:
        base_name = os.path.basename(file_path)
        base_name_no_ext = os.path.splitext(base_name)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract group_sizes
            group_sizes = data.get("group_sizes", [])

            if not isinstance(group_sizes, list) or not group_sizes:
                print(f"Skipping {base_name}: No valid group_sizes found.")
                continue

            # Plot histogram for this specific file
            plt.figure(figsize=(10, 6))

            # Using 10 bins or auto-adjust if fewer unique values
            bins = 10

            # Create histogram
            n, bins, patches = plt.hist(
                group_sizes, bins=bins, color="skyblue", edgecolor="black", alpha=0.7
            )

            plt.title(f"Distribution of Group Sizes: {base_name_no_ext}")
            plt.xlabel("Group Size (Number of Rows in Window)")
            plt.ylabel("Frequency")
            plt.grid(axis="y", alpha=0.5)

            # Add text labels on top of bars
            for i in range(len(n)):
                if n[i] > 0:
                    plt.text(
                        bins[i] + (bins[i + 1] - bins[i]) / 2,
                        n[i],
                        int(n[i]),
                        ha="center",
                        va="bottom",
                    )

            # Save with the same name as the JSON file
            output_file = os.path.join(output_dir, f"{base_name_no_ext}.png")
            plt.savefig(output_file)
            print(f"Generated histogram for {base_name} -> {output_file}")
            plt.close()

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    generate_group_size_histogram()
