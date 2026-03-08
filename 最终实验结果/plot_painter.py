import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set font to handle Chinese characters appropriately
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

target_dir_path = "outputs_带权重，sqrt权重均值化，X添加道路类型/results"

mape_model_statistics = []

best_models = {}
if os.path.isdir(target_dir_path):

    for file in os.listdir(target_dir_path):

        if file.endswith(".json") and not file.startswith("LZG"):
            # Add try-except and encoding for robustness
            try:
                with open(
                    os.path.join(target_dir_path, file), "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    min_mape = float("inf")
                    best_model = None

                    # Safely get results
                    results = data.get("results", [])
                    for result in results:
                        this_model = result.get("model")
                        metrics = result.get("full_model_metrics", {})
                        this_mape = metrics.get("mape")

                        if this_mape is not None and this_mape < min_mape:
                            min_mape = this_mape
                            best_model = this_model

                    if best_model:
                        if best_model in best_models:
                            best_models[best_model] += 1
                        else:
                            best_models[best_model] = 1
                        mape_model_statistics.append((file, best_model, min_mape))
            except Exception as e:
                print(f"Error reading {file}: {e}")

# --- Data Visualization (Box Plot with Jitter) ---
if mape_model_statistics:
    print(f"Total processed files: {len(mape_model_statistics)}")
    print(f"Best models distribution: {best_models}")

    # Convert to DataFrame
    df = pd.DataFrame(mape_model_statistics, columns=["Filename", "Best Model", "MAPE"])

    # Set theme
    sns.set_theme(style="ticks")
    plt.figure(figsize=(5, 4))

    # 1. Box Plot: Shows median, quartiles, and range
    # showfliers=False prevents plotting outliers twice (since stripplot shows all)
    # color='white' gives a clean box background
    ax = sns.boxplot(y="MAPE", data=df, color="white", showfliers=False, width=0.5)

    # 2. Jitter Plot (Stripplot): Overlays individual data points
    # hue maps color to 'Best Model'
    sns.stripplot(y="MAPE", data=df, hue="Best Model", jitter=True, size=7, alpha=1)

    plt.ylabel("MAPE (%)", fontsize=12)

    # Reference line at MAPE = 10%
    ax.axhline(10, color="red", linestyle="-", linewidth=1.5)

    # Move legend inside
    # plt.legend(title="Best Model", loc="upper right")
    plt.legend(title="Best Model", bbox_to_anchor=(1, 1), loc="upper left")

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_filename = "clustered_Optimal_MAPE_boxplot_jitter.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis plot saved to {output_filename}")

else:
    print("No valid statistics found for plotting.")
