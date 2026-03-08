import os
import json
import numpy as np
import pandas as pd


data_dir = "outputs_带权重，sqrt权重均值化，X添加道路类型/results"

canonical_models = [
    "GAM",
    "polynomial",
    "linear",
    "ridge",
    "lasso",
    "LightGBM",
    "XGBoost",
    "RandomForest",
]

alias_to_canonical = {
    "gam": "GAM",
    "polynomial": "polynomial",
    "poly": "polynomial",
    "linear": "linear",
    "ridge": "ridge",
    "lasso": "lasso",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "randomforest": "RandomForest",
    "random_forest": "RandomForest",
    "rf": "RandomForest",
}

model_metrics = {
    model: {"mape": [], "rmse": [], "r2": []} for model in canonical_models
}

best_counts = {model: {"mape": 0, "rmse": 0, "r2": 0} for model in canonical_models}

failed_r2_counts = {model: 0 for model in canonical_models}


def normalize_model_name(name: str) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    return alias_to_canonical.get(key)


if os.path.isdir(data_dir):
    for file in os.listdir(data_dir):
        if not (file.endswith(".json") and file.startswith("LZG")):
            continue
        try:
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        results = data.get("results", [])
        per_file_metrics = {
            model: {"mape": None, "rmse": None, "r2": None}
            for model in canonical_models
        }

        for result in results:
            model_name = normalize_model_name(result.get("model"))
            if model_name not in per_file_metrics:
                continue
            metrics = result.get("full_model_metrics", {})
            mape = metrics.get("mape")
            rmse = metrics.get("rmse")
            r2 = metrics.get("r2")

            per_file_metrics[model_name]["mape"] = mape
            per_file_metrics[model_name]["rmse"] = rmse
            per_file_metrics[model_name]["r2"] = r2

            if mape is not None:
                model_metrics[model_name]["mape"].append(mape)
            if rmse is not None:
                model_metrics[model_name]["rmse"].append(rmse)
            if r2 is not None:
                model_metrics[model_name]["r2"].append(r2)
                if r2 < 0:
                    failed_r2_counts[model_name] += 1

        mape_candidates = [
            (model, v["mape"])
            for model, v in per_file_metrics.items()
            if v["mape"] is not None
        ]
        rmse_candidates = [
            (model, v["rmse"])
            for model, v in per_file_metrics.items()
            if v["rmse"] is not None
        ]
        r2_candidates = [
            (model, v["r2"])
            for model, v in per_file_metrics.items()
            if v["r2"] is not None
        ]

        if mape_candidates:
            best_mape_value = min(v for _, v in mape_candidates)
            for model, value in mape_candidates:
                if value == best_mape_value:
                    best_counts[model]["mape"] += 1

        if rmse_candidates:
            best_rmse_value = min(v for _, v in rmse_candidates)
            for model, value in rmse_candidates:
                if value == best_rmse_value:
                    best_counts[model]["rmse"] += 1

        if r2_candidates:
            best_r2_value = max(v for _, v in r2_candidates)
            for model, value in r2_candidates:
                if value == best_r2_value:
                    best_counts[model]["r2"] += 1

summary_rows = []
for model in canonical_models:
    mape_values = model_metrics[model]["mape"]
    rmse_values = model_metrics[model]["rmse"]
    r2_values = model_metrics[model]["r2"]

    mape_mean = float(np.mean(mape_values)) if mape_values else np.nan
    mape_std = float(np.std(mape_values, ddof=0)) if mape_values else np.nan
    rmse_mean = float(np.mean(rmse_values)) if rmse_values else np.nan
    rmse_std = float(np.std(rmse_values, ddof=0)) if rmse_values else np.nan
    r2_mean = float(np.mean(r2_values)) if r2_values else np.nan

    summary_rows.append(
        {
            "Model": model,
            "MAPE (Mean ± Std)": (
                f"{mape_mean:.4f} ± {mape_std:.4f}" if mape_values else "N/A"
            ),
            "RMSE (Mean ± Std)": (
                f"{rmse_mean:.4f} ± {rmse_std:.4f}" if rmse_values else "N/A"
            ),
            "Avg $R^2$": f"{r2_mean:.4f}" if r2_values else "N/A",
            "Best MAPE": best_counts[model]["mape"],
            "best RMSE": best_counts[model]["rmse"],
            "best R^2": best_counts[model]["r2"],
            "failed(R^2<0)": failed_r2_counts[model],
        }
    )

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Model",
        "MAPE (Mean ± Std)",
        "RMSE (Mean ± Std)",
        "Avg $R^2$",
        "Best MAPE",
        "best RMSE",
        "best R^2",
        "failed(R^2<0)",
    ],
)

output_filename = "model_metrics_summary.csv"
summary_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Model metrics summary saved to {output_filename}")
