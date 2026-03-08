import logging
import os
import json
import numpy as np
import pandas as pd
import itertools

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


def get_logger(name: str):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    return logging.getLogger(name)


def save_result(result_data: dict, file_path: str):
    """Save experiment results to a JSON file with a specific structure."""

    # Keys that belong to the top level
    top_level_keys = [
        "source",
        "number_of_groups",
        "group_sizes",
        "correlation_speed_power",
    ]

    # Construct the model result object (everything that is not top level)
    model_result = {k: v for k, v in result_data.items() if k not in top_level_keys}

    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    if not isinstance(data, dict):
        # If file content is not a dict (e.g. list from previous version),
        # we initialize a new structure using current result_data's top level info.
        data = {k: result_data.get(k) for k in top_level_keys}
        data["results"] = []

    # Ensure top level keys exist
    for k in top_level_keys:
        if k not in data:
            data[k] = result_data.get(k)

    # Ensure results list exists
    if "results" not in data or not isinstance(data["results"], list):
        data["results"] = []

    # Append current model result
    data["results"].append(model_result)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_predict_result(predictor, X, target_predict_path, split=30):
    """
    Generate a grid of points based on X's range, predict with intervals, and save to CSV.
    """
    # Determine columns/values
    if isinstance(X, pd.DataFrame):
        # Try to find speed and power columns by name
        # Assuming standard names or just taking first two
        # The prompt says "speed_mean" and "power_mean"
        speed_col = next((c for c in X.columns if "speed" in c.lower()), None)
        power_col = next((c for c in X.columns if "power" in c.lower()), None)

        if speed_col and power_col:
            s_vals = X[speed_col].values
            p_vals = X[power_col].values
        else:
            # Fallback to iloc 0 and 1
            s_vals = X.iloc[:, 0].values
            p_vals = X.iloc[:, 1].values
        road_type_default = None
        if "road_type" in X.columns:
            road_type_default = X["road_type"].mode().iloc[0]
    else:
        X_arr = np.array(X)
        s_vals = X_arr[:, 0]
        p_vals = X_arr[:, 1]
        road_type_default = None

    def get_extended_grid(vals, n_points):
        v_min, v_max = np.min(vals), np.max(vals)
        v_range = v_max - v_min
        if v_range == 0:
            v_range = 1.0

        new_min = v_min - 0.1 * v_range
        new_max = v_max + 0.1 * v_range

        # "如果遇到0则最小取0" -> If original min >= 0, clamp extended min to 0
        if v_min >= 0 and new_min < 0:
            new_min = 0

        return np.linspace(new_min, new_max, n_points)

    s_grid = get_extended_grid(s_vals, split)
    p_grid = get_extended_grid(p_vals, split)

    # Create grid (Cartesian product)
    # itertools.product returns (s, p) tuples
    grid_points = np.array(list(itertools.product(s_grid, p_grid)))

    # Build DataFrame with default road_type for prediction
    if road_type_default is None:
        road_type_default = "其他"
    grid_df = pd.DataFrame(
        {
            "speed_mean": grid_points[:, 0],
            "power_mean": grid_points[:, 1],
            "road_type": road_type_default,
        }
    )

    # Predict
    # predictor.predict_with_interval returns (preds, lower, upper)
    preds, lower, upper = predictor.predict_with_interval(grid_df, confidence=0.95)

    # Save
    df_res = pd.DataFrame(
        {
            "speed": grid_points[:, 0],
            "power": grid_points[:, 1],
            "usage_rate": preds,
            "usage_rate_lower": lower,
            "usage_rate_upper": upper,
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(target_predict_path), exist_ok=True)
    df_res.to_csv(target_predict_path, index=False)
