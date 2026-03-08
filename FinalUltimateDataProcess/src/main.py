import argparse
import pandas as pd
import numpy as np
import sys
import os
import json

# Add project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import build_dataset_from_source, features_labels_from_sampled
from src.model_new import Predictor
from src.utils import get_logger, SKLEARN_AVAILABLE, save_result, save_predict_result

# 注意：visualizer 内部可能需要修改以适应 DataFrame 输入，否则这里调用可能会报错
from src.visualizer import visualize_3d_model, save_2d_plot

if SKLEARN_AVAILABLE:
    from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate consumption predictor"
    )
    parser.add_argument(
        "--source", required=True, help="CSV file or directory with CSV files"
    )

    parser.add_argument(
        "--model",
        default="linear",
        choices=(
            (
                "linear",
                "ridge",
                "lasso",
                "rf",
                "xgboost",
                "lightgbm",
                "polynomial",
                "gam",
            )
        ),
        help="model type",
    )

    parser.add_argument(
        "--min_rows", type=int, default=10, help="Sampler: minimum rows in a window"
    )
    parser.add_argument(
        "--max_interval",
        type=int,
        default=15,
        help="Sampler: max interval in minutes between rows",
    )
    parser.add_argument(
        "--min_coverage", type=float, default=0.5, help="Sampler: minimum coverage rate"
    )

    # --- No need to modify. ---
    parser.add_argument(
        "--time_col", default="Time", help="time column name in CSV if different"
    )
    parser.add_argument("--speed_col", default="speed", help="speed column name")
    parser.add_argument("--power_col", default="power", help="power column name")
    parser.add_argument(
        "--acc_col", default="accumulated_usage", help="accumulated usage column name"
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--show_plot", action="store_true", help="whether to show 3D plot"
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        default=False,
        help="whether to save 3D plot image",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save evaluation results",
    )
    parser.add_argument(
        "--save_predict_result",
        action="store_true",
        help="whether to save evaluation results",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="custom base name for output files (optional)",
    )
    args = parser.parse_args()

    logger.info(f"Loading and sampling source: {args.source}")
    sampled, metadata = build_dataset_from_source(
        args.source,
        time_col=args.time_col,
        speed_col=args.speed_col,
        power_col=args.power_col,
        acc_col=args.acc_col,
        min_rows_in_window=args.min_rows,
        max_interval_minutes=args.max_interval,
        min_coverage_rate=args.min_coverage,
    )
    if sampled.empty:
        logger.error("No valid samples generated. Check column names and data.")
        return

    # X is now a DataFrame (speed, power, road_type), y is array, weights is array
    X, y, weights = features_labels_from_sampled(sampled)

    # Split data including weights
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=args.test_size, random_state=args.random_seed
    )

    # --- Analysis: Correlation between Speed and Power ---
    # FIX: Use .iloc or column names because X is a DataFrame now
    speed_vals = X["speed_mean"]
    power_vals = X["power_mean"]
    corr = np.corrcoef(speed_vals, power_vals)[0, 1]

    logger.info(f"Correlation between Speed and Power: {corr:.4f}")
    if abs(corr) > 0.8:  # 会导致参数估计不稳定、模型解释性变差
        logger.warning("High multicollinearity detected between Speed and Power!")

    # --- Experiment 1: Full Model (Speed + Power) ---
    logger.info("--- Training Full Model (Speed + Power) ---")
    logger.info(f"--- Training Full Model ({args.model}) ---")

    predictor = Predictor(model_type=args.model)
    predictor.fit(X_train, y_train, sample_weight=w_train)
    metrics_full = predictor.evaluate(X_test, y_test)

    feature_names = ["speed_mean", "power_mean", "road_type"]
    coefs_dict, intercept = predictor.get_coefficients(feature_names=feature_names)

    result_data = {
        "source": args.source,
        "model": args.model,
        "number_of_groups": metadata.get("number_of_groups"),
        "group_sizes": metadata.get("group_sizes"),
        "correlation_speed_power": float(corr),
        "full_model_metrics": metrics_full,
        "full_model_coefficients": None,
        "relative_importance": None,
        "hyperparameters": (
            predictor.model.get_params()
            if hasattr(predictor.model, "get_params")
            else {}
        ),
    }

    # Clean hyperparameters for JSON serialization
    # Some sklearn objects (like steps in Pipeline) are not serializable
    if "steps" in result_data["hyperparameters"]:
        # For Pipeline, simplify the steps representation
        steps = result_data["hyperparameters"]["steps"]
        result_data["hyperparameters"]["steps"] = str(steps)

    # Convert any other non-serializable objects to string
    for k, v in result_data["hyperparameters"].items():
        try:
            json.dumps(v)
        except (TypeError, OverflowError):
            result_data["hyperparameters"][k] = str(v)

    if coefs_dict:
        logger.info(f"Model Coefficients: {coefs_dict}")
        if intercept is not None:
            logger.info(f"Intercept: {intercept:.4f}")

        # Convert numpy types to float for JSON serialization
        serializable_coefs = {k: float(v) for k, v in coefs_dict.items()}
        if intercept is not None:
            serializable_coefs["intercept"] = float(intercept)

        result_data["full_model_coefficients"] = serializable_coefs

        # Relative Importance Calculation (New Version)
        # 这一步需要处理 One-Hot 后的多列问题

        # 1. 获取预处理后的特征矩阵 (Transformed X) 以计算标准差
        try:
            # 检查是否为 Pipeline 模型 (GAM 不包含 named_steps)
            if hasattr(predictor.model, "named_steps"):
                # 从 Pipeline 中提取预处理步骤
                preprocessor = predictor.model.named_steps["preprocessor"]
                # 将 X_train 转换为模型实际看到的数值矩阵
                X_trans = preprocessor.transform(X_train)

                # 获取特征名称 (处理 One-Hot 后的名字)
                try:
                    # Sklearn >= 1.0
                    feature_names_out = preprocessor.get_feature_names_out()
                except AttributeError:
                    # Fallback for older sklearn
                    feature_names_out = [f"feat_{i}" for i in range(X_trans.shape[1])]

                # 2. 计算每个细分特征的 "Raw Importance" (|Coef| * Std)
                # 注意：X_trans 可能是稀疏矩阵，转为 dense array 计算 std
                if hasattr(X_trans, "toarray"):
                    X_trans = X_trans.toarray()

                # 计算每列的标准差
                stds = np.std(X_trans, axis=0)

                # 匹配系数
                # 线性模型有 coef_, 树模型有 feature_importances_
                raw_importances = {}
                regressor = predictor.model.named_steps["regressor"]

                if hasattr(regressor, "coef_"):
                    coefs = regressor.coef_
                    if coefs.ndim > 1:
                        coefs = coefs.ravel()
                    for name, coef, std in zip(feature_names_out, coefs, stds):
                        raw_importances[name] = abs(coef * std)
                elif hasattr(regressor, "feature_importances_"):
                    # 树模型不需要乘 std，feature_importances_ 本身就是相对贡献
                    importances = regressor.feature_importances_
                    for name, imp in zip(feature_names_out, importances):
                        raw_importances[name] = imp

                # 3. 聚合回原始特征 (Group by Prefix)
                # feature_names_out 例如: ['num__speed_mean', 'num__power_mean', 'cat__road_type_高速', 'cat__road_type_国道'...]
                aggregated_importance = {}

                for name, imp in raw_importances.items():
                    # 解析原始特征名
                    if "road_type" in name:
                        key = "road_type"
                    elif "speed" in name:
                        key = "speed"
                    elif "power" in name:
                        key = "power"
                    else:
                        key = "other"

                    aggregated_importance[key] = (
                        aggregated_importance.get(key, 0.0) + imp
                    )

                # 4. 计算百分比
                total_imp = sum(aggregated_importance.values())
                if total_imp > 0:
                    relative_importance = {
                        k: v / total_imp for k, v in aggregated_importance.items()
                    }

                    logger.info(
                        f"Relative Importance (Aggregated): {relative_importance}"
                    )
                    result_data["relative_importance"] = relative_importance

        except Exception as e:
            logger.warning(f"Could not calculate relative importance: {e}")

    logger.info(f"Full Model Metrics: {metrics_full}")

    # --- NOTE: Ablation Study Disabled ---
    # Since Predictor now uses a Pipeline that strictly expects 'road_type',
    # passing single-column DataFrames for Ablation will cause a crash.
    # To re-enable, one would need to bypass the Predictor class and use raw sklearn models here.
    """
    if args.model == "linear":
    logger.info("--- Ablation Study skipped due to Pipeline constraints ---")

        # --- Experiment 2: Ablation Study (Remove Power) ---
        logger.info("--- Training Reduced Model (Speed Only) ---")
        # X[:, 0] is Speed (assuming order is [speed, power])
        X_train_speed = X_train[:, 0].reshape(-1, 1)
        X_test_speed = X_test[:, 0].reshape(-1, 1)

        predictor_speed = Predictor(model_type=args.model)
        predictor_speed.fit(X_train_speed, y_train, sample_weight=w_train)
        metrics_speed = predictor_speed.evaluate(X_test_speed, y_test)

        logger.info(f"Reduced Model Metrics: {metrics_speed}")

        # Compare
        r2_drop = metrics_full["r2"] - metrics_speed["r2"]
        logger.info(f"Impact of removing Power: R2 dropped by {r2_drop:.4f}")

        result_data["reduced_model_metrics_speed_only"] = metrics_speed
        result_data["impact_removing_power_r2_drop"] = float(r2_drop)

        # --- Experiment 3: Ablation Study (Remove Speed) ---
        logger.info("--- Training Reduced Model (Power Only) ---")
        # X[:, 1] is Power (assuming order is [speed, power])
        X_train_power = X_train[:, 1].reshape(-1, 1)
        X_test_power = X_test[:, 1].reshape(-1, 1)

        predictor_power = Predictor(model_type=args.model)
        predictor_power.fit(X_train_power, y_train, sample_weight=w_train)
        metrics_power = predictor_power.evaluate(X_test_power, y_test)

        logger.info(f"Reduced Model Metrics: {metrics_power}")

        # Compare
        r2_drop_power = metrics_full["r2"] - metrics_power["r2"]
        logger.info(f"Impact of removing Speed: R2 dropped by {r2_drop_power:.4f}")

        result_data["reduced_model_metrics_power_only"] = metrics_power
        result_data["impact_removing_speed_r2_drop"] = float(r2_drop_power)
"""
    if args.save_predict_result:
        target_predict_dir = "outputs/predictions/"
        if not os.path.exists(target_predict_dir):
            os.makedirs(target_predict_dir)

        if args.output_name:
            base_name = args.output_name
        else:
            base_name = os.path.basename(args.source)
            base_name = os.path.splitext(base_name)[0]

        target_predict_path = os.path.join(
            target_predict_dir, f"{base_name}_predictions.csv"
        )
        # Ensure save_predict_result handles DataFrame X correctly
        save_predict_result(predictor, X, target_predict_path)

    # --- NOTE: Visualization Disabled or Needs Update ---
    # The visualizer likely expects X to be a 2D numpy array [speed, power] to generate a meshgrid.
    # It does not know how to handle the categorical 'road_type' column for 3D plotting.
    # Suggestion: Update visualizer to fix road_type='国道' when predicting surface.

    if args.show_plot:
        # visualize_3d_model(X, y, predictor)
        logger.warning(
            "3D Plot disabled: Visualizer needs update for categorical features."
        )
        pass

    if args.save_plot:
        target_plot_dir = "outputs/plots/"
        if not os.path.exists(target_plot_dir):
            os.makedirs(target_plot_dir)

        if args.output_name:
            base_name = args.output_name
        else:
            base_name = os.path.basename(args.source)
            base_name = os.path.splitext(base_name)[0]

        target_plot_path = os.path.join(target_plot_dir, f"{base_name}.png")
        if not os.path.exists(target_plot_path):
            # save_2d_plot(X, y, target_plot_path)
            pass

    if args.save_result:
        results_dir = "outputs/results/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if args.output_name:
            base_name = args.output_name
        else:
            base_name = os.path.basename(args.source)
            base_name = os.path.splitext(base_name)[0]

        target_result_path = os.path.join(results_dir, f"{base_name}.json")
        save_result(result_data, target_result_path)


if __name__ == "__main__":
    main()
