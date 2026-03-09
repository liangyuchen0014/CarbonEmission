from __future__ import annotations
import numpy as np
import pandas as pd
from src.utils import SKLEARN_AVAILABLE, JOBLIB_AVAILABLE

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from scipy import stats
from typing import Any, Optional, cast

if SKLEARN_AVAILABLE:
    from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from pygam import LinearGAM, s, f

    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

if JOBLIB_AVAILABLE:
    import joblib


class Predictor:
    """
    Wrapper for various regression models with automatic preprocessing for mixed data types.
    """

    def __init__(
        self,
        model_type: str = "linear",
        random_seed: int = 42,
        woa_population: int = 12,
        woa_iterations: int = 20,
        woa_val_size: float = 0.2,
        woa_lam_min: float = 1e-3,
        woa_lam_max: float = 1e3,
        woa_n_splines_min: int = 5,
        woa_n_splines_max: int = 50,
    ):
        self.model_type = model_type.lower()
        self.random_seed = random_seed
        self.woa_population = woa_population
        self.woa_iterations = woa_iterations
        self.woa_val_size = woa_val_size
        self.woa_lam_bounds = (woa_lam_min, woa_lam_max)
        self.woa_n_splines_bounds = (woa_n_splines_min, woa_n_splines_max)
        self.model: Any = self._build_model(self.model_type)
        # 用于存储线性模型的统计信息
        self.xtx_inv_ = None
        self.mse_resid_ = None
        self.dof_ = None
        self.default_road_type_ = "其他"
        self.preprocessor_: Any = None
        self.gam_params_: dict[str, Any] = {}
        self.woa_info_: dict[str, Any] = {}

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """确保输入为包含 speed_mean/power_mean/road_type 的 DataFrame"""
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X)
            X_df = pd.DataFrame(X_arr, columns=["speed_mean", "power_mean"])

        if "road_type" not in X_df.columns:
            print(
                "Warning: 'road_type' column not found in input. Using default value."
            )
            X_df["road_type"] = self.default_road_type_

        # 允许用户传入 speed/power 的别名列
        if "speed_mean" not in X_df.columns:
            speed_col = next((c for c in X_df.columns if "speed" in c.lower()), None)
            if speed_col is not None:
                X_df = X_df.rename(columns={speed_col: "speed_mean"})
        if "power_mean" not in X_df.columns:
            power_col = next((c for c in X_df.columns if "power" in c.lower()), None)
            if power_col is not None:
                X_df = X_df.rename(columns={power_col: "power_mean"})

        return cast(pd.DataFrame, X_df[["speed_mean", "power_mean", "road_type"]])

    def _build_preprocessor(self, model_type: str):
        """构建特征预处理管道"""
        numeric_features = ["speed_mean", "power_mean"]
        categorical_features = ["road_type"]

        # 定义已知的道路类型，确保编码一致性，并对应 '其他' 为最后一个(index 3)
        known_cats = [["高速", "国道", "省道", "其他"]]

        if model_type in ["gam", "woa_gam"]:
            # GAM 需要数值输入，对于类别特征通常使用 Ordinal 编码 + Factor term f()
            # 设置 handle_unknown='use_encoded_value' 和 unknown_value=-1
            # 在 python 数组索引中，-1 代表最后一个元素。
            # 我们的 known_cats[0] 最后一个是 '其他'。
            # 因此，遇到未知类别（编码为-1）及其它转为整数后的 -1，
            # 传入 pygam 的 f() 项时，若是用来索引系数，可能会对应到 '其他'。
            # (注：pygam 具体如何处理 input array index -1 需视其实现而定，
            #  但在 array lookup 语义下 -1 是有效的 safe fallback)
            return ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_features),
                    (
                        "cat",
                        OrdinalEncoder(
                            categories=known_cats,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                        categorical_features,
                    ),
                ]
            )
        else:
            # 线性模型、树模型通用：One-Hot 编码
            # handle_unknown='ignore' 防止预测时遇到训练集中没见过的类别报错
            return ColumnTransformer(
                transformers=[
                    (
                        "num",
                        StandardScaler(),
                        numeric_features,
                    ),  # 线性模型建议归一化，树模型无所谓
                    (
                        "cat",
                        OneHotEncoder(
                            categories=known_cats,
                            sparse_output=False,
                            handle_unknown="ignore",
                        ),
                        categorical_features,
                    ),
                ]
            )

    def _build_model(self, model_type: str):
        preprocessor = self._build_preprocessor(model_type)

        if model_type == "linear":
            return Pipeline(
                [("preprocessor", preprocessor), ("regressor", LinearRegression())]
            )

        if model_type == "ridge" and SKLEARN_AVAILABLE:
            return Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])

        if model_type == "lasso" and SKLEARN_AVAILABLE:
            return Pipeline([("preprocessor", preprocessor), ("regressor", Lasso())])

        if model_type == "rf" and SKLEARN_AVAILABLE:
            return Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        RandomForestRegressor(n_estimators=100, random_state=42),
                    ),
                ]
            )

        if model_type == "polynomial" and SKLEARN_AVAILABLE:
            return Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("regressor", LinearRegression()),
                ]
            )

        if model_type == "xgboost":
            if XGBOOST_AVAILABLE:
                return Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        (
                            "regressor",
                            xgb.XGBRegressor(n_estimators=100, random_state=42),
                        ),
                    ]
                )
            else:
                raise ValueError("XGBoost not installed.")

        if model_type == "lightgbm":
            if LIGHTGBM_AVAILABLE:
                return Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        (
                            "regressor",
                            lgb.LGBMRegressor(n_estimators=100, random_state=42),
                        ),
                    ]
                )
            else:
                raise ValueError("LightGBM not installed.")

        if model_type in ["gam", "woa_gam"]:
            if PYGAM_AVAILABLE:
                # 这里的 model 只是一个占位符，GAM 不太兼容 sklearn Pipeline 的 fit 逻辑
                # 我们将在 fit() 方法中单独处理 GAM
                return None
            else:
                raise ValueError("pygam not installed.")

        raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def _smape(y_true, y_pred) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0 + 1e-8
        return float(np.mean(np.abs(y_true_arr - y_pred_arr) / denominator) * 100)

    def _create_gam_model(
        self,
        lam_speed: float = 0.6,
        lam_power: float = 0.6,
        n_splines_speed: int = 10,
        n_splines_power: int = 10,
    ):
        self.gam_params_ = {
            "lam_speed": float(lam_speed),
            "lam_power": float(lam_power),
            "n_splines_speed": int(n_splines_speed),
            "n_splines_power": int(n_splines_power),
        }
        terms = cast(
            Any,
            s(
                0,
                lam=self.gam_params_["lam_speed"],
                n_splines=self.gam_params_["n_splines_speed"],
            )
            + s(
                1,
                lam=self.gam_params_["lam_power"],
                n_splines=self.gam_params_["n_splines_power"],
            )
            + f(2),
        )
        return LinearGAM(terms)

    def _fit_gam_model(self, X_trans, y_arr, sample_weight=None, params=None):
        params = params or {}
        model = self._create_gam_model(**params)
        model.fit(X_trans, y_arr, weights=sample_weight)
        return model

    def _decode_woa_position(self, position, max_n_splines: int) -> dict[str, Any]:
        log_lam_min = np.log10(self.woa_lam_bounds[0])
        log_lam_max = np.log10(self.woa_lam_bounds[1])
        lam_speed = 10 ** float(np.clip(position[0], log_lam_min, log_lam_max))
        lam_power = 10 ** float(np.clip(position[1], log_lam_min, log_lam_max))

        n_splines_low = max(4, int(self.woa_n_splines_bounds[0]))
        n_splines_high = max(n_splines_low, int(max_n_splines))

        n_splines_speed = int(
            np.clip(np.rint(position[2]), n_splines_low, n_splines_high)
        )
        n_splines_power = int(
            np.clip(np.rint(position[3]), n_splines_low, n_splines_high)
        )

        return {
            "lam_speed": lam_speed,
            "lam_power": lam_power,
            "n_splines_speed": n_splines_speed,
            "n_splines_power": n_splines_power,
        }

    def _run_woa_search(self, X_trans, y_arr, sample_weight=None):
        if not SKLEARN_AVAILABLE:
            raise ValueError("scikit-learn is required for WOA-GAM validation split.")

        n_samples = len(y_arr)
        if n_samples < 5:
            default_params = self._decode_woa_position(
                np.array([0.0, 0.0, 10.0, 10.0], dtype=float),
                max_n_splines=max(5, n_samples - 1),
            )
            return {
                "best_params": default_params,
                "best_val_smape": np.nan,
                "population": 1,
                "iterations": 0,
                "validation_size": 0.0,
                "search_space": {
                    "lam": [
                        float(self.woa_lam_bounds[0]),
                        float(self.woa_lam_bounds[1]),
                    ],
                    "n_splines": [
                        int(self.woa_n_splines_bounds[0]),
                        int(self.woa_n_splines_bounds[1]),
                    ],
                },
                "evaluations": 0,
            }

        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
            X_trans,
            y_arr,
            test_size=self.woa_val_size,
            random_state=self.random_seed,
        )

        if sample_weight is not None:
            w_train_inner, w_val_inner = train_test_split(
                np.asarray(sample_weight),
                test_size=self.woa_val_size,
                random_state=self.random_seed,
            )
        else:
            w_train_inner = None
            w_val_inner = None

        max_n_splines = min(
            int(self.woa_n_splines_bounds[1]),
            max(int(self.woa_n_splines_bounds[0]), X_train_inner.shape[0] - 1),
        )

        rng = np.random.default_rng(self.random_seed)
        dim = 4
        log_lam_min = np.log10(self.woa_lam_bounds[0])
        log_lam_max = np.log10(self.woa_lam_bounds[1])
        lower_bounds = np.array(
            [
                log_lam_min,
                log_lam_min,
                float(self.woa_n_splines_bounds[0]),
                float(self.woa_n_splines_bounds[0]),
            ],
            dtype=float,
        )
        upper_bounds = np.array(
            [
                log_lam_max,
                log_lam_max,
                float(max_n_splines),
                float(max_n_splines),
            ],
            dtype=float,
        )

        population = max(3, int(self.woa_population))
        iterations = max(1, int(self.woa_iterations))
        positions = rng.uniform(lower_bounds, upper_bounds, size=(population, dim))
        fitness = np.full(population, np.inf, dtype=float)
        best_position = positions[0].copy()
        best_fitness = np.inf
        evaluations = 0

        def objective(position):
            nonlocal evaluations
            params = self._decode_woa_position(position, max_n_splines=max_n_splines)
            try:
                candidate_model = self._fit_gam_model(
                    X_train_inner,
                    y_train_inner,
                    sample_weight=w_train_inner,
                    params=params,
                )
                pred_val = candidate_model.predict(X_val_inner)
                score = self._smape(y_val_inner, pred_val)
            except Exception:
                score = np.inf
            evaluations += 1
            return score

        for idx in range(population):
            fitness[idx] = objective(positions[idx])

        best_idx = int(np.argmin(fitness))
        best_fitness = float(fitness[best_idx])
        best_position = positions[best_idx].copy()

        for iteration in range(iterations):
            a = 2 - 2 * (iteration / max(iterations, 1))
            for idx in range(population):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = rng.random()
                l = rng.uniform(-1.0, 1.0)

                if p < 0.5:
                    if np.linalg.norm(A, ord=2) < 1:
                        distance = np.abs(C * best_position - positions[idx])
                        new_position = best_position - A * distance
                    else:
                        rand_idx = int(rng.integers(0, population))
                        random_position = positions[rand_idx]
                        distance = np.abs(C * random_position - positions[idx])
                        new_position = random_position - A * distance
                else:
                    distance = np.abs(best_position - positions[idx])
                    new_position = (
                        distance * np.exp(1.0 * l) * np.cos(2 * np.pi * l)
                        + best_position
                    )

                positions[idx] = np.clip(new_position, lower_bounds, upper_bounds)
                fitness[idx] = objective(positions[idx])

                if fitness[idx] < best_fitness:
                    best_fitness = float(fitness[idx])
                    best_position = positions[idx].copy()

        best_params = self._decode_woa_position(
            best_position, max_n_splines=max_n_splines
        )
        return {
            "best_params": best_params,
            "best_val_smape": best_fitness,
            "population": population,
            "iterations": iterations,
            "validation_size": float(self.woa_val_size),
            "search_space": {
                "lam": [float(self.woa_lam_bounds[0]), float(self.woa_lam_bounds[1])],
                "n_splines": [int(self.woa_n_splines_bounds[0]), int(max_n_splines)],
            },
            "evaluations": evaluations,
        }

    def fit(self, X, y, sample_weight=None):
        # 确保输入是 DataFrame
        X = self._ensure_dataframe(X)

        y_arr = np.asarray(y)

        # 特殊处理 GAM
        if self.model_type in ["gam", "woa_gam"]:
            # 手动执行预处理
            self.preprocessor_ = self._build_preprocessor("gam")
            X_trans = np.asarray(self.preprocessor_.fit_transform(X))

            # 定义 GAM 结构：0,1列是数值(spline)，2列是类别(factor)
            # s(0) -> speed, s(1) -> power, f(2) -> road_type
            if self.model_type == "woa_gam":
                self.woa_info_ = self._run_woa_search(
                    X_trans, y_arr, sample_weight=sample_weight
                )
                best_params = self.woa_info_.get("best_params", {})
            else:
                self.woa_info_ = {}
                best_params = {
                    "lam_speed": 0.6,
                    "lam_power": 0.6,
                    "n_splines_speed": 10,
                    "n_splines_power": 10,
                }

            self.model = self._fit_gam_model(
                X_trans, y_arr, sample_weight=sample_weight, params=best_params
            )
            return self

        # 标准 Sklearn Pipeline 流程
        # 注意：Pipeline 传递 sample_weight 需要带上前缀 'regressor__'
        model = self.model
        if sample_weight is not None:
            fit_params = {"regressor__sample_weight": sample_weight}
            model.fit(X, y_arr, **fit_params)
        else:
            model.fit(X, y_arr)

        # --- 计算预测区间统计量 (仅针对线性模型) ---
        if self.model_type in ["linear", "ridge", "lasso"]:
            pipeline_model = cast(Any, self.model)
            # 获取经过转换后的特征矩阵 X_trans
            X_trans = pipeline_model.named_steps["preprocessor"].transform(X)
            n_samples, n_features = X_trans.shape

            # 添加截距项列
            X_design = np.hstack([np.ones((n_samples, 1)), X_trans])

            # (X^T X)^-1
            xtx = X_design.T @ X_design
            try:
                self.xtx_inv_ = np.linalg.inv(xtx)
            except np.linalg.LinAlgError:
                self.xtx_inv_ = None

            # 计算残差 MSE
            preds = pipeline_model.predict(X)
            residuals = y_arr - preds
            dof = n_samples - n_features - 1
            if dof > 0:
                self.mse_resid_ = np.sum(residuals**2) / dof
            else:
                self.mse_resid_ = None

            self.dof_ = dof

        return self

    def predict(self, X):
        X = self._ensure_dataframe(X)
        if self.model_type in ["gam", "woa_gam"]:
            X_trans = np.asarray(self.preprocessor_.transform(X))
            return self.model.predict(X_trans)

        # Pipeline 会自动处理 transform
        return self.model.predict(X)

    def predict_with_interval(self, X, confidence=0.95):
        """
        计算置信区间 (仅支持 Linear/Ridge/Lasso)
        """
        X = self._ensure_dataframe(X)
        preds = self.predict(X)

        if (
            self.model_type not in ["linear", "ridge", "lasso"]
            or self.xtx_inv_ is None
            or self.mse_resid_ is None
        ):
            return preds, preds, preds

        pipeline_model = cast(Any, self.model)
        # 获取转换后的特征 (One-Hot 之后)
        X_trans = pipeline_model.named_steps["preprocessor"].transform(X)
        n_samples = X_trans.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X_trans])

        # 计算 Leverage
        leverage = np.sum(X_design.dot(self.xtx_inv_) * X_design, axis=1)
        leverage = np.maximum(leverage, 0)

        pred_var = self.mse_resid_ * (1 + leverage)
        pred_std = np.sqrt(pred_var)

        alpha = 1 - confidence
        # t 分布临界值
        t_score = stats.t.ppf(1 - alpha / 2, self.dof_)

        margin = t_score * pred_std
        lower = preds - margin
        upper = preds + margin

        return preds, lower, upper

    def evaluate(self, X, y) -> dict:
        pred = self.predict(X)
        y_arr = np.asarray(y)
        pred_arr = np.asarray(pred)

        mae = mean_absolute_error(y_arr, pred_arr)
        rmse = np.sqrt(mean_squared_error(y_arr, pred_arr))
        r2 = r2_score(y_arr, pred_arr)

        # return {"mae": mae, "rmse": rmse, "r2": r2}
        epsilon = 1e-8
        mask = np.abs(y_arr) > epsilon
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_arr[mask] - pred_arr[mask]) / y_arr[mask])) * 100
        else:
            mape = np.nan

        # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
        # Formula: 100/n * sum( |y - y_pred| / ( (|y| + |y_pred|)/2 + epsilon ) )
        # This handles y=0 cases better, but still needs epsilon for y=y_pred=0
        smape = self._smape(y_arr, pred_arr)

        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "smape": smape}

    def get_hyperparameters(self) -> dict[str, Any]:
        if self.model is not None and hasattr(self.model, "get_params"):
            params = dict(self.model.get_params())
        else:
            params = {}

        if self.model_type in ["gam", "woa_gam"]:
            params.update(self.gam_params_)

        if self.model_type == "woa_gam" and self.woa_info_:
            params["woa"] = self.woa_info_.copy()

        return params

    def get_woa_details(self) -> dict[str, Any]:
        return self.woa_info_.copy()

    def get_coefficients(self, feature_names: Optional[list[str]] = None):
        """提取系数，自动处理 One-Hot 后的特征名"""
        if self.model_type in ["linear", "ridge", "lasso"]:
            pipeline_model = cast(Any, self.model)
            reg = pipeline_model.named_steps["regressor"]
            pre = pipeline_model.named_steps["preprocessor"]

            # 获取 One-Hot 后的特征名
            try:
                names = list(pre.get_feature_names_out())
            except:
                names = [f"feat_{i}" for i in range(len(reg.coef_))]

            coefs = reg.coef_
            return dict(zip(names, coefs)), reg.intercept_

        return {}, None
