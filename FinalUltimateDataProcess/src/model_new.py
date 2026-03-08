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

    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type.lower()
        self.model = self._build_model(self.model_type)
        # 用于存储线性模型的统计信息
        self.xtx_inv_ = None
        self.mse_resid_ = None
        self.dof_ = None
        self.default_road_type_ = "其他"

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

        return X_df[["speed_mean", "power_mean", "road_type"]]

    def _build_preprocessor(self, model_type: str):
        """构建特征预处理管道"""
        numeric_features = ["speed_mean", "power_mean"]
        categorical_features = ["road_type"]

        # 定义已知的道路类型，确保编码一致性，并对应 '其他' 为最后一个(index 3)
        known_cats = [["高速", "国道", "省道", "其他"]]

        if model_type == "gam":
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

        if model_type == "gam":
            if PYGAM_AVAILABLE:
                # 这里的 model 只是一个占位符，GAM 不太兼容 sklearn Pipeline 的 fit 逻辑
                # 我们将在 fit() 方法中单独处理 GAM
                return None
            else:
                raise ValueError("pygam not installed.")

        raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X, y, sample_weight=None):
        # 确保输入是 DataFrame
        X = self._ensure_dataframe(X)

        y_arr = np.asarray(y)

        # 特殊处理 GAM
        if self.model_type == "gam":
            # 手动执行预处理
            self.preprocessor_ = self._build_preprocessor("gam")
            X_trans = self.preprocessor_.fit_transform(X)

            # 定义 GAM 结构：0,1列是数值(spline)，2列是类别(factor)
            # s(0) -> speed, s(1) -> power, f(2) -> road_type
            self.model = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10) + f(2))
            self.model.fit(X_trans, y_arr, weights=sample_weight)
            return self

        # 标准 Sklearn Pipeline 流程
        # 注意：Pipeline 传递 sample_weight 需要带上前缀 'regressor__'
        if sample_weight is not None:
            fit_params = {"regressor__sample_weight": sample_weight}
            self.model.fit(X, y_arr, **fit_params)
        else:
            self.model.fit(X, y_arr)

        # --- 计算预测区间统计量 (仅针对线性模型) ---
        if self.model_type in ["linear", "ridge", "lasso"]:
            # 获取经过转换后的特征矩阵 X_trans
            X_trans = self.model.named_steps["preprocessor"].transform(X)
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
            preds = self.model.predict(X)
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
        if self.model_type == "gam":
            X_trans = self.preprocessor_.transform(X)
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

        # 获取转换后的特征 (One-Hot 之后)
        X_trans = self.model.named_steps["preprocessor"].transform(X)
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
        denominator = (np.abs(y_arr) + np.abs(pred_arr)) / 2.0 + epsilon
        smape = np.mean(np.abs(y_arr - pred_arr) / denominator) * 100

        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "smape": smape}

    def get_coefficients(self, feature_names: list[str] = None):
        """提取系数，自动处理 One-Hot 后的特征名"""
        if self.model_type in ["linear", "ridge", "lasso"]:
            reg = self.model.named_steps["regressor"]
            pre = self.model.named_steps["preprocessor"]

            # 获取 One-Hot 后的特征名
            try:
                feature_names = pre.get_feature_names_out()
            except:
                feature_names = [f"feat_{i}" for i in range(len(reg.coef_))]

            coefs = reg.coef_
            return dict(zip(feature_names, coefs)), reg.intercept_

        return {}, None
