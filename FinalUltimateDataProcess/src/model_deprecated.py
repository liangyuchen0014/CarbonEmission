from __future__ import annotations
import numpy as np
from src.utils import SKLEARN_AVAILABLE, JOBLIB_AVAILABLE

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
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
    from pygam import LinearGAM, s

    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

if JOBLIB_AVAILABLE:
    import joblib


class Predictor:
    """A simple predictor wrapper. Supports 'linear' by default and other sklearn models if available.

    Contract (inputs/outputs):
    - fit(X: np.ndarray or DataFrame, y: np.ndarray)
    - predict(X) -> np.ndarray
    - evaluate(X, y) -> dict of metrics
    """

    def __init__(self, model_type: str = "linear"):
        self.model_type = model_type.lower()
        self.model = self._build_model(self.model_type)

    def _build_model(self, model_type: str):
        if model_type == "linear":
            return LinearRegression()
        if model_type == "ridge" and SKLEARN_AVAILABLE:
            return Ridge()
        if model_type == "lasso" and SKLEARN_AVAILABLE:
            return Lasso()
        if model_type == "rf" and SKLEARN_AVAILABLE:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        if model_type == "xgboost":
            if XGBOOST_AVAILABLE:
                return xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError("XGBoost not installed.")
        if model_type == "lightgbm":
            if LIGHTGBM_AVAILABLE:
                return lgb.LGBMRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError("LightGBM not installed.")
        if model_type == "polynomial":
            return make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
            )
        if model_type == "gam":
            if PYGAM_AVAILABLE:
                # s(0) is spline for feature 0 (Speed), s(1) is spline for feature 1 (Power)
                # n_splines controls complexity (default 25)
                return LinearGAM(s(0, n_splines=10) + s(1, n_splines=10))
            else:
                raise ValueError("pygam not installed.")
        raise ValueError(
            f"Unknown model_type or required package not available: {model_type}"
        )

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if self.model is None:
            # implement closed-form solution for linear regression (with intercept)
            # add bias column
            Xb = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
            # solve least squares
            # Note: sample_weight not supported in this simple closed form implementation
            coef, *_ = np.linalg.lstsq(Xb, y_arr, rcond=None)
            self.coef_ = coef
            self.model = "closed_form_linear"
            return self
        else:
            if sample_weight is not None and self.model_type in [
                "linear",
                "ridge",
                "lasso",
                "rf",
                "xgboost",
                "lightgbm",
            ]:
                self.model.fit(X_arr, y_arr, sample_weight=sample_weight)
            elif self.model_type == "gam" and PYGAM_AVAILABLE:
                self.model.fit(X_arr, y_arr, weights=sample_weight)
            else:
                self.model.fit(
                    X_arr, y_arr, linearregression__sample_weight=sample_weight
                )
            # --- Calculate statistics for prediction intervals (Linear Models only) ---
            if self.model_type in ["linear", "ridge", "lasso"]:
                n_samples, n_features = X_arr.shape
                # Add intercept column for calculation
                X_design = np.hstack([np.ones((n_samples, 1)), X_arr])

                # (X^T X)^-1
                xtx = X_design.T @ X_design
                try:
                    self.xtx_inv_ = np.linalg.inv(xtx)
                except np.linalg.LinAlgError:
                    self.xtx_inv_ = None  # Singular matrix

                # MSE of residuals
                preds = self.model.predict(X_arr)
                residuals = y_arr - preds
                # degrees of freedom = n - p - 1 (p features + 1 intercept)
                dof = n_samples - n_features - 1
                if dof > 0:
                    self.mse_resid_ = np.sum(residuals**2) / dof
                else:
                    self.mse_resid_ = None

                self.n_samples_ = n_samples
                self.dof_ = dof

            return self

    def predict_with_interval(self, X, confidence=0.95):
        """
        Predicts with confidence intervals.
        Only supported for linear models (linear, ridge, lasso) where fit() calculated stats.
        Returns:
            predictions (np.ndarray)
            lower_bound (np.ndarray)
            upper_bound (np.ndarray)
        """
        preds = self.predict(X)

        if (
            self.model_type not in ["linear", "ridge", "lasso"]
            or not hasattr(self, "xtx_inv_")
            or self.xtx_inv_ is None
            or not hasattr(self, "mse_resid_")
            or self.mse_resid_ is None
        ):
            # Fallback: return preds as bounds if interval calc not possible
            return preds, preds, preds

        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X_arr])

        # Variance of prediction: MSE * (1 + x_0^T (X^T X)^-1 x_0)
        # leverage = x_0^T (X^T X)^-1 x_0
        leverage = np.sum(X_design.dot(self.xtx_inv_) * X_design, axis=1)

        # Ensure non-negative variance (numerical issues might cause tiny negative values)
        leverage = np.maximum(leverage, 0)

        pred_var = self.mse_resid_ * (1 + leverage)
        pred_std = np.sqrt(pred_var)

        alpha = 1 - confidence
        t_score = stats.t.ppf(1 - alpha / 2, self.dof_)

        margin = t_score * pred_std
        lower = preds - margin
        upper = preds + margin

        return preds, lower, upper

    def predict(self, X):
        X_arr = np.asarray(X)
        if getattr(self, "model", None) == "closed_form_linear":
            Xb = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
            return Xb.dot(self.coef_)
        return self.model.predict(X_arr)

    def evaluate(self, X, y) -> dict:
        pred = self.predict(X)

        # Ensure inputs are numpy arrays for consistent indexing/math
        y_arr = np.asarray(y)
        pred_arr = np.asarray(pred)

        mae = (
            mean_absolute_error(y_arr, pred_arr)
            if SKLEARN_AVAILABLE
            else float(np.mean(np.abs(y_arr - pred_arr)))
        )
        rmse = (
            np.sqrt(mean_squared_error(y_arr, pred_arr))
            if SKLEARN_AVAILABLE
            else float(np.sqrt(np.mean((y_arr - pred_arr) ** 2)))
        )
        r2 = (
            r2_score(y_arr, pred_arr)
            if SKLEARN_AVAILABLE
            else float(
                1
                - np.sum((y_arr - pred_arr) ** 2)
                / np.sum((y_arr - np.mean(y_arr)) ** 2)
            )
        )

        # Calculate MAPE (Mean Absolute Percentage Error)
        # Filter out zero or near-zero actual values to avoid infinity
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

    def save(self, path: str):
        if JOBLIB_AVAILABLE and not (self.model == "closed_form_linear"):
            joblib.dump({"model": self.model, "model_type": self.model_type}, path)
        else:
            # save closed form coefficients
            np.savez(
                path, model_type=self.model_type, coef=getattr(self, "coef_", None)
            )

    @classmethod
    def load(cls, path: str) -> "Predictor":
        if JOBLIB_AVAILABLE:
            data = joblib.load(path)
            inst = cls(model_type=data.get("model_type", "linear"))
            inst.model = data.get("model")
            return inst
        else:
            npz = np.load(path, allow_pickle=True)
            inst = cls(model_type=str(npz.get("model_type", "linear")))
            inst.coef_ = npz.get("coef")
            inst.model = "closed_form_linear"
            return inst

    def get_coefficients(self, feature_names: list[str] = None):
        """Return model coefficients as a dict if available."""
        coefs_dict = {}
        intercept = None

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(100)]  # Dummy fallback


        if isinstance(self.model, Pipeline) and self.model_type == "polynomial":
            linear_model = self.model.named_steps["linearregression"]
            poly = self.model.named_steps["polynomialfeatures"]
            try:
                out_names = poly.get_feature_names_out(feature_names)
            except Exception:
                out_names = [f"poly_feat_{i}" for i in range(len(linear_model.coef_))]

            vals = linear_model.coef_
            intercept = linear_model.intercept_
            coefs_dict = dict(zip(out_names, vals))

        elif self.model_type == "gam" and PYGAM_AVAILABLE:
            # GAM coefficients are for basis functions, which are not very interpretable directly.
            # We skip returning them to avoid cluttering the output.
            coefs_dict = {}
            intercept = None

        elif hasattr(self.model, "coef_"):
            # Linear, Ridge, Lasso
            vals = self.model.coef_
            intercept = self.model.intercept_
            if len(vals) == len(feature_names):
                coefs_dict = dict(zip(feature_names, vals))
            else:
                coefs_dict = {f"feat_{i}": v for i, v in enumerate(vals)}

        elif hasattr(self.model, "feature_importances_"):
            # RF, XGB, LGBM
            vals = self.model.feature_importances_
            if len(vals) == len(feature_names):
                coefs_dict = dict(zip(feature_names, vals))
            else:
                coefs_dict = {f"feat_{i}": v for i, v in enumerate(vals)}

        return coefs_dict, intercept
