import pandas as pd
import numpy as np
from typing import Tuple
from src.data_loader import load_csvs
from src.sampler import sample_by_step
from src.utils import get_logger

logger = get_logger(__name__)


def build_dataset_from_source(
    source: str,
    time_col: str = "Time",
    speed_col: str = "speed",
    power_col: str = "power",
    acc_col: str = "accumulated_usage",
    min_rows_in_window: int = 10,
    max_interval_minutes: int = 15,
    min_coverage_rate: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    # Check if source is a comma-separated list of files (PINs)
    if "," in source:
        sources = [s.strip() for s in source.split(",")]
        all_sampled = []
        total_groups = 0
        all_group_sizes = []

        for s in sources:
            try:
                df = load_csvs(s)
                sampled, metadata = sample_by_step(
                    df,
                    time_col=time_col,
                    speed_col=speed_col,
                    power_col=power_col,
                    acc_col=acc_col,
                    min_rows_in_window=min_rows_in_window,
                    max_interval_minutes=max_interval_minutes,
                    min_coverage_rate=min_coverage_rate,
                )
                if not sampled.empty:
                    all_sampled.append(sampled)
                    total_groups += metadata.get("number_of_groups", 0)
                    all_group_sizes.extend(metadata.get("group_sizes", []))
            except Exception as e:
                logger.warning(f"Failed to process source {s}: {e}")
                continue

        if not all_sampled:
            return pd.DataFrame(), {"number_of_groups": 0, "group_sizes": []}

        final_sampled = pd.concat(all_sampled, ignore_index=True)
        final_metadata = {
            "number_of_groups": total_groups,
            "group_sizes": all_group_sizes,
        }

        # drop samples with NaN usage_rate
        final_sampled = final_sampled.dropna(
            subset=["usage_rate", "speed_mean", "power_mean"]
        ).reset_index(drop=True)

        return final_sampled, final_metadata

    # Original logic for single file or directory
    df = load_csvs(source)

    sampled, metadata = sample_by_step(
        df,
        time_col=time_col,
        speed_col=speed_col,
        power_col=power_col,
        acc_col=acc_col,
        min_rows_in_window=min_rows_in_window,
        max_interval_minutes=max_interval_minutes,
        min_coverage_rate=min_coverage_rate,
    )
    # drop samples with NaN usage_rate
    sampled = sampled.dropna(
        subset=["usage_rate", "speed_mean", "power_mean","road_type"]
    ).reset_index(drop=True)
    return sampled, metadata


# build_dataset_from_source('data/LZGJLG846PX051993.csv')


def features_labels_from_sampled(
    sampled: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    修改点：X 现在返回 DataFrame，包含 road_type，而不是 numpy array。
    预处理（One-Hot等）将在模型内部处理。
    """
    # 确保列存在，如果不存在（例如旧数据），填充默认值
    if "road_type" not in sampled.columns:
        sampled["road_type"] = "其他"

    X = sampled[["speed_mean", "power_mean", "road_type"]].copy()

    y = sampled["usage_rate"].to_numpy(dtype=float)

    weights = (
        sampled["sample_weight"].to_numpy(dtype=float)
        if "sample_weight" in sampled.columns
        else np.ones(len(y))
    )
    return X, y, weights