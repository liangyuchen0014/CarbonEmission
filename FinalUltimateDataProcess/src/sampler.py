import pandas as pd
import numpy as np


def sample_by_step(
    df: pd.DataFrame,
    time_col: str = "__time",
    speed_col: str = "speed",
    power_col: str = "power",
    acc_col: str = "accumulated_usage",
    min_rows_in_window: int = 10,
    max_interval_minutes: int = 15,
    min_coverage_rate: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Sample DataFrame by finding maximal valid windows.

    Constraints:
    1. min_rows_in_window: Minimum number of rows in a sample.
    2. max_interval_minutes: Max allowed gap between adjacent rows in a sample.
    3. min_coverage_rate: rows / total_minutes >= min_coverage_rate.

    Strategy:
    - Iterate through data.
    - Expand window as much as possible while satisfying max_interval constraint.
    - Once a break (large gap) is found or end of data, check if the accumulated window satisfies min_rows and coverage.
    - If valid, add sample.
    - Note: The user requested "maximize len(window)".
      We will segment the data by `max_interval` breaks first.
      Each segment is a candidate.
      If a segment is valid (rows >= min, coverage >= 50%), we take it as one sample.
      (If a segment is too long? The prompt says "maximize len(window)", so we take the whole segment if valid).
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in df columns")

    # Ensure datetime and sort
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # Calculate time diffs to identify breaks
    # diff[i] = time[i] - time[i-1]
    # We want to group rows where diff <= max_interval
    # The first row of a group is where diff > max_interval or it's the very first row

    # Convert to minutes for easier comparison
    # We'll iterate manually or use vectorization to find breaks

    times = df[time_col]
    diffs = times.diff().dt.total_seconds() / 60.0

    # Identify where the gap is larger than max_interval_minutes
    # The first element diff is NaN, fill with 0 or treat as start of group
    is_break = (diffs > max_interval_minutes) | (diffs.isna())

    # Assign group IDs
    group_ids = is_break.cumsum()
    n_groups = group_ids.nunique()
    group_sizes = df.groupby(group_ids).size().tolist()
    print("number of groups before trim:", n_groups)
    print("group sizes before trim:", group_sizes)

    samples = []

    def get_merged_road_type(sub_df: pd.DataFrame) -> str:
        """Return the most frequent road_type in the time window (group)."""
        if "road_type" not in sub_df.columns:
            return "unknown"
        # mode() returns a Series (can be multiple modes or empty)
        modes = sub_df["road_type"].mode()
        if not modes.empty:
            return str(modes.iloc[0])
        return "unknown"

    for _, group in df.groupby(group_ids):
        # Iteratively trim from ends to improve coverage rate
        while len(group) >= min_rows_in_window:
            start_time = group[time_col].iloc[0]
            end_time = group[time_col].iloc[-1]

            total_minutes = (end_time - start_time).total_seconds() / 60.0

            if total_minutes <= 0:
                coverage_rate = 1.0 if len(group) > 0 else 0.0
            else:
                coverage_rate = len(group) / total_minutes

            if coverage_rate >= min_coverage_rate:
                break

            # Try to trim the end with larger gap
            diff_start = (
                group[time_col].iloc[1] - group[time_col].iloc[0]
            ).total_seconds()
            diff_end = (
                group[time_col].iloc[-1] - group[time_col].iloc[-2]
            ).total_seconds()

            if diff_start >= diff_end:
                group = group.iloc[1:]
            else:
                group = group.iloc[:-1]

        if len(group) < min_rows_in_window:
            continue

        # Final check for valid duration before usage calculation
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        total_minutes = (end_time - start_time).total_seconds() / 60.0

        if total_minutes <= 0:
            continue

        # Calculate features and label
        if acc_col in group.columns:
            acc_start = group[acc_col].iloc[0]
            acc_end = group[acc_col].iloc[-1]
            # Label: usage per minute
            usage_rate = (acc_end - acc_start) / total_minutes
        else:
            usage_rate = np.nan

        # Skip invalid usage_rate (NaN/inf) or negative values
        if not np.isfinite(usage_rate) or usage_rate < 0:
            continue

        speed_mean = group[speed_col].mean() if speed_col in group.columns else np.nan
        power_mean = group[power_col].mean() if power_col in group.columns else np.nan

        road_type = get_merged_road_type(group)
        samples.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "speed_mean": float(speed_mean) if not pd.isna(speed_mean) else np.nan,
                "power_mean": float(power_mean) if not pd.isna(power_mean) else np.nan,
                "usage_rate": float(usage_rate) if not pd.isna(usage_rate) else np.nan,
                "road_type": road_type,
                "n_rows": len(group),
                "coverage_rate": coverage_rate,
                "sample_weight": (end_time - start_time).total_seconds() / 60.0,
            }
        )

    print("number of groups after trim:", len(samples))
    print("group sizes after trim:", [s["n_rows"] for s in samples])

    # Update metadata to reflect the valid samples actually returned
    metadata = {
        "number_of_groups": len(samples),
        "group_sizes": [s["n_rows"] for s in samples],
    }
    # mean normalization
    for s in samples:
        s["sample_weight"] = np.sqrt(s["sample_weight"])
    mean_weight = np.mean([s["sample_weight"] for s in samples]) if samples else 1.0
    for s in samples:
        s["sample_weight"] = (
            s["sample_weight"] / mean_weight if mean_weight > 0 else s["sample_weight"]
        )
    sampled = pd.DataFrame(samples)

    return sampled, metadata
