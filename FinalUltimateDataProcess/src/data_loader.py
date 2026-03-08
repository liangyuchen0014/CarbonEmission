import os
import glob
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def load_csvs(path: str) -> pd.DataFrame:
    """Load a single CSV or all CSVs in a directory and return a concatenated DataFrame.
    The function will try to parse common timestamp columns if present.

    suggested usage:
        df = load_csvs("data/LZGJLG846PX051993.csv")  # load a single CSV

    """
    print("Loading CSV(s) from path:", path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        # !!!NOT SO GOOD:DO NOT CONCAT DIRECTLY, BECAUSE TIME IS DIFFERENT IN EACH FILE
    elif os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        raise FileNotFoundError(path)

    time_cols = [c for c in df.columns if c == "Time"]
    if time_cols:
        df["__time"] = pd.to_datetime(df[time_cols[0]])
    else:
        # If there is a column named index-like or the dataframe index is datetime-like, try that
        try:
            df["__time"] = pd.to_datetime(df.index)
        except Exception:
            # As last resort, create a monotonic time based on row number (not ideal)
            df["__time"] = pd.RangeIndex(len(df))

    df = df.sort_values("__time").reset_index(drop=True)
    return df
