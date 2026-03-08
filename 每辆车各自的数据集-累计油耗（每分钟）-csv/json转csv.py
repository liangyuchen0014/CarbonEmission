import os
import json
import csv
import glob
import re
from datetime import datetime, timedelta
import pandas as pd

source_dict = "../每辆车各自的数据集-累计油耗（每分钟）"
target_dict = "."
"""
source file example:
{
    "LZGCD2L16PX016618": {
        "c_vin17": "LZGCR2868PX033622",
        ...,
        "accumulated_usage": [61403.0, 61403.5, ...],
        "time": ["2024-09-01 13:44:37", "2024-09-01 13:45:37", ...],
    }
}
target file example:
locationTime,accumulated_usage
"""


def process_csv(file_path):
    """
    对CSV文件进行后处理：
    1. 将locationTime舍入到最近的分钟。
    2. 按locationTime聚合，计算accumulated_usage的平均值。
    """
    try:
        df = pd.read_csv(file_path)
        if "locationTime" not in df.columns:
            print(f"文件 {file_path} 中缺少 'locationTime' 列，跳过处理。")
            return

        # 将locationTime转换为datetime对象，无效的将变为NaT
        df["locationTime"] = pd.to_datetime(df["locationTime"], errors="coerce")
        df.dropna(subset=["locationTime"], inplace=True)

        # 舍入到最近的分钟
        df["locationTime"] = df["locationTime"].dt.round("min")

        # 按时间聚合，计算累计油耗平均值
        df_agg = df.groupby("locationTime")["accumulated_usage"].mean().reset_index()

        # 格式化时间并保存
        df_agg["locationTime"] = df_agg["locationTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_agg.to_csv(file_path, index=False, encoding="utf-8")
        print(f"已处理并保存文件: {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


os.makedirs(target_dict, exist_ok=True)

# 支持目录下的 .json 或 .JSON 文件
for json_path in glob.glob(os.path.join(source_dict, "*.json")) + glob.glob(
    os.path.join(source_dict, "*.JSON")
):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        # 如果某个文件不能读取或解析
        print(f"无法读取或解析 {json_path}，跳过。")
        continue

    if isinstance(obj, dict):
        vid = obj["c_vin17"]
        times = obj["time"]
        accumulated_usages = obj["accumulated_usage"]

        if not times or not accumulated_usages:
            print(f"车辆 {vid} 缺少时间或速度数据，跳过。")
            continue
        if len(times) != len(accumulated_usages):
            print(f"车辆 {vid} 的数据长度不匹配！")
            n = min(len(times), len(accumulated_usages))
        else:
            n = len(times)
        # 安全的文件名
        safe_vid = re.sub(r"[^\w\-_\. ]", "_", str(vid))
        out_path = os.path.join(target_dict, f"{safe_vid}.csv")

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["locationTime", "accumulated_usage"])
                for i in range(n):
                    t = times[i]
                    s = accumulated_usages[i]

                    writer.writerow([t, s])
            # 对生成的文件进行后处理
            process_csv(out_path)

        except Exception as e:
            # 如果写入或处理失败，跳过此车辆
            print(f"写入或处理车辆 {vid} 的数据时出错: {e}")
            continue
