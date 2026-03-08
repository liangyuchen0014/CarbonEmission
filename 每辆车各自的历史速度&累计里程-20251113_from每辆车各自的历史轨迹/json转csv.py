import os
import json
import csv
import glob
import re
from datetime import datetime, timedelta
import pandas as pd

source_dict = "../每辆车各自的历史轨迹/LZGJL484XPX038050.json"
target_dict = "../每辆车各自的历史速度&累计里程-20251113_from每辆车各自的历史轨迹"
"""
source file example:
{
    "LZGCD2L16PX016618": {
        "speed": [0.0, 11.0, 10.0, 18.0, ...],
        "totalMileage": [0.0, 100.0, 200.0, 300.0, ...],
        "locationTime": [20240801003754.0, 20240801023200.0, ...],
        "lng": [103.834567, 103.835678, ...],
        "lat": [30.123456, 30.124567, ...],
    }
}
target file example:
locationTime,speed,totalMileage
"""


def process_csv(file_path):
    """
    对CSV文件进行后处理：
    1. 将locationTime舍入到最近的分钟。
    2. 按locationTime聚合，计算speed的平均值和totalMileage的平均值（同时进行）。
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

        # 按时间聚合，同时计算 speed 的平均值 和 totalMileage 的最大值（如果存在）
        agg_dict = {}
        if "speed" in df.columns:
            agg_dict["speed"] = "mean"
        else:
            print(f"文件 {file_path} 中缺少 'speed' 列，无法计算速度均值。")

        if "totalMileage" in df.columns:
            agg_dict["totalMileage"] = "mean"
        else:
            # 若没有 totalMileage，则不计算该列，但仍保存 speed
            print(f"文件 {file_path} 中不包含 'totalMileage' 列，将只计算 speed。")

        if not agg_dict:
            print(f"文件 {file_path} 中没有可聚合的列，跳过。")
            return

        df_agg = df.groupby("locationTime").agg(agg_dict).reset_index()

        # 格式化时间并保存
        df_agg["locationTime"] = df_agg["locationTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        # 保证列顺序为 locationTime, speed, totalMileage（若存在）
        cols = ["locationTime"]
        if "speed" in df_agg.columns:
            cols.append("speed")
        if "totalMileage" in df_agg.columns:
            cols.append("totalMileage")

        df_agg.to_csv(file_path, index=False, encoding="utf-8", columns=cols)
        print(f"已处理并保存文件: {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


os.makedirs(target_dict, exist_ok=True)

# 支持目录下的 .json 或 .JSON 文件
# for json_path in glob.glob(os.path.join(source_dict, "*.json")) + glob.glob(
# os.path.join(source_dict, "*.JSON")):
try:
    with open(source_dict, "r", encoding="utf-8") as f:
        obj = json.load(f)
except Exception:
    # 如果某个文件不能读取或解析
    print(f"无法读取或解析 {source_dict}，跳过。")


# obj 预期是 { vehicle_id: { "locationTime": [...], "speed": [...], ... }, ... }
if isinstance(obj, dict):
    for vid, rec in obj.items():
        if not isinstance(rec, dict):
            print(f"车辆 {vid} 的记录不是字典，跳过。")
            continue
        times = rec.get("locationTime") or []
        speeds = rec.get("speed") or []
        totalMileages = rec.get("totalMileage") or []
        print(
            f"处理车辆 {vid}，记录数: 时间={len(times)}, 速度={len(speeds)}, 里程={len(totalMileages)}"
        )
        if not times or not speeds:
            print(f"车辆 {vid} 缺少时间或速度数据，跳过。")
            continue
        if len(times) != len(speeds):
            print(f"车辆 {vid} 的数据长度不匹配！")
            n = min(len(times), len(speeds))
        else:
            n = len(times)
        # 安全的文件名
        safe_vid = re.sub(r"[^\w\-_\. ]", "_", str(vid))
        out_path = os.path.join(target_dict, f"{safe_vid}.csv")

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["locationTime", "speed", "totalMileage"])
                for i in range(n):
                    t = times[i]
                    s = speeds[i]
                    m = totalMileages[i] if i < len(totalMileages) else None
                    # 将类似 20240801003754.0 转为不带小数的字符串
                    if isinstance(t, float) or (
                        isinstance(t, str)
                        and t.replace(".", "", 1).isdigit()
                        and "." in t
                    ):
                        try:
                            s_time = str(t).split(".", 1)[0]  # 去掉小数部分
                            s_time = re.sub(r"\D", "", s_time)  # 只保留数字
                            t_str = datetime.strptime(s_time, "%Y%m%d%H%M%S").strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        except Exception:
                            t_str = str(t)
                    else:
                        t_str = str(t)
                    # 如果没有 totalMileage，用空字符串占位，保持列对齐
                    writer.writerow([t_str, s, "" if m is None else m])

            # 对生成的文件进行后处理
            process_csv(out_path)

        except Exception as e:
            # 如果写入或处理失败，跳过此车辆
            print(f"写入或处理车辆 {vid} 的数据时出错: {e}")
            continue
