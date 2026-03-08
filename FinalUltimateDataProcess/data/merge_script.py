import os
import pandas as pd
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

source_dict1 = "../每辆车各自的历史速度&累计油耗/data"
source_dict2 = "../Rev_Tor_Power"
target_dict = "./data"
"""
source_dict1下的.csv文件格式为：
locationTime,accumulated_usage,speed
2024-09-01 00:00:00,26606.0,3.0
……
source_dict2下的.csv文件格式为：
currentTime,Rev,Torque%,Torque,power
2024-08-17 10:19:00+00:00,800.49,20.94736842,213.663157884,17.910760082368675
……
目标文件格式为：
Time,accumulated_usage,speed,power
2024-09-01 00:00:00,26606.0,3.0,XX.XXXX
"""


def find_matches(files1, files2):
    """返回匹配对列表 (f1, f2)。匹配规则：
    - 如果任一尾部是另一个的后缀（endswith），则认为匹配
    一旦某个文件被匹配，默认把它标记为已用（避免重复匹配）。
    """
    used1 = set()
    used2 = set()
    pairs = []

    basename1 = {f: Path(f).stem for f in files1}
    basename2 = {f: Path(f).stem for f in files2}

    for f2 in files2:
        b2 = basename2[f2]
        tail2 = b2.rsplit("-", 1)[-1]
        matched = False
        for f1 in files1:
            if f1 in used1 or f2 in used2:
                continue
            b1 = basename1[f1]
            if tail2 and b1.endswith(tail2):
                pairs.append((f1, f2))
                used1.add(f1)
                used2.add(f2)
                matched = True
                break
        if not matched:
            logging.debug("未为 %s 找到匹配项", f2)
    logging.info("匹配对:  %s", pairs)
    return pairs


def merge_pair(f1, f2, out_dir):
    """按 time 字段合并 f1（source1）和 f2（source2），并把结果写到 out_dir。
    返回生成的输出文件路径（字符串）或 None（如果没有匹配行）。
    """
    try:
        df1 = pd.read_csv(f1)
    except Exception as e:
        logging.error("读取 %s 失败: %s", f1, e)
        return None

    try:
        df2 = pd.read_csv(f2)
    except Exception as e:
        logging.error("读取 %s 失败: %s", f2, e)
        return None

    # 解析时间为 UTC-aware datetime（失败则为 NaT）
    if "locationTime" not in df1.columns:
        logging.error("文件 %s 无列 locationTime，跳过", f1)
        return None
    if "currentTime" not in df2.columns:
        logging.error("文件 %s 无列 currentTime，跳过", f2)
        return None

    df1["Time"] = pd.to_datetime(df1["locationTime"], utc=True)
    df2["Time"] = pd.to_datetime(df2["currentTime"], utc=True)
    # print(df1["Time"])
    # print("-----------", "df2")
    # print(df2["Time"])
    # 取关心的列（防止列名冲突），并丢弃时间解析失败的行
    left = df1[["Time", "accumulated_usage", "speed"]].dropna(subset=["Time"])
    right = df2[["Time", "power"]].dropna(subset=["Time"])

    if left.empty or right.empty:
        logging.info("在 %s 和 %s 中没有有效的时间行可用于合并", f1, f2)
        return None

    merged = pd.merge(left, right, on="Time", how="inner")

    if merged.empty:
        logging.info("合并后无匹配行：%s <-> %s", f1, f2)
        return None

    # 将 Time 列格式化为没有时区信息的字符串（UTC 时间）
    merged["Time"] = (
        merged["Time"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    b1 = Path(f1).stem
    out_name = f"{b1}.csv"
    out_path = os.path.join(out_dir, out_name)

    # 指定列顺序：Time,accumulated_usage,speed,power
    merged = merged[["Time", "speed", "power", "accumulated_usage"]]

    merged.to_csv(out_path, index=False)
    logging.info("已写: %s (%d 行)", out_path, len(merged))
    return out_path


def main(src1=source_dict1, src2=source_dict2, tgt=target_dict):
    src1 = os.path.abspath(src1)
    src2 = os.path.abspath(src2)
    tgt = os.path.abspath(tgt)

    files1 = glob.glob(os.path.join(src1, "*.csv"))
    files2 = glob.glob(os.path.join(src2, "*.csv"))

    logging.info("找到 %d files in %s", len(files1), src1)
    logging.info("找到 %d files in %s", len(files2), src2)

    if not files1 or not files2:
        logging.warning("其中一个目录没有 CSV 文件，退出")
        return

    pairs = find_matches(files1, files2)
    logging.info("匹配到 %d 对文件", len(pairs))

    if not pairs:
        logging.warning("没有匹配的文件对，检查文件名后缀匹配规则。")

    for f1, f2 in pairs:
        merge_pair(f1, f2, tgt)


if __name__ == "__main__":
    # 允许通过环境或命令行修改路径，简单起见直接使用全局变量
    main()
