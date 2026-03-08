import os
import pandas as pd
import glob

source_dict1 = "../每辆车各自的历史速度&累计里程-20251113_from每辆车各自的历史轨迹"
source_dict2 = "../每辆车各自的历史速度&累计油耗"
target_dict = "../每辆车各自的历史速度&累计油耗"

# 确保目标目录存在
os.makedirs(target_dict, exist_ok=True)

# 获取 source_dict1 中的所有 CSV 文件
csv_files1 = glob.glob(os.path.join(source_dict1, "*.csv"))

for csv_file1 in csv_files1:
    # 获取文件名（不含路径）
    filename = os.path.basename(csv_file1)
    csv_file2 = os.path.join(source_dict2, filename)

    # 检查 source_dict2 中是否存在同名文件
    if not os.path.exists(csv_file2):
        print(f"跳过 {filename}：在 {source_dict2} 中未找到对应文件。")
        continue

    try:
        # 读取两个 CSV 文件
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)

        # 检查必要的列是否存在
        if "locationTime" not in df1.columns or "accumulated_usage" not in df1.columns:
            print(
                f"跳过 {filename}：{csv_file1} 缺少必要的列（locationTime 或 accumulated_usage）。"
            )
            continue
        if "locationTime" not in df2.columns or "speed" not in df2.columns:
            print(
                f"跳过 {filename}：{csv_file2} 缺少必要的列（locationTime 或 speed）。"
            )
            continue

        # 合并数据：基于 locationTime 内连接
        merged_df = pd.merge(
            df1[["locationTime", "accumulated_usage"]],
            df2[["locationTime", "speed"]],
            on="locationTime",
            how="inner",
        )

        # 如果合并后没有数据，跳过
        if merged_df.empty:
            print(f"跳过 {filename}：合并后无有效数据。")
            continue

        # 保存到目标目录
        output_path = os.path.join(target_dict, filename)
        merged_df.to_csv(output_path, index=False)
        print(
            f"已合并并保存：{output_path}，共 {len(merged_df)} 条记录。历史速度:{len(df1)},累计油耗{len(df2)}"
        )

    except Exception as e:
        print(f"处理 {filename} 时出错：{e}")
        continue
