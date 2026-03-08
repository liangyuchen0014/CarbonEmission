import os
import pandas as pd
import warnings

# 忽略 openpyxl 的 default style 警告
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

tor = "/data/CarbonEmission/CarTor"
# 获取所有时间子目录
subdirs = [d for d in os.listdir(tor) if os.path.isdir(os.path.join(tor, d))]
# tor_file_list = os.listdir(os.path.join(tor, "2024-0826-0901"))
tor_file_list = ["Rev_Car61-0128.xlsx", "Rev_Car72-3680.xlsx"]
for file_name in tor_file_list:
    file_paths = []
    # 收集所有子目录下的同名文件路径
    subdirs.sort(reverse=True)  # 按时间降序处理
    for subdir in subdirs:
        file_path = os.path.join(tor, subdir, file_name)
        if os.path.exists(file_path):
            file_paths.append(file_path)
    # 合并所有同名文件的数据
    dfs = []
    for fp in file_paths:
        print("processing :" + fp)
        try:
            df = pd.read_excel(fp)
            df.drop(columns=["carNo", "vin", "vin17", "type"], inplace=True)
            # 转换 currentTime 为 datetime，重采样
            df["currentTime"] = pd.to_datetime(df["currentTime"], format="%Y%m%d%H%M%S")
            df = (
                df.set_index("currentTime")
                .resample("min")
                .mean(numeric_only=True)
                .dropna()
                .reset_index()
            )
            # 转为 ISO8601 Z 格式
            df["currentTime"] = df["currentTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            dfs.append(df)
        except Exception as e:
            print(f"读取 {fp} 失败: {e}")
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        out_csv = os.path.join(
            "/data/CarbonEmission/NewCarTor", file_name.replace(".xlsx", ".csv")
        )
        merged_df.to_csv(out_csv, index=False)
        print(f"已保存: {out_csv}")
    else:
        print(f"未找到任何 {file_name} 文件")
