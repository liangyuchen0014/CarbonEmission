import os
import pandas as pd


def deduplicate_and_average(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path)
            # 按 currentTime 分组，对数值列取均值
            if "currentTime" in df.columns:
                # 只对数值列做均值，保留 currentTime
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                grouped = df.groupby("currentTime", as_index=False)[numeric_cols].mean()
                # 如果有非数值列且需要保留，可以补充处理
                # 保存去重后的文件
                out_path = os.path.join(output_dir, file)
                grouped.to_csv(out_path, index=False)
                print(f"Processed and saved: {out_path}")
            else:
                print(f"No 'currentTime' column in {file_path}, skipped.")


if __name__ == "__main__":
    # 对 CarRev 目录去重
    deduplicate_and_average("./CarRev", "./CarRev_dedup")
    # 对 CarTor 目录去重
    deduplicate_and_average("./CarTor", "./CarTor_dedup")
