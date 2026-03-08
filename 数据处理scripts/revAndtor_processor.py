import os

import pandas as pd


def SearchMaxTorque(vin_4, vin_info) -> float:
    for i in vin_info:
        if i["c_vin17"].endswith(vin_4):
            return float(i["最大扭矩(N.m)"])
    return -1.0


rev_file_path = "./CarRev_dedup"
tor_file_path = "./CarTor_dedup"
output_file_path = "./Rev_Tor_Power"
file_names = []

# get all the file's name under rev_file_path
for file in os.listdir(rev_file_path):
    if file.endswith(".csv"):
        file_names.append(file)

vin_info_path = "./车辆信息_new.json"
import json

with open(vin_info_path, "r", encoding="utf-8") as f:
    vin_info = json.load(f)

for file_name in file_names:
    rev_file = os.path.join(rev_file_path, file_name)
    tor_file = os.path.join(tor_file_path, file_name)
    output_file_name = file_name.split("_")[1]
    print(output_file_name)
    output_file = os.path.join(output_file_path, output_file_name)

    vin_4 = file_name.split(".")[0][-4:]
    max_torque = SearchMaxTorque(vin_4, vin_info)
    assert max_torque > 0, f"Max torque not found for VIN ending {vin_4}"
    if os.path.exists(rev_file) and os.path.exists(tor_file):
        try:
            rev_df = pd.read_csv(rev_file, parse_dates=["currentTime"])
            tor_df = pd.read_csv(tor_file, parse_dates=["currentTime"])
            # Merge dataframes on 'currentTime' (精确匹配)
            merged_df = pd.merge(rev_df, tor_df, on="currentTime", how="inner")
            merged_df.columns = ["currentTime", "Rev", "Torque%"]
            merged_df["Torque"] = merged_df["Torque%"] * max_torque / 100
            merged_df["power"] = merged_df["Rev"] * merged_df["Torque"] / 9549.3

            merged_df.to_csv(output_file, index=False)
            print(
                f"Merged and saved: {output_file}.rev_rows={len(rev_df)}, tor_rows={len(tor_df)}, merged_rows={len(merged_df)}"
            )
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
    else:
        print(f"Files for {file_name} not found in both directories.")
