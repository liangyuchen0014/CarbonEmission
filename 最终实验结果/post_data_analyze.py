import json
from pathlib import Path
import numpy as np

#!/usr/bin/env python3
# File: /data/CarbonEmission/FinalUltimateDataProcess/outputs/post_data_analyze.py
# Requires: python3, matplotlib, numpy

import matplotlib.pyplot as plt


def main():
    results_dir = (
        Path.cwd().parent.joinpath("results")
        if Path.cwd().name == "outputs"
        else Path("./results")
    )
    if not results_dir.exists():
        results_dir = Path("./results")
    out_dir = Path(__file__).resolve().parent.joinpath("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(results_dir.rglob("*.json"))
    if not json_files:
        print(f"No json files found in {results_dir}")
        return

    number_of_groups = []
    all_group_sizes = []  # list of arrays
    labels = []

    for jf in json_files:
        if not jf.name.startswith("LZ"):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        ng = data.get("number_of_groups")
        gos = data.get("group_of_sizes") or data.get("group_sizes") or []
        if isinstance(ng, (int, float)):
            number_of_groups.append(int(ng))
        # ensure numeric list
        """ if isinstance(gos, (list, tuple)):
            arr = np.array(
                [
                    float(x)
                    for x in gos
                    if isinstance(x, (int, float))
                    or (isinstance(x, str) and x.strip() != "")
                ],
                dtype=float,
            )
            if arr.size > 0:
                all_group_sizes.append(arr)
                labels.append(jf.stem)
                # per-file histogram
                plt.figure(figsize=(6, 4))
                plt.hist(arr, bins="auto", color="C0", alpha=0.7)
                plt.title(f"{jf.stem} group_of_sizes")
                plt.xlabel("size")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(
                    out_dir.joinpath(f"{jf.stem}_group_sizes_hist.png"), dpi=150
                )
                plt.close() """

    # histogram of number_of_groups across files
    if number_of_groups:
        print("number_of_groups:" + ", ".join(str(x) for x in number_of_groups))
        plt.figure(figsize=(6, 4))
        minv = min(number_of_groups)
        maxv = max(number_of_groups)
        # choose a larger bin width (at least 2, scaled to the range)
        rng = maxv - minv
        bin_width = max(2, int(np.ceil(rng / 8)) if rng > 0 else 2)
        bins = np.arange(minv, maxv + bin_width, bin_width)
        plt.hist(number_of_groups, bins=bins, color="C1", rwidth=0.8)
        plt.xlabel("number_of_groups")
        plt.ylabel("count")
        plt.title("Distribution of number_of_groups")
        plt.xticks(bins)
        plt.tight_layout()
        plt.savefig(out_dir.joinpath("number_of_groups_hist.png"), dpi=150)
        plt.close()
        return

    # overlay trends of group_of_sizes histograms (density)
    if all_group_sizes:
        # common bins
        global_min = min(arr.min() for arr in all_group_sizes)
        global_max = max(arr.max() for arr in all_group_sizes)
        if global_min == global_max:
            bins = np.linspace(global_min - 0.5, global_max + 0.5, 10)
        else:
            bins = np.linspace(global_min, global_max, 100)
        mids = (bins[:-1] + bins[1:]) / 2.0

        plt.figure(figsize=(8, 5))
        for arr, label in zip(all_group_sizes, labels):
            hist, _ = np.histogram(arr, bins=bins, density=True)
            plt.plot(mids, hist, label=label, alpha=0.8, linewidth=1)
        plt.xlabel("group size")
        plt.ylabel("density")
        plt.title("Overlayed group_of_sizes density trends")
        plt.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=(0, 0, 0.78, 1))
        plt.savefig(out_dir.joinpath("group_of_sizes_overlay.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
