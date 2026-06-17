"""Diagnose low ambient-temperature / low-efficiency non-heating samples."""

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ch5_efficiency_factor_plots import (  # noqa: E402
    PLANT_METRICS_PATH,
    RAW_DATA_PATH,
    build_factor_data,
    split_conditions,
)


def print_table(title, table):
    print(f"\n{title}")
    print(table.to_string(float_format=lambda x: f"{x:8.2f}"))


def main():
    raw = pd.read_csv(RAW_DATA_PATH)
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    factor_data = build_factor_data(raw, metrics)
    selected_raw = raw.iloc[metrics["idx"].astype(int)].reset_index(drop=True)
    non_factor, heat_factor = split_conditions(factor_data)

    gt1_run = metrics["GT1.是否运行"].astype(str).str.lower().isin(["true", "1", "yes"])
    gt2_run = metrics["GT2.是否运行"].astype(str).str.lower().isin(["true", "1", "yes"])
    extra = pd.DataFrame(
        {
            "idx": metrics["idx"],
            "运行燃机台数": gt1_run.astype(int) + gt2_run.astype(int),
            "GT1运行": gt1_run,
            "GT2运行": gt2_run,
            "GT1燃机热效率": metrics["GT1.燃机热效率"] * 100,
            "GT2燃机热效率": metrics["GT2.燃机热效率"] * 100,
            "GT1燃料能量": metrics["GT1.燃料能量"] / 1e6,
            "GT2燃料能量": metrics["GT2.燃料能量"] / 1e6,
            "GT总燃料能量": (metrics["GT1.燃料能量"].fillna(0) + metrics["GT2.燃料能量"].fillna(0)) / 1e6,
            "PLANT供热原值": metrics["PLANT.供热"] / 1e6,
            "ST总体等熵效率": metrics["ST.汽机总体等熵效率"] * 100,
            "ST高压缸效率": metrics["ST.高压缸等熵效率"] * 100,
            "ST中压缸效率": metrics["ST.中压缸等熵效率"] * 100,
            "ST低压缸效率": metrics["ST.低压缸等熵效率"] * 100,
            "HRSG1换热效率": metrics["HRSG1.余热锅炉换热效率"] * 100,
            "HRSG2换热效率": metrics["HRSG2.余热锅炉换热效率"] * 100,
            "凝汽器压力": selected_raw.iloc[:, 109] / 1000,
            "凝汽器温度": selected_raw.iloc[:, 110] - 273.15,
            "低压缸进汽压力": selected_raw.iloc[:, 107] / 1000,
            "热网抽汽流量": selected_raw.iloc[:, 114],
            "热网换热量": selected_raw.iloc[:, 115] / 1e6,
            "原始燃料流量1": selected_raw.iloc[:, 131],
            "原始燃料流量2": selected_raw.iloc[:, 130],
        }
    )
    data = pd.concat([factor_data, extra], axis=1)
    non_heating = data.loc[non_factor.index].copy()
    heating = data.loc[heat_factor.index].copy()

    print(f"raw shape: {raw.shape}; metrics shape: {metrics.shape}")
    print(f"non-heating samples: {len(non_heating)}; heating samples: {len(heating)}")
    print(
        "non-heating ambient temperature range: "
        f"{non_heating['环境温度'].min():.2f} to {non_heating['环境温度'].max():.2f} C"
    )
    print(
        "non-heating power efficiency range: "
        f"{non_heating['联合循环发电效率'].min():.2f} to "
        f"{non_heating['联合循环发电效率'].max():.2f} %"
    )

    factor_cols = [
        "环境温度",
        "环境压力",
        "环境湿度",
        "整机出力",
        "燃机总出力",
        "汽机出力",
        "高压主蒸汽流量",
        "中压主蒸汽流量",
        "空气流量",
        "透平排气温度",
    ]
    corr = non_heating[factor_cols + ["联合循环发电效率"]].corr(numeric_only=True)[
        "联合循环发电效率"
    ].sort_values()
    print_table("Correlation with non-heating power efficiency", corr.to_frame("r"))

    non = non_heating.copy()
    non["temp_bin"] = pd.qcut(non["环境温度"], 6, duplicates="drop")
    temp_bins = non.groupby("temp_bin", observed=True).agg(
        n=("联合循环发电效率", "size"),
        temp_mean=("环境温度", "mean"),
        eff_mean=("联合循环发电效率", "mean"),
        eff_med=("联合循环发电效率", "median"),
        power_mean=("整机出力", "mean"),
        gt_mean=("燃机总出力", "mean"),
        st_mean=("汽机出力", "mean"),
        air_mean=("空气流量", "mean"),
        hp_flow=("高压主蒸汽流量", "mean"),
        exh_mean=("透平排气温度", "mean"),
    )
    print_table("Non-heating samples grouped by ambient-temperature quantile", temp_bins)

    low_temp_cut = non_heating["环境温度"].quantile(0.20)
    low_temp = non_heating[non_heating["环境温度"] <= low_temp_cut]
    low_eff = low_temp[
        low_temp["联合循环发电效率"] <= low_temp["联合循环发电效率"].quantile(0.25)
    ]
    high_eff = low_temp[
        low_temp["联合循环发电效率"] >= low_temp["联合循环发电效率"].quantile(0.75)
    ]
    warm = non_heating[
        non_heating["环境温度"] >= non_heating["环境温度"].quantile(0.60)
    ]

    describe_cols = [
        "环境温度",
        "联合循环发电效率",
        "整机出力",
        "燃机总出力",
        "汽机出力",
        "空气流量",
        "高压主蒸汽流量",
        "中压主蒸汽流量",
        "透平排气温度",
        "环境湿度",
        "环境压力",
    ]
    for title, subset in [
        ("Low-temperature / low-efficiency subset", low_eff),
        ("Low-temperature / high-efficiency subset", high_eff),
        ("Warmer subset", warm),
    ]:
        desc = subset[describe_cols].describe(percentiles=[0.25, 0.5, 0.75])
        print(f"\n{title}: n={len(subset)}")
        print(desc.loc[["mean", "25%", "50%", "75%"]].to_string(float_format=lambda x: f"{x:8.2f}"))

        diagnostics = subset[
            [
                "idx",
                "运行燃机台数",
                "GT1燃机热效率",
                "GT2燃机热效率",
                "GT总燃料能量",
                "PLANT供热原值",
                "ST总体等熵效率",
                "ST高压缸效率",
                "ST中压缸效率",
                "ST低压缸效率",
                "HRSG1换热效率",
                "HRSG2换热效率",
                "凝汽器压力",
                "凝汽器温度",
                "低压缸进汽压力",
                "热网抽汽流量",
                "热网换热量",
                "原始燃料流量1",
                "原始燃料流量2",
            ]
        ].describe(percentiles=[0.25, 0.5, 0.75])
        print(diagnostics.loc[["mean", "25%", "50%", "75%"]].to_string(float_format=lambda x: f"{x:8.2f}"))

        print("running GT count:")
        print(subset["运行燃机台数"].value_counts(dropna=False).sort_index().to_string())

    print("\nHeating-identification checks")
    checks = [
        ("低温低效", low_eff),
        ("低温高效", high_eff),
        ("全部非供热", non_heating),
        ("已识别供热", heating),
    ]
    for title, subset in checks:
        raw_heat = subset["PLANT供热原值"]
        heat_flow = subset["热网抽汽流量"]
        print(f"\n{title}: n={len(subset)}")
        print(f"PLANT供热原值 > 1 MW: {(raw_heat > 1).sum()}")
        print(f"PLANT供热原值 < 0 MW: {(raw_heat < 0).sum()}")
        print(f"热网换热量 > 1 MW: {(subset['热网换热量'] > 1).sum()}")
        print(f"热网抽汽流量 > 1 kg/s: {(heat_flow > 1).sum()}")
        print(
            subset[
                [
                    "供热量",
                    "PLANT供热原值",
                    "热网换热量",
                    "热网抽汽流量",
                    "汽机出力",
                    "低压缸进汽压力",
                    "凝汽器压力",
                ]
            ]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .loc[["mean", "25%", "50%", "75%"]]
            .to_string(float_format=lambda x: f"{x:8.2f}")
        )


if __name__ == "__main__":
    main()
