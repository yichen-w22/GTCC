"""PPT — 燃气轮机出力机械效率修正与误差诊断.

本脚本打印统计结果并用 plt.show() 显示图形，不保存图片。
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "Energy_Utilization_Analysis"
RAW_DATA_PATH = PROJECT_ROOT / "data_precessing" / "averaged_data_10min.csv"
PLANT_METRICS_PATH = PROJECT_ROOT / "temp" / "plant_metrics.csv"

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

UPPER_MW = 300
MECHANICAL_EFFICIENCY_1 = 0.98
MECHANICAL_EFFICIENCY_2 = 0.99
SUMMER_TEMP_C = 20.0
LOAD_BINS_MW = [100, 200, UPPER_MW]
LOAD_LABELS = ["100-200 MW", "200 MW以上"]


def correct_calculated_power(power: pd.Series) -> pd.Series:
    return power * MECHANICAL_EFFICIENCY_1 * MECHANICAL_EFFICIENCY_2


def ambient_temperature_c(raw_df: pd.DataFrame, idx: pd.Series, col: str) -> pd.Series:
    temp = raw_df.loc[idx, col].reset_index(drop=True)
    if temp.median() > 200:
        temp = temp - 273.15
    return temp


def error_summary(data: pd.DataFrame) -> pd.Series:
    relative_error_pct = data["残差_MW"] / data["实际功率_MW"] * 100
    return pd.Series(
        {
            "样本数": len(data),
            "均方误差_MW2": (data["残差_MW"] ** 2).mean(),
            "均方根误差_MW": ((data["残差_MW"] ** 2).mean()) ** 0.5,
            "平均残差_MW": data["残差_MW"].mean(),
            "平均绝对误差_MW": data["残差_MW"].abs().mean(),
            "残差中位数_MW": data["残差_MW"].median(),
            "平均百分比偏差_%": relative_error_pct.mean(),
            "平均绝对百分比误差_%": relative_error_pct.abs().mean(),
            "百分比误差中位数_%": relative_error_pct.median(),
        }
    )


def relative_rmse_pct(data: pd.DataFrame) -> float:
    relative_error = data["残差_MW"] / data["实际功率_MW"]
    return float(((relative_error**2).mean()) ** 0.5 * 100)


def print_section(title: str, data) -> None:
    print(f"\n{title}")
    print(data.to_string())


def main() -> None:
    metrics = pd.read_csv(PLANT_METRICS_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)
    idx = metrics["idx"].astype(int)

    frames = []
    for unit in (1, 2):
        actual_mw = metrics[f"GT{unit}.燃机实际功"] / 1e6
        calculated_mw = correct_calculated_power(metrics[f"GT{unit}.燃机计算功"]) / 1e6
        temp_c = ambient_temperature_c(raw, idx, f"环境温度_{unit}")
        valid = (calculated_mw > 0) & (calculated_mw < UPPER_MW)

        unit_data = pd.DataFrame(
            {
                "燃机": f"GT{unit}",
                "原始序号": idx,
                "环境温度_C": temp_c,
                "实际功率_MW": actual_mw,
                "修正计算功率_MW": calculated_mw,
            }
        )
        unit_data = unit_data.loc[valid].copy()
        unit_data["残差_MW"] = unit_data["修正计算功率_MW"] - unit_data["实际功率_MW"]
        unit_data["绝对残差_MW"] = unit_data["残差_MW"].abs()
        unit_data["夏季工况"] = unit_data["环境温度_C"] >= SUMMER_TEMP_C
        frames.append(unit_data)

    data = pd.concat(frames, ignore_index=True)

    print(f"机械效率修正：计算功率 * {MECHANICAL_EFFICIENCY_1:.2f} * {MECHANICAL_EFFICIENCY_2:.2f}")
    print(f"夏季工况阈值：环境温度 >= {SUMMER_TEMP_C:.1f} ℃")
    print_section("全部工况误差", error_summary(data))
    print_section("夏季工况误差", error_summary(data.loc[data["夏季工况"]]))

    summary_cols = ["实际功率_MW", "残差_MW"]
    by_unit = data.groupby("燃机", sort=False)[summary_cols].apply(error_summary)
    print_section("按燃机分解", by_unit)

    data["温度分组"] = pd.qcut(data["环境温度_C"], 4, duplicates="drop")
    by_temp = data.groupby("温度分组", observed=True)[summary_cols].apply(error_summary)
    print_section("按环境温度四分位分解", by_temp)

    data["负荷分组"] = pd.qcut(data["实际功率_MW"], 4, duplicates="drop")
    by_load = data.groupby("负荷分组", observed=True)[summary_cols].apply(error_summary)
    print_section("按实际功率四分位分解", by_load)

    gt1 = data.loc[(data["燃机"] == "GT1") & (data["实际功率_MW"] >= LOAD_BINS_MW[0])].copy()
    gt1["季节"] = gt1["夏季工况"].map({True: "夏季工况", False: "冬季工况"})
    gt1["负荷段"] = pd.cut(
        gt1["实际功率_MW"],
        bins=LOAD_BINS_MW,
        labels=LOAD_LABELS,
        include_lowest=True,
        right=False,
    )
    gt1 = gt1.dropna(subset=["负荷段"])
    gt1["相对误差"] = gt1["残差_MW"] / gt1["实际功率_MW"]
    gt1_rrmse_table = gt1.pivot_table(
        values="相对误差",
        index="季节",
        columns="负荷段",
        aggfunc=lambda x: ((x**2).mean()) ** 0.5 * 100,
        observed=True,
    ).reindex(index=["夏季工况", "冬季工况"], columns=LOAD_LABELS)
    gt1_rrmse_table.loc["全部工况"] = [
        relative_rmse_pct(gt1.loc[gt1["负荷段"] == label]) for label in LOAD_LABELS
    ]
    gt1_count_table = gt1.pivot_table(
        values="相对误差",
        index="季节",
        columns="负荷段",
        aggfunc="count",
        observed=True,
    ).reindex(index=["夏季工况", "冬季工况"], columns=LOAD_LABELS)
    gt1_count_table.loc["全部工况"] = [int((gt1["负荷段"] == label).sum()) for label in LOAD_LABELS]
    print_section("GT1 PPT表格建议：相对均方根误差 / %", gt1_rrmse_table.round(2))
    print_section("GT1 各格样本数", gt1_count_table.fillna(0).astype(int))

    columns = ["燃机", "原始序号", "环境温度_C", "实际功率_MW", "修正计算功率_MW", "残差_MW", "绝对残差_MW"]
    top_errors = data.sort_values("绝对残差_MW", ascending=False).head(20)[columns]
    print_section("绝对残差最大的20个点", top_errors)

    over_count = (data["残差_MW"] > 0).sum()
    under_count = (data["残差_MW"] < 0).sum()
    print(f"\n残差方向：计算值高于实测 {over_count} 个点，计算值低于实测 {under_count} 个点。")

    configs = [
        ("GT1", data["燃机"] == "GT1", "GT1 实际功率 / MW", "GT1 修正计算功率 / MW"),
        ("GT2", data["燃机"] == "GT2", "GT2 实际功率 / MW", "GT2 修正计算功率 / MW"),
    ]
    vmin = data["环境温度_C"].quantile(0.01)
    vmax = data["环境温度_C"].quantile(0.99)

    for unit, mask, xlab, ylab in configs:
        unit_data = data.loc[mask]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sc = ax.scatter(
            unit_data["实际功率_MW"],
            unit_data["修正计算功率_MW"],
            c=unit_data["环境温度_C"],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=4,
            alpha=0.45,
            edgecolors="none",
        )
        ax.plot([0, UPPER_MW], [0, UPPER_MW], "r--", linewidth=1.2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(100, 300)
        ax.set_ylim(100, 300)
        ax.set_title(f"{unit} 燃机出力对比")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label("环境温度 / ℃")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
