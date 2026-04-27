from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
GT1_PATH = CURRENT_DIR / "gt1_results_wide.csv"
GT2_PATH = CURRENT_DIR / "gt2_results_wide.csv"


def add_overall_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fuel_energy = df["fuel_m_dot"] * df["chamber_fuel_lhv"]
    df["overall_efficiency"] = df["net_power"] / fuel_energy
    return df


def plot_efficiency_scatter(df: pd.DataFrame, unit: int, column: str, ylabel: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["source_idx"], df[column], s=18, alpha=0.75)
    ax.set_xlabel("source_idx")
    ax.set_ylabel(ylabel)
    ax.set_title(f"GT{unit} {ylabel} scatter")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / output_name, dpi=150)
    plt.close(fig)


def plot_power_comparison(df: pd.DataFrame, unit: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df["actual_power"] / 1e6
    y = df["net_power"] / 1e6
    ax.scatter(x, y, s=18, alpha=0.75)
    line_min = min(x.min(), y.min())
    line_max = max(x.max(), y.max())
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual power / MW")
    ax.set_ylabel("Calculated power / MW")
    ax.set_title(f"GT{unit} actual vs calculated power")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / f"gt{unit}_actual_vs_calculated_power.png", dpi=150)
    plt.close(fig)


def process_unit(unit: int, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = add_overall_efficiency(df)

    plot_efficiency_scatter(
        df=df,
        unit=unit,
        column="compressor_efficiency",
        ylabel="compressor efficiency",
        output_name=f"gt{unit}_compressor_efficiency_scatter.png",
    )
    plot_efficiency_scatter(
        df=df,
        unit=unit,
        column="turbine_efficiency",
        ylabel="turbine efficiency",
        output_name=f"gt{unit}_turbine_efficiency_scatter.png",
    )
    plot_efficiency_scatter(
        df=df,
        unit=unit,
        column="overall_efficiency",
        ylabel="overall efficiency",
        output_name=f"gt{unit}_overall_efficiency_scatter.png",
    )
    plot_power_comparison(df=df, unit=unit)


def main() -> None:
    process_unit(1, GT1_PATH)
    process_unit(2, GT2_PATH)
    print(f"Saved plots to {CURRENT_DIR}")


if __name__ == "__main__":
    main()
