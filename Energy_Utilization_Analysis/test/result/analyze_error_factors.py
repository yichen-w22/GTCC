from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
GT1_PATH = CURRENT_DIR / "gt1_results_wide.csv"
GT2_PATH = CURRENT_DIR / "gt2_results_wide.csv"


def build_error_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["net_power"] - df["actual_power"]
    df["abs_error"] = df["error"].abs()
    df["relative_error"] = df["error"] / df["actual_power"]
    fuel_energy = df["fuel_m_dot"] * df["chamber_fuel_lhv"]
    df["overall_efficiency"] = df["net_power"] / fuel_energy
    return df


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    target_cols = [
        "actual_power",
        "net_power",
        "error",
        "abs_error",
        "relative_error",
        "compressor_efficiency",
        "turbine_efficiency",
        "overall_efficiency",
        "compressor_power",
        "turbine_power",
        "fuel_m_dot",
        "state_1_T",
        "state_2_T",
        "state_3_T",
        "state_4_T",
        "state_2_P",
        "state_3_P",
        "state_4_P",
    ]
    available_cols = [col for col in target_cols if col in df.columns]
    return df[available_cols].corr(numeric_only=True)


def plot_correlation_matrix(corr: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title(title)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def print_error_summary(df: pd.DataFrame, label: str) -> None:
    rmse_mw = ((df["error"] ** 2).mean() ** 0.5) / 1e6
    print("=" * 80)
    print(label)
    print(f"sample_count = {len(df)}")
    print(f"mean_actual_power_MW = {df['actual_power'].mean() / 1e6:.6f}")
    print(f"mean_net_power_MW = {df['net_power'].mean() / 1e6:.6f}")
    print(f"mean_error_MW = {df['error'].mean() / 1e6:.6f}")
    print(f"mae_MW = {df['abs_error'].mean() / 1e6:.6f}")
    print(f"rmse_MW = {rmse_mw:.6f}")


def process_unit(label: str, csv_path: Path, output_name: str) -> None:
    df = pd.read_csv(csv_path)
    df = build_error_table(df)
    corr = build_correlation_matrix(df)
    plot_correlation_matrix(
        corr=corr,
        title=f"{label} correlation matrix",
        output_path=CURRENT_DIR / output_name,
    )
    print_error_summary(df, label)


def main() -> None:
    process_unit("GT1", GT1_PATH, "gt1_correlation_matrix.png")
    process_unit("GT2", GT2_PATH, "gt2_correlation_matrix.png")
    print(f"Saved correlation matrix plots to {CURRENT_DIR}")


if __name__ == "__main__":
    main()
