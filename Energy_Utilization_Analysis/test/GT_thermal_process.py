from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent / "result"
GT_RESULT_FILES = {
    1: RESULT_DIR / "gt1_results_wide.csv",
    2: RESULT_DIR / "gt2_results_wide.csv",
}
STATE_SEQUENCE = ["state_1", "state_2", "state_3", "state_4", "state_1"]


def load_gt_result(unit: int) -> pd.DataFrame:
    if unit not in GT_RESULT_FILES:
        raise ValueError("unit must be 1 or 2")
    return pd.read_csv(GT_RESULT_FILES[unit])


def _build_ts_cycle(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for state_name in STATE_SEQUENCE:
        records.append(
            {
                "state": state_name,
                "s": df[f"{state_name}_s"],
                "T": df[f"{state_name}_T"],
            }
        )

    cycle_df = pd.concat(
        [pd.DataFrame(record) for record in records],
        ignore_index=True,
    )
    cycle_df["s_kj"] = cycle_df["s"] / 1000.0
    mean_cycle = (
        cycle_df.groupby("state", sort=False)[["s_kj", "T"]]
        .mean()
        .reset_index()
    )
    return cycle_df, mean_cycle


def plot_gt_ts_diagram(unit: int, alpha: float = 0.06, linewidth: float = 0.8) -> None:
    df = load_gt_result(unit)
    cycle_df, mean_cycle = _build_ts_cycle(df)

    plt.figure(figsize=(10, 7))

    for _, row in df.iterrows():
        s_values = [row[f"{state_name}_s"] / 1000.0 for state_name in STATE_SEQUENCE]
        t_values = [row[f"{state_name}_T"] for state_name in STATE_SEQUENCE]
        plt.plot(s_values, t_values, color="#4c78a8", alpha=alpha, linewidth=linewidth)

    plt.plot(
        mean_cycle["s_kj"],
        mean_cycle["T"],
        color="#d62728",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label=f"GT{unit} mean cycle",
    )

    for _, row in mean_cycle.iterrows():
        plt.annotate(
            row["state"],
            (row["s_kj"], row["T"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    plt.title(f"GT{unit} T-s Diagram")
    plt.xlabel("Entropy s (kJ/kg-K)")
    plt.ylabel("Temperature T (K)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_gt_ts_diagrams() -> None:
    for unit in (1, 2):
        plot_gt_ts_diagram(unit)


if __name__ == "__main__":
    plot_all_gt_ts_diagrams()
