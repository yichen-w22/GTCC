from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(
    r"C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\energy_analysis\GT_model\result\gt1_results_wide.csv"
)
TARGET_COLUMN = "mass_correction_ratio"
FEATURE_COLUMNS = [
    "state_1_m_dot",
    "fuel_m_dot",
]
ROW_SLICE = slice(100, 4000)


def mass_correction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["power_residual"] = df["net_power"] - df["actual_power"]
    df["mass_residual"] = df["power_residual"] / (df["state_4_h"] - df["state_1_h"])
    df["mass_correction_ratio"] = df["mass_residual"] / df["state_1_m_dot"]
    df["corrected_mass"] = df["state_1_m_dot"] + df["mass_residual"]
    df["air_fuel_ratio"] = df["state_1_m_dot"] / df["fuel_m_dot"]
    return df


def load_regression_data(
    data_path: Path,
    feature_columns: list[str],
    target_column: str,
    row_slice: slice | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    if row_slice is not None:
        df = df.iloc[row_slice].copy()
    df = mass_correction(df)

    selected_columns = feature_columns + [target_column]
    clean_df = df[selected_columns].dropna().copy()

    x = clean_df[feature_columns].to_numpy(dtype=float)
    y = clean_df[target_column].to_numpy(dtype=float)
    return clean_df, x, y


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    design_matrix = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    intercept = coefficients[0]
    weights = coefficients[1:]
    return intercept, weights


def predict_linear_regression(x: np.ndarray, intercept: float, weights: np.ndarray) -> np.ndarray:
    return intercept + x @ weights


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def calc_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def build_expression(target_column: str, feature_columns: list[str], intercept: float, weights: np.ndarray) -> str:
    terms = [f"{intercept:.6g}"]
    for feature_name, weight in zip(feature_columns, weights):
        sign = "+" if weight >= 0 else "-"
        terms.append(f" {sign} {abs(weight):.6g} * {feature_name}")
    return f"{target_column} = " + "".join(terms)


def plot_regression_result(y_true: np.ndarray, y_pred: np.ndarray, target_column: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolor="none")
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    axes[0].plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", label="ideal")
    axes[0].set(
        title=f"{target_column} Regression",
        xlabel="Actual",
        ylabel="Predicted",
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_true, label="actual", linewidth=1.6)
    axes[1].plot(y_pred, label="predicted", linewidth=1.6)
    axes[1].set(
        title=f"{target_column} Sequence Comparison",
        xlabel="Sample Index",
        ylabel=target_column,
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def main() -> None:
    _, x, y = load_regression_data(
        data_path=DATA_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        row_slice=ROW_SLICE,
    )

    intercept, weights = fit_linear_regression(x, y)
    y_pred = predict_linear_regression(x, intercept, weights)

    expression = build_expression(TARGET_COLUMN, FEATURE_COLUMNS, intercept, weights)

    print("Linear regression expression:")
    print(expression)
    print()
    print(f"samples = {len(x)}")
    print(f"r2 = {calc_r2(y, y_pred):.6f}")
    print(f"mse = {calc_mse(y, y_pred):.6f}")

    plot_regression_result(y, y_pred, TARGET_COLUMN)


if __name__ == "__main__":
    main()
