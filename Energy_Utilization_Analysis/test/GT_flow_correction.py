import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.GT_model.GT_model import GTModel
from energy_analysis.working_fluid.gas import (
    GasComposition,
    GasState,
    build_air_composition,
    build_flue_gas_composition,
    create_gas_reference_env,
)


DATA_PATH = PROJECT_ROOT / "data_precessing" / "continuous_data_10min.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_precessing"

N_FIT = 400
N_VALIDATION = 100


def unit_columns(unit: int) -> dict[str, str]:
    return {
        "air_flow": f"燃机进口空气流量_{unit}",
        "fuel_flow": f"燃料质量流量_{unit}",
        "actual_power": f"燃机出力_{unit}",
        "fuel_T": f"燃料温度_{unit}",
        "fuel_P": f"燃料压力_{unit}",
        "ambient_T": f"环境温度_{unit}",
        "ambient_P": f"压气机入口压力_{unit}",
        "ambient_RH": f"大气相对湿度_{unit}",
        "compressor_out_T": f"压气机出口温度_{unit}",
        "compressor_out_P": f"压气机出口压力_{unit}",
        "flue_T": f"进口烟温_{unit}",
        "flue_P": f"进口烟压_{unit}",
    }


def fuel_composition_from_row(row: pd.Series) -> GasComposition:
    return GasComposition.from_dict(
        {
            "H2": row["H2"],
            "N2": row["N2"],
            "CO2": row["CO2"],
            "CH4": row["CH4"],
            "CO": row["CO"],
            "O2+Ar": row["O2+Ar"],
            "C2H6": row["C2H6"],
            "C3H8": row["C3H8"],
            "iC4H10": row["iC4H10"],
            "nC4H10": row["nC4H10"],
            "iC5H12": row["iC5H12"],
            "nC5H12": row["nC5H12"],
        }
    )


def reference_env_from_row(row: pd.Series):
    return create_gas_reference_env(
        T0=(row["环境温度_1"] + row["环境温度_2"]) / 2,
        P0=(row["压气机入口压力_1"] + row["压气机入口压力_2"]) / 2,
    )


def solve_gt_model(row: pd.Series, unit: int, air_flow: float) -> float:
    cols = unit_columns(unit)
    fuel_flow = row[cols["fuel_flow"]]

    gas_ref = reference_env_from_row(row)
    air_composition = build_air_composition(
        T=row[cols["ambient_T"]],
        P=row[cols["ambient_P"]],
        RH=row[cols["ambient_RH"]],
    )
    fuel_composition = fuel_composition_from_row(row)
    flue_gas_composition = build_flue_gas_composition(
        fuel_composition=fuel_composition,
        air_composition=air_composition,
        m_dot_fuel=fuel_flow,
        m_dot_air=air_flow,
    )

    fuel = GasState.from_TP(
        T=row[cols["fuel_T"]],
        P=row[cols["fuel_P"]],
        m_dot=fuel_flow,
        composition=fuel_composition,
        name=f"{unit}号炉燃料",
        ref=gas_ref,
    )
    state_1 = GasState.from_TP(
        T=row[cols["ambient_T"]],
        P=row[cols["ambient_P"]],
        m_dot=air_flow,
        composition=air_composition,
        name=f"{unit}号燃机入口空气",
        ref=gas_ref,
    )
    state_2 = GasState.from_TP(
        T=row[cols["compressor_out_T"]],
        P=row[cols["compressor_out_P"]],
        m_dot=air_flow,
        composition=air_composition,
        name=f"{unit}号燃机压气机出口",
        ref=gas_ref,
    )
    state_4 = GasState.from_TP(
        T=row[cols["flue_T"]],
        P=row[cols["flue_P"]],
        m_dot=fuel_flow + air_flow,
        composition=flue_gas_composition,
        name=f"{unit}号余热锅炉入口烟气",
        ref=gas_ref,
    )

    return GTModel().solve(state_1, state_2, fuel, state_4).net_power


def calc_uncorrected_sample(df: pd.DataFrame, idx: int, unit: int) -> dict[str, float]:
    row = df.iloc[idx].copy()
    cols = unit_columns(unit)

    original_air_flow = row[cols["air_flow"]]
    fuel_flow = row[cols["fuel_flow"]]
    actual_power = row[cols["actual_power"]]

    gas_ref = reference_env_from_row(row)
    air_composition = build_air_composition(
        T=row[cols["ambient_T"]],
        P=row[cols["ambient_P"]],
        RH=row[cols["ambient_RH"]],
    )
    fuel_composition = fuel_composition_from_row(row)
    flue_gas_composition = build_flue_gas_composition(
        fuel_composition=fuel_composition,
        air_composition=air_composition,
        m_dot_fuel=fuel_flow,
        m_dot_air=original_air_flow,
    )

    fuel = GasState.from_TP(
        T=row[cols["fuel_T"]],
        P=row[cols["fuel_P"]],
        m_dot=fuel_flow,
        composition=fuel_composition,
        name=f"{unit}号炉燃料",
        ref=gas_ref,
    )
    state_1 = GasState.from_TP(
        T=row[cols["ambient_T"]],
        P=row[cols["ambient_P"]],
        m_dot=original_air_flow,
        composition=air_composition,
        name=f"{unit}号燃机入口空气",
        ref=gas_ref,
    )
    state_2 = GasState.from_TP(
        T=row[cols["compressor_out_T"]],
        P=row[cols["compressor_out_P"]],
        m_dot=original_air_flow,
        composition=air_composition,
        name=f"{unit}号燃机压气机出口",
        ref=gas_ref,
    )
    state_4 = GasState.from_TP(
        T=row[cols["flue_T"]],
        P=row[cols["flue_P"]],
        m_dot=fuel_flow + original_air_flow,
        composition=flue_gas_composition,
        name=f"{unit}号余热锅炉入口烟气",
        ref=gas_ref,
    )

    result = GTModel().solve(state_1, state_2, fuel, state_4)
    original_power = result.net_power
    power_error = original_power - actual_power

    delta_h = result.state_4.h - result.state_1.h
    air_error = power_error / delta_h
    corrected_air_flow = original_air_flow + air_error
    corr_coff = corrected_air_flow / original_air_flow

    return {
        "unit": unit,
        "idx": idx,
        "air_flow": original_air_flow,
        "fuel_flow": fuel_flow,
        "air_fuel_ratio": original_air_flow / fuel_flow,
        "actual_power": actual_power,
        "original_power": original_power,
        "original_power_error": power_error,
        "original_power_error_ratio": power_error / actual_power,
        "target_corrected_air_flow": corrected_air_flow,
        "target_air_delta": air_error,
        "corr_coff": corr_coff,
    }


def build_uncorrected_samples(
    df: pd.DataFrame,
    indices: np.ndarray,
    unit: int,
    label: str,
) -> pd.DataFrame:
    records = []
    for sample_no, idx in enumerate(indices, start=1):
        try:
            records.append(calc_uncorrected_sample(df, int(idx), unit))
        except Exception as exc:
            print(f"{unit}号燃机第 {idx} 行修正前样本计算失败: {exc}")

        if sample_no % 10 == 0:
            print(
                f"{unit}号燃机 {label} 修正前样本已处理 "
                f"{sample_no}/{len(indices)} 个，当前 idx={idx}，有效样本={len(records)}"
            )

    sample_df = pd.DataFrame(records).dropna()
    if sample_df.empty:
        return sample_df

    return sample_df[
        (sample_df["air_flow"] > 0)
        & (sample_df["fuel_flow"] > 0)
        & np.isfinite(sample_df["corr_coff"])
        & (sample_df["corr_coff"] > 0)
    ].copy()


def fit_corr_coff(sample_df: pd.DataFrame):
    features = ["fuel_flow", "air_flow", "air_fuel_ratio"]
    X = sample_df[features]
    y = sample_df["corr_coff"]

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    return {
        "model": model,
        "features": features,
        "intercept": model.intercept_,
        "coef": model.coef_,
        "r2": r2_score(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "mae": mean_absolute_error(y, pred),
    }


def predict_corr_coff(row: pd.Series, unit: int, fit_result: dict) -> float:
    cols = unit_columns(unit)
    air_flow = row[cols["air_flow"]]
    fuel_flow = row[cols["fuel_flow"]]
    X = pd.DataFrame(
        {
            "fuel_flow": [fuel_flow],
            "air_flow": [air_flow],
            "air_fuel_ratio": [air_flow / fuel_flow],
        }
    )
    return fit_result["model"].predict(X)[0]


def validate_model(df: pd.DataFrame, sample_df: pd.DataFrame, unit: int, fit_result: dict) -> pd.DataFrame:
    records = []
    cols = unit_columns(unit)

    for _, sample in sample_df.iterrows():
        idx = int(sample["idx"])
        row = df.iloc[idx].copy()

        try:
            air_flow = row[cols["air_flow"]]
            actual_power = row[cols["actual_power"]]
            corr_coff_pred = predict_corr_coff(row, unit, fit_result)
            corrected_air_flow = air_flow * corr_coff_pred
            corrected_power = solve_gt_model(row, unit, corrected_air_flow)

            records.append(
                {
                    **sample.to_dict(),
                    "corr_coff_pred": corr_coff_pred,
                    "corrected_air_flow": corrected_air_flow,
                    "corrected_power": corrected_power,
                    "corrected_power_error": corrected_power - actual_power,
                    "corrected_power_error_ratio": (corrected_power - actual_power) / actual_power,
                }
            )
        except Exception as exc:
            print(f"{unit}号燃机第 {idx} 行验证计算失败: {exc}")

    return pd.DataFrame(records)


def print_power_metrics(title: str, result_df: pd.DataFrame) -> None:
    print(title)
    print(f"样本数 = {len(result_df)}")
    print(
        f"修正前 MAE  = "
        f"{mean_absolute_error(result_df['actual_power'], result_df['original_power']):.6f}"
    )
    print(
        f"修正前 RMSE = "
        f"{mean_squared_error(result_df['actual_power'], result_df['original_power']) ** 0.5:.6f}"
    )
    print(
        f"修正后 MAE  = "
        f"{mean_absolute_error(result_df['actual_power'], result_df['corrected_power']):.6f}"
    )
    print(
        f"修正后 RMSE = "
        f"{mean_squared_error(result_df['actual_power'], result_df['corrected_power']) ** 0.5:.6f}"
    )


def run_unit(df: pd.DataFrame, unit: int, fit_indices: np.ndarray, validation_indices: np.ndarray) -> None:
    print()
    print(f"========== {unit}号燃机 ==========")

    fit_samples = build_uncorrected_samples(df, fit_indices, unit, "训练集")
    validation_samples = build_uncorrected_samples(df, validation_indices, unit, "验证集")

    fit_sample_path = OUTPUT_DIR / f"gt{unit}_uncorrected_fit_samples.csv"
    validation_sample_path = OUTPUT_DIR / f"gt{unit}_uncorrected_validation_samples.csv"
    fit_samples.to_csv(fit_sample_path, index=False, encoding="utf-8-sig")
    validation_samples.to_csv(validation_sample_path, index=False, encoding="utf-8-sig")

    print(f"修正前训练样本已保存: {fit_sample_path}")
    print(f"修正前验证样本已保存: {validation_sample_path}")

    fit_result = fit_corr_coff(fit_samples)
    intercept = fit_result["intercept"]
    b, c, d = fit_result["coef"]

    print()
    print("修正系数拟合表达式:")
    print(
        f"corr_coff = {intercept:.10f} "
        f"+ ({b:.10e}) * fuel_flow "
        f"+ ({c:.10e}) * air_flow "
        f"+ ({d:.10e}) * air_flow/fuel_flow"
    )
    print(f"R2   = {fit_result['r2']:.6f}")
    print(f"RMSE = {fit_result['rmse']:.6f}")
    print(f"MAE  = {fit_result['mae']:.6f}")

    fit_result_df = validate_model(df, fit_samples, unit, fit_result)
    validation_result_df = validate_model(df, validation_samples, unit, fit_result)

    fit_result_path = OUTPUT_DIR / f"gt{unit}_correction_fit_result.csv"
    validation_result_path = OUTPUT_DIR / f"gt{unit}_correction_validation_result.csv"
    fit_result_df.to_csv(fit_result_path, index=False, encoding="utf-8-sig")
    validation_result_df.to_csv(validation_result_path, index=False, encoding="utf-8-sig")

    print()
    print_power_metrics("训练集功率误差:", fit_result_df)
    print()
    print_power_metrics("验证集功率误差:", validation_result_df)
    print()
    print(f"训练结果已保存: {fit_result_path}")
    print(f"验证结果已保存: {validation_result_path}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    all_indices = np.linspace(0, len(df) - 1, N_FIT + N_VALIDATION, dtype=int)
    fit_indices = all_indices[::2]
    validation_indices = all_indices[1::2]

    for unit in (1, 2):
        run_unit(df, unit, fit_indices, validation_indices)
