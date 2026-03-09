import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import product

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================
# 0. 基础参数
# =========================================================
dt = 60.0  # s

BASE_PARAMS = {
    "cp_g": 1200.0,   # J/(kg·K)
    "cp_s": 2750.0,   # J/(kg·K)
    "a_s": 0.8,
    "K_s": 0.9e4,
    "a_g": 0.7,
    "K_g": 1.5e4,
    "C_m": 1.1e8,
    "T_m0": 785.0,
}


# =========================================================
# 1. 单步预测
# =========================================================
def one_step(Tg_in, mg, Ts_in, ms, Tm, params):
    mg = max(float(mg), 1e-8)
    ms = max(float(ms), 1e-8)

    cp_g = params["cp_g"]
    cp_s = params["cp_s"]
    a_s = params["a_s"]
    K_s = params["K_s"]
    a_g = params["a_g"]
    K_g = params["K_g"]
    C_m = params["C_m"]

    UA_g = K_g * mg ** a_g
    UA_s = K_s * ms ** a_s

    UA_tot = UA_g + UA_s
    tau = C_m / max(UA_tot, 1e-8)
    T_eq = (UA_g * Tg_in + UA_s * Ts_in) / max(UA_tot, 1e-8)

    exp_term = np.exp(-dt / tau)
    Tm_next = T_eq + (Tm - T_eq) * exp_term
    Tm_avg = T_eq + (Tm - T_eq) * (tau / dt) * (1 - exp_term)

    Tg_out = Tg_in - UA_g / (mg * cp_g) * (Tg_in - Tm_avg)
    Ts_out = Ts_in + UA_s / (ms * cp_s) * (Tm_avg - Ts_in)

    Q_gm = UA_g * (Tg_in - Tm_avg)
    Q_ms = UA_s * (Tm_avg - Ts_in)

    return Tg_out, Ts_out, Tm_next, UA_g, UA_s, Q_gm, Q_ms


# =========================================================
# 2. 对单个片段做多步预测
# =========================================================
def predict_on_segment(seg_df: pd.DataFrame, params, verbose=False):
    seg_df = seg_df.reset_index(drop=True).copy()

    Tm = params["T_m0"]
    results = []

    for k in range(len(seg_df) - 1):
        row_in = seg_df.iloc[k]
        row_true = seg_df.iloc[k + 1]

        Tg_in = row_in["高压过热器1入口烟温"]
        mg = row_in["烟气流量"]
        Ts_in = row_in["高压过热器1入口蒸汽温度"]
        ms = row_in["高压主蒸汽流量"]

        Tg_out_pred, Ts_out_pred, Tm_next, UA_g, UA_s, Q_gm, Q_ms = one_step(
            Tg_in, mg, Ts_in, ms, Tm, params
        )

        Tg_out_true = row_true["高压过热器1出口烟温"]
        Ts_out_true = row_true["高压过热器1出口蒸汽温度"]

        result_row = {
            "step": k + 1,
            "Tg_in": Tg_in,
            "mg": mg,
            "Ts_in": Ts_in,
            "ms": ms,
            "Tm_used": Tm,
            "UA_g": UA_g,
            "UA_s": UA_s,
            "Q_gm": Q_gm,
            "Q_ms": Q_ms,
            "Tg_out_pred": Tg_out_pred,
            "Ts_out_pred": Ts_out_pred,
            "Tm_next": Tm_next,
            "Tg_out_true": Tg_out_true,
            "Ts_out_true": Ts_out_true,
            "Tg_err": Tg_out_pred - Tg_out_true,
            "Ts_err": Ts_out_pred - Ts_out_true,
        }
        results.append(result_row)

        # if verbose and (k + 1) % 200 == 0:
        #     print(f"已完成 {k+1}/{len(seg_df)-1} 步")

        Tm = Tm_next

    return pd.DataFrame(results)


# =========================================================
# 3. 误差指标
# =========================================================
def calc_metrics(pred_df: pd.DataFrame):
    rmse_tg = np.sqrt(np.mean((pred_df["Tg_out_pred"] - pred_df["Tg_out_true"]) ** 2))
    rmse_ts = np.sqrt(np.mean((pred_df["Ts_out_pred"] - pred_df["Ts_out_true"]) ** 2))

    mae_tg = np.mean(np.abs(pred_df["Tg_out_pred"] - pred_df["Tg_out_true"]))
    mae_ts = np.mean(np.abs(pred_df["Ts_out_pred"] - pred_df["Ts_out_true"]))

    # 目标函数：两个出口温度 RMSE 之和
    objective = rmse_tg + rmse_ts

    return {
        "rmse_tg": rmse_tg,
        "rmse_ts": rmse_ts,
        "mae_tg": mae_tg,
        "mae_ts": mae_ts,
        "objective": objective,
    }


# =========================================================
# 4. 网格搜索
# =========================================================
def grid_search(seg_df: pd.DataFrame, base_params: dict, search_space: dict):
    keys = list(search_space.keys())
    values_list = [search_space[k] for k in keys]
    total = np.prod([len(v) for v in values_list])

    print(f"开始网格搜索，总组合数: {total}")

    best_params = None
    best_metrics = None
    best_pred_df = None

    records = []
    count = 0

    for values in product(*values_list):
        count += 1
        params = base_params.copy()
        for k, v in zip(keys, values):
            params[k] = v

        pred_df = predict_on_segment(seg_df, params, verbose=False)
        metrics = calc_metrics(pred_df)

        record = params.copy()
        record.update(metrics)
        records.append(record)

        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_params = params.copy()
            best_metrics = metrics.copy()
            best_pred_df = pred_df.copy()

            print(
                f"[{count}/{total}] 当前最优: "
                f"objective={best_metrics['objective']:.4f}, "
                f"rmse_ts={best_metrics['rmse_ts']:.4f}, "
                f"rmse_tg={best_metrics['rmse_tg']:.4f}"
            )
            print(best_params)

        if count % 20 == 0 or count == total:
            print(f"搜索进度: {count}/{total}")

    search_df = pd.DataFrame(records)
    search_df = search_df.sort_values("objective").reset_index(drop=True)

    return best_params, best_metrics, best_pred_df, search_df


# =========================================================
# 5. 绘图
# =========================================================
def plot_results(pred_df: pd.DataFrame):
    fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    axes[0].plot(pred_df["step"], pred_df["Ts_out_true"], label="蒸汽出口真实值")
    axes[0].plot(pred_df["step"], pred_df["Ts_out_pred"], label="蒸汽出口预测值")
    axes[0].set_ylabel("温度")
    axes[0].set_title("蒸汽出口温度预测")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pred_df["step"], pred_df["Tg_out_true"], label="烟气出口真实值")
    axes[1].plot(pred_df["step"], pred_df["Tg_out_pred"], label="烟气出口预测值")
    axes[1].set_ylabel("温度")
    axes[1].set_title("烟气出口温度预测")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pred_df["step"], pred_df["Tm_used"], label="金属温度")
    axes[2].set_ylabel("温度")
    axes[2].set_title("金属温度演化")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(pred_df["step"], pred_df["mg"], label="烟气流量")
    axes[3].set_ylabel("流量")
    axes[3].set_title("烟气流量")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(pred_df["step"], pred_df["ms"], label="蒸汽流量")
    axes[4].set_ylabel("流量")
    axes[4].set_title("蒸汽流量")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv(r"HRSG\jyrd高压余热锅炉过热器1_滑窗平均数据1.csv")

    cols_needed = [
        "高压过热器1入口烟温",
        "高压过热器1出口烟温",
        "高压过热器1入口蒸汽温度",
        "高压过热器1出口蒸汽温度",
        "烟气流量",
        "高压主蒸汽流量",
    ]

    df = df[cols_needed].dropna().reset_index(drop=True)

    # -----------------------------
    # 搜索段：先别用太长，否则很慢
    # -----------------------------
    seg_df_search = df.iloc[:1500].copy()

    # -----------------------------
    # 网格搜索空间
    # 不建议一开始把每个参数都放太多取值
    # -----------------------------
    search_space = {
        "a_s": [0.8],
        "K_s": [0.8e4, 0.9e4, 1.0e4],
        "a_g": [0.6, 0.7, 0.8],
        "K_g": [1.4e4, 1.5e4, 1.6e4],
        "C_m": [0.7e8, 0.8e8, 0.9e8],
        "T_m0": [750.0, 760.0, 770.0],
        # 如果你也想搜 cp，可以放开
        "cp_g": [1000.0, 1100.0, 1200.0, 1300.0],
        "cp_s": [2300.0, 2600.0, 2900.0],
    }

    best_params, best_metrics, best_pred_df, search_df = grid_search(
        seg_df_search,
        BASE_PARAMS,
        search_space,
    )

    print("\n最优参数：")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    print("\n最优指标：")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.6f}")

    search_df.to_csv(
        r"HRSG\superheater_grid_search_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------
    # 用最优参数在更长片段上重新预测
    # -----------------------------
    seg_df_full = df.iloc[:5000].copy()
    pred_df = predict_on_segment(seg_df_full, best_params, verbose=True)

    metrics_full = calc_metrics(pred_df)
    print("\n全片段预测指标：")
    for k, v in metrics_full.items():
        print(f"{k}: {v:.6f}")

    pred_df.to_csv(
        r"HRSG\superheater_prediction_results_best.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_results(pred_df)