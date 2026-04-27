import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from CoolProp.CoolProp import PropsSI
from iapws import IAPWS97

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================
# 0. 全局常数
# =========================================================
dt = 60.0       # s

a_s = 0.8       # 蒸汽侧指数固定
K_s = 2.0e4     # 蒸汽侧系数

a_g = 0.7       # 烟气侧指数固定
K_g = 3.7e4     # 烟气侧系数
C_m = 0.3e8     # 金属热容
T_m0 = 749.0    # 金属初始温度

P_ATM = 101325.0   # Pa，大气压


# =========================================================
# 1. 物性函数
# =========================================================
def cp_flue_gas(T, P=P_ATM):
    """
    烟气定压比热
    T : K
    P : Pa
    return : J/(kg·K)

    用质量分数近似加权：
    N2 0.75, O2 0.12, CO2 0.08, H2O 0.05
    """
    comp = {
        "Nitrogen": 0.75,
        "Oxygen": 0.12,
        "CarbonDioxide": 0.08,
        "Water": 0.05
    }

    cp = 0.0
    for gas, x in comp.items():
        cp_i = PropsSI("Cpmass", "T", T, "P", P, gas)
        cp += x * cp_i
    return cp


def cp_steam_iapws(T, P):
    """
    水蒸气定压比热
    T : K
    P_MPa : MPa
    return : J/(kg·K)

    iapws.IAPWS97 的 cp 返回单位是 kJ/(kg·K)，这里转成 J/(kg·K)
    """
    state = IAPWS97(P=P / 1e6, T=T)
    return state.cp * 1000.0


# =========================================================
# 2. 单步预测（带平均温度 + 动态 cp）
# =========================================================
def one_step(Tg_in, mg, Ts_in, ms, P_drum, Tm, n_iter=3):
    """
    Tg_in : ℃
    mg    : kg/s
    Ts_in : ℃
    ms    : kg/s
    P_drum_MPa : MPa
    Tm    : ℃
    """

    mg = max(float(mg), 1e-8)
    ms = max(float(ms), 1e-8)

    UA_g = K_g * mg ** a_g
    UA_s = K_s * ms ** a_s

    # 初值：先用入口温度近似平均温度
    Tg_out = float(Tg_in)
    Ts_out = float(Ts_in)

    # 小迭代：因为 cp 依赖平均温度，而平均温度依赖出口温度
    for _ in range(n_iter):
        Tg_avg_C = 0.5 * (Tg_in + Tg_out)
        Ts_avg_C = 0.5 * (Ts_in + Ts_out)

        cp_g = cp_flue_gas(T=Tg_avg_C + 273.15, P=P_ATM)
        cp_s = cp_steam_iapws(T=Ts_avg_C + 273.15, P=P_drum)

        # 平均温度形式下显式出口温度
        Tg_out = (
            (mg * cp_g - UA_g / 2.0) * Tg_in + UA_g * Tm
        ) / (mg * cp_g + UA_g / 2.0)

        Ts_out = (
            (ms * cp_s - UA_s / 2.0) * Ts_in + UA_s * Tm
        ) / (ms * cp_s + UA_s / 2.0)

    # 收敛后再算一次平均温度与换热量
    Tg_avg_C = 0.5 * (Tg_in + Tg_out)
    Ts_avg_C = 0.5 * (Ts_in + Ts_out)

    cp_g = cp_flue_gas(T=Tg_avg_C + 273.15, P=P_ATM)
    cp_s = cp_steam_iapws(T=Ts_avg_C + 273.15, P=P_drum)

    Q_gm = UA_g * (Tg_avg_C - Tm)
    Q_ms = UA_s * (Tm - Ts_avg_C)

    # 金属节点解析离散更新
    UA_tot = UA_g + UA_s
    T_eq = (UA_g * Tg_avg_C + UA_s * Ts_avg_C) / UA_tot
    tau = C_m / UA_tot

    Tm_next = T_eq + (Tm - T_eq) * np.exp(-dt / tau)

    return Tg_out, Ts_out, Tm_next, UA_g, UA_s, Q_gm, Q_ms, cp_g, cp_s


# =========================================================
# 3. 对单个片段做多步预测
# =========================================================
def predict_on_segment(seg_df: pd.DataFrame, verbose=True):
    seg_df = seg_df.reset_index(drop=True).copy()

    Tm = T_m0
    results = []

    for k in range(len(seg_df) - 1):
        row_in = seg_df.iloc[k]
        row_true = seg_df.iloc[k + 1]

        Tg_in = row_in["高压过热器1入口烟温"]
        mg = row_in["烟气流量"]
        Ts_in = row_in["高压过热器1入口蒸汽温度"]
        ms = row_in["高压主蒸汽流量"]
        P_drum = row_in["汽包压力"]   # MPa

        Tg_out_pred, Ts_out_pred, Tm_next, UA_g, UA_s, Q_gm, Q_ms, cp_g, cp_s = one_step(
            Tg_in, mg, Ts_in, ms, P_drum, Tm
        )

        Tg_out_true = row_true["高压过热器1出口烟温"]
        Ts_out_true = row_true["高压过热器1出口蒸汽温度"]

        result_row = {
            "step": k + 1,
            "Tg_in": Tg_in,
            "mg": mg,
            "Ts_in": Ts_in,
            "ms": ms,
            "P_drum": P_drum,
            "Tm_used": Tm,
            "UA_g": UA_g,
            "UA_s": UA_s,
            "cp_g": cp_g,
            "cp_s": cp_s,
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

        Tm = Tm_next

    pred_df = pd.DataFrame(results)
    return pred_df


# =========================================================
# 4. 绘图
# =========================================================
def plot_results(pred_df: pd.DataFrame):

    fig, axes = plt.subplots(7, 1, figsize=(12, 36), sharex=True)

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

    axes[5].plot(pred_df["step"], pred_df["Tg_in"], label="高压过热器1入口烟温")
    axes[5].set_ylabel("温度")
    axes[5].set_title("烟气入口温度")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(pred_df["step"], pred_df["Ts_in"], label="高压过热器1入口蒸汽温度")
    axes[6].set_ylabel("温度")
    axes[6].set_title("蒸汽入口温度")
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    # axes[7].plot(pred_df["step"], pred_df["cp_g"], label="烟气 cp")
    # axes[7].set_ylabel("J/(kg·K)")
    # axes[7].set_title("烟气比热")
    # axes[7].legend()
    # axes[7].grid(True, alpha=0.3)

    # axes[8].plot(pred_df["step"], pred_df["cp_s"], label="蒸汽 cp")
    # axes[8].set_ylabel("J/(kg·K)")
    # axes[8].set_title("蒸汽比热")
    # axes[8].legend()
    # axes[8].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================================================
# 5. 主程序
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
        "汽包压力"
    ]

    df = df[cols_needed].dropna().reset_index(drop=True)

    seg_df = df.iloc[:5000].copy()

    pred_df = predict_on_segment(seg_df, verbose=False)

    pred_df.to_csv(r"HRSG\superheater_prediction_results.csv", index=False, encoding="utf-8-sig")

    plot_results(pred_df)