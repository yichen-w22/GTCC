import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================
# 0. 全局常数
# =========================================================
cp_g = 1100.0   # J/(kg·K)
cp_s = 2550.0   # J/(kg·K)
dt = 60.0       # s

a_s = 0.8       # 蒸汽侧指数固定
K_s = 2.1e4       # 蒸汽侧系数

a_g = 0.7       # 烟气侧指数固定
K_g = 3.8e4       # 烟气侧系数
C_m = 0.9e8      # 金属热容
T_m0 = 760.0    # 金属初始温度


# =========================================================
# 1. 单步预测
# =========================================================
# def one_step(Tg_in, mg, Ts_in, ms, Tm):
#     mg = max(float(mg), 1e-8)
#     ms = max(float(ms), 1e-8)

#     UA_g = K_g * mg ** a_g
#     UA_s = K_s * ms ** a_s

#     Tg_out = Tg_in - UA_g / (mg * cp_g) * (Tg_in - Tm)
#     Ts_out = Ts_in + UA_s / (ms * cp_s) * (Tm - Ts_in)

#     Q_gm = UA_g * (Tg_in - Tm)
#     Q_ms = UA_s * (Tm - Ts_in)

#     Tm_next = Tm + dt / C_m * (Q_gm - Q_ms)

#     return Tg_out, Ts_out, Tm_next, UA_g, UA_s, Q_gm, Q_ms

# def one_step(Tg_in, mg, Ts_in, ms, Tm):
#     mg = max(float(mg), 1e-8)
#     ms = max(float(ms), 1e-8)

#     UA_g = K_g * mg ** a_g
#     UA_s = K_s * ms ** a_s

#     UA_tot = UA_g + UA_s
#     tau = C_m / UA_tot
#     T_eq = (UA_g * Tg_in + UA_s * Ts_in) / UA_tot

#     # 解析更新后的金属温度
#     exp_term = np.exp(-dt / tau)
#     Tm_next = T_eq + (Tm - T_eq) * exp_term

#     # 步内平均金属温度
#     Tm_avg = T_eq + (Tm - T_eq) * (tau / dt) * (1 - exp_term)

#     # 用步内平均金属温度计算出口
#     Tg_out = Tg_in - UA_g / (mg * cp_g) * (Tg_in - Tm_avg)
#     Ts_out = Ts_in + UA_s / (ms * cp_s) * (Tm_avg - Ts_in)

#     Q_gm = UA_g * (Tg_in - Tm_avg)
#     Q_ms = UA_s * (Tm_avg - Ts_in)

#     return Tg_out, Ts_out, Tm_next, UA_g, UA_s, Q_gm, Q_ms

def one_step(Tg_in, mg, Ts_in, ms, Tm):
    mg = max(float(mg), 1e-8)
    ms = max(float(ms), 1e-8)

    # 两侧等效传热能力
    UA_g = K_g * mg ** a_g
    UA_s = K_s * ms ** a_s

    # -------------------------------------------------
    # 1. 由平均温度形式推导出的显式出口温度
    # 烟气侧：
    # mg*cp_g*(Tg_in - Tg_out) = UA_g * ((Tg_in + Tg_out)/2 - Tm)
    #
    # 蒸汽侧：
    # ms*cp_s*(Ts_out - Ts_in) = UA_s * (Tm - (Ts_in + Ts_out)/2)
    # -------------------------------------------------
    Tg_out = (
        (mg * cp_g - UA_g / 2.0) * Tg_in + UA_g * Tm
    ) / (mg * cp_g + UA_g / 2.0)

    Ts_out = (
        (ms * cp_s - UA_s / 2.0) * Ts_in + UA_s * Tm
    ) / (ms * cp_s + UA_s / 2.0)

    # 平均温度
    Tg_avg = 0.5 * (Tg_in + Tg_out)
    Ts_avg = 0.5 * (Ts_in + Ts_out)

    # 两侧换热量
    Q_gm = UA_g * (Tg_avg - Tm)
    Q_ms = UA_s * (Tm - Ts_avg)

    # -------------------------------------------------
    # 2. 金属节点解析离散更新
    # C_m * dTm/dt = UA_g*(Tg_avg - Tm) - UA_s*(Tm - Ts_avg)
    # -------------------------------------------------
    UA_tot = UA_g + UA_s
    T_eq = (UA_g * Tg_avg + UA_s * Ts_avg) / UA_tot
    tau = C_m / UA_tot

    Tm_next = T_eq + (Tm - T_eq) * np.exp(-dt / tau)

    return Tg_out, Ts_out, Tm_next, UA_g, UA_s, Q_gm, Q_ms

# =========================================================
# 2. 对单个片段做多步预测
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

        Tg_out_pred, Ts_out_pred, Tm_next, UA_g, UA_s, Q_gm, Q_ms = one_step(
            Tg_in, mg, Ts_in, ms, Tm
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

        # if verbose:
        #     print(
        #         f"step={k+1:02d} | "
        #         f"Tg_in={Tg_in:.2f}, Ts_in={Ts_in:.2f}, "
        #         f"mg={mg:.2f}, ms={ms:.2f} | "
        #         f"Tm={Tm:.2f} -> Tm_next={Tm_next:.2f} | "
        #         f"Tg_out_pred={Tg_out_pred:.2f}, Tg_out_true={Tg_out_true:.2f}, err={Tg_out_pred - Tg_out_true:.2f} | "
        #         f"Ts_out_pred={Ts_out_pred:.2f}, Ts_out_true={Ts_out_true:.2f}, err={Ts_out_pred - Ts_out_true:.2f}"
        #     )

        Tm = Tm_next

    pred_df = pd.DataFrame(results)
    return pred_df

# =========================================================
# 4. 绘图
# =========================================================
def plot_results(pred_df: pd.DataFrame):

    fig, axes = plt.subplots(7, 1, figsize=(12, 30), sharex=True)

    # =========================
    # 1 蒸汽出口温度
    # =========================
    axes[0].plot(pred_df["step"], pred_df["Ts_out_true"], label="蒸汽出口真实值")
    axes[0].plot(pred_df["step"], pred_df["Ts_out_pred"], label="蒸汽出口预测值")
    axes[0].set_ylabel("温度")
    axes[0].set_title("蒸汽出口温度预测")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # =========================
    # 2 烟气出口温度
    # =========================
    axes[1].plot(pred_df["step"], pred_df["Tg_out_true"], label="烟气出口真实值")
    axes[1].plot(pred_df["step"], pred_df["Tg_out_pred"], label="烟气出口预测值")
    axes[1].set_ylabel("温度")
    axes[1].set_title("烟气出口温度预测")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # =========================
    # 3 金属温度
    # =========================
    axes[2].plot(pred_df["step"], pred_df["Tm_used"], label="金属温度")
    axes[2].set_ylabel("温度")
    axes[2].set_title("金属温度演化")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # =========================
    # 4 烟气流量
    # =========================
    axes[3].plot(pred_df["step"], pred_df["mg"], label="烟气流量")
    axes[3].set_ylabel("流量")
    axes[3].set_title("烟气流量")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # =========================
    # 5 蒸汽流量
    # =========================
    axes[4].plot(pred_df["step"], pred_df["ms"], label="蒸汽流量")
    axes[4].set_ylabel("流量")
    axes[4].set_title("蒸汽流量")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    # =========================
    # 6 烟气入口温度
    # =========================
    axes[5].plot(pred_df["step"], pred_df["Tg_in"], label="高压过热器1入口烟温")
    axes[5].set_ylabel("温度")
    axes[5].set_title("烟气入口温度")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    # =========================
    # 7 蒸汽入口温度
    # =========================
    axes[6].plot(pred_df["step"], pred_df["Ts_in"], label="高压过热器1入口蒸汽温度")
    axes[6].set_ylabel("温度")
    axes[6].set_title("蒸汽入口温度")
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

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

    # 你可以改成取前50个点、后100个点等
    seg_df = df.iloc[:5000].copy()

    pred_df = predict_on_segment(seg_df, verbose=True)

    # print("\n前100步结果：")
    # print(pred_df.head(100))

    # 保存结果
    pred_df.to_csv(r"HRSG\superheater_prediction_results.csv", index=False, encoding="utf-8-sig")

    # 绘图
    plot_results(pred_df)