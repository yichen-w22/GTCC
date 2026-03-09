import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


# =========================================================
# 0. 全局常数
# =========================================================
cp_g = 1100.0   # J/(kg·K)
cp_s = 2600.0   # J/(kg·K)
dt = 60.0       # s
a_s = 0.8       # 蒸汽侧指数固定


# =========================================================
# 1. 单步预测
# =========================================================
def one_step(Tg_in, mg, Ts_in, ms, Tm, K_g, a_g, K_s, a_s, C_m):
    mg = max(mg, 1e-8)
    ms = max(ms, 1e-8)

    UA_g = K_g * mg ** a_g
    UA_s = K_s * ms ** a_s

    Tg_out = Tg_in - UA_g / (mg * cp_g) * (Tg_in - Tm)
    Ts_out = Ts_in + UA_s / (ms * cp_s) * (Tm - Ts_in)

    Q_gm = UA_g * (Tg_in - Tm)
    Q_ms = UA_s * (Tm - Ts_in)

    Tm_next = Tm + dt / C_m * (Q_gm - Q_ms)

    return Tg_out, Ts_out, Tm_next


# =========================================================
# 2. 用前两个点估计初始金属温度
# =========================================================
def estimate_initial_Tm(two_points: pd.DataFrame, K_g, a_g, K_s, a_s, C_m, n_grid=200):
    row0 = two_points.iloc[0]
    row1 = two_points.iloc[1]

    Tg_in_0 = row0["高压过热器1入口烟温"]
    mg_0 = row0["烟气流量"]
    Ts_in_0 = row0["高压过热器1入口蒸汽温度"]
    ms_0 = row0["高压主蒸汽流量"]

    Tg_out_true_1 = row1["高压过热器1出口烟温"]
    Ts_out_true_1 = row1["高压过热器1出口蒸汽温度"]

    Tm_low = min(Ts_in_0, Tg_in_0)
    Tm_high = max(Ts_in_0, Tg_in_0)

    Tm_candidates = np.linspace(Tm_low, Tm_high, n_grid)

    best_Tm0 = None
    best_loss = np.inf

    for Tm0 in Tm_candidates:
        Tg_out_pred_1, Ts_out_pred_1, _ = one_step(
            Tg_in_0, mg_0, Ts_in_0, ms_0, Tm0,
            K_g, a_g, K_s, a_s, C_m
        )

        loss = (
            (Tg_out_pred_1 - Tg_out_true_1) ** 2
            + (Ts_out_pred_1 - Ts_out_true_1) ** 2
        )

        if loss < best_loss:
            best_loss = loss
            best_Tm0 = Tm0

    return best_Tm0, best_loss


# =========================================================
# 3. 对单个片段做多步预测
#    seg_len = 12 时：前2点估计Tm0，后10步预测
# =========================================================
def predict_on_segment(seg_df: pd.DataFrame, K_g, a_g, K_s, a_s, C_m, pred_steps=10):
    seg_df = seg_df.reset_index(drop=True)

    if len(seg_df) < pred_steps + 1:
        return None

    Tm0, init_loss = estimate_initial_Tm(
        seg_df.iloc[:2], K_g, a_g, K_s, a_s, C_m
    )

    Tm = Tm0
    results = []

    for k in range(pred_steps):
        row_in = seg_df.iloc[k]
        row_true = seg_df.iloc[k + 1]

        Tg_in = row_in["高压过热器1入口烟温"]
        mg = row_in["烟气流量"]
        Ts_in = row_in["高压过热器1入口蒸汽温度"]
        ms = row_in["高压主蒸汽流量"]

        Tg_out_pred, Ts_out_pred, Tm_next = one_step(
            Tg_in, mg, Ts_in, ms, Tm,
            K_g, a_g, K_s, a_s, C_m
        )

        results.append({
            "step": k + 1,
            "Tm_used": Tm,
            "Tg_out_pred": Tg_out_pred,
            "Ts_out_pred": Ts_out_pred,
            "Tg_out_true": row_true["高压过热器1出口烟温"],
            "Ts_out_true": row_true["高压过热器1出口蒸汽温度"],
            "Tg_err": Tg_out_pred - row_true["高压过热器1出口烟温"],
            "Ts_err": Ts_out_pred - row_true["高压过热器1出口蒸汽温度"],
        })

        Tm = Tm_next

    pred_df = pd.DataFrame(results)
    pred_df["init_loss"] = init_loss
    return pred_df


# =========================================================
# 4. 从原始数据中找连续无空值片段
# =========================================================
def split_into_continuous_segments(df: pd.DataFrame, cols_needed, min_len=12):
    df_use = df[cols_needed].copy()

    valid_mask = df_use.notna().all(axis=1)
    group_id = (valid_mask != valid_mask.shift()).cumsum()

    segments = []
    for _, seg in df_use[valid_mask].groupby(group_id[valid_mask]):
        seg = seg.reset_index(drop=True)
        if len(seg) >= min_len:
            segments.append(seg)

    return segments


# =========================================================
# 5. 把长片段切成固定长度的小片段
# =========================================================
def sample_segments_from_long_ones(segments, seg_len=12, stride=6, max_segments=None):
    sampled = []

    for seg in segments:
        n = len(seg)
        if n < seg_len:
            continue

        for start in range(0, n - seg_len + 1, stride):
            sampled.append(seg.iloc[start:start + seg_len].reset_index(drop=True))

    if max_segments is not None and len(sampled) > max_segments:
        sampled = sampled[:max_segments]

    return sampled


# =========================================================
# 6. 评估一组参数在所有片段上的误差
# =========================================================
def evaluate_params_on_segments(segment_list, K_g, a_g, K_s, a_s, C_m, pred_steps=10):
    gas_sse = 0.0
    steam_sse = 0.0
    gas_n = 0
    steam_n = 0

    all_pred = []

    for seg in segment_list:
        pred_df = predict_on_segment(
            seg, K_g, a_g, K_s, a_s, C_m, pred_steps=pred_steps
        )
        if pred_df is None:
            continue

        gas_sse += np.sum((pred_df["Tg_out_pred"] - pred_df["Tg_out_true"]) ** 2)
        steam_sse += np.sum((pred_df["Ts_out_pred"] - pred_df["Ts_out_true"]) ** 2)

        gas_n += len(pred_df)
        steam_n += len(pred_df)

        all_pred.append(pred_df)

    if gas_n == 0 or steam_n == 0:
        return None

    gas_mse = gas_sse / gas_n
    steam_mse = steam_sse / steam_n
    total_mse = gas_mse + steam_mse

    return {
        "K_g": K_g,
        "a_g": a_g,
        "K_s": K_s,
        "a_s": a_s,
        "C_m": C_m,
        "gas_mse": gas_mse,
        "steam_mse": steam_mse,
        "total_mse": total_mse,
        "n_points": gas_n
    }


# =========================================================
# 7. 网格搜索
#    这里直接接受 list，例如 [2e3, 4e3, 6e3, 8e3]
# =========================================================
def grid_search_params(
    df: pd.DataFrame,
    kg_grid,
    ag_grid,
    ks_grid,
    cm_grid,
    a_s=0.8,
    seg_len=12,
    stride=6,
    min_len=12,
    max_segments=None,
    cols_needed=None
):
    if cols_needed is None:
        cols_needed = [
            "高压过热器1入口烟温",
            "高压过热器1出口烟温",
            "烟气流量",
            "高压过热器1入口蒸汽温度",
            "高压主蒸汽流量",
            "高压过热器1出口蒸汽温度",
        ]

    segments = split_into_continuous_segments(df, cols_needed, min_len=min_len)
    segment_list = sample_segments_from_long_ones(
        segments,
        seg_len=seg_len,
        stride=stride,
        max_segments=max_segments
    )

    print(f"连续无空值大段数量: {len(segments)}")
    print(f"参与搜索的小片段数量: {len(segment_list)}")

    results = []

    total_cases = len(kg_grid) * len(ag_grid) * len(ks_grid) * len(cm_grid)
    case_idx = 0

    for K_g, a_g, K_s, C_m in product(kg_grid, ag_grid, ks_grid, cm_grid):
        case_idx += 1
        print(f"正在搜索: {case_idx}/{total_cases} | "
              f"K_g={K_g}, a_g={a_g}, K_s={K_s}, C_m={C_m}")

        eval_res = evaluate_params_on_segments(
            segment_list=segment_list,
            K_g=K_g,
            a_g=a_g,
            K_s=K_s,
            a_s=a_s,
            C_m=C_m,
            pred_steps=seg_len - 2
        )

        if eval_res is not None:
            results.append(eval_res)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("total_mse").reset_index(drop=True)
    return result_df, segment_list


# =========================================================
# 8. 用最优参数对单个片段画图
# =========================================================
def plot_one_segment_prediction(seg_df, K_g, a_g, K_s, a_s, C_m, pred_steps=10):
    pred_df = predict_on_segment(
        seg_df, K_g, a_g, K_s, a_s, C_m, pred_steps=pred_steps
    )

    if pred_df is None:
        print("该片段长度不足，无法绘图。")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 4))
    plt.plot(pred_df["Ts_out_true"].values, label="蒸汽出口真实值")
    plt.plot(pred_df["Ts_out_pred"].values, label="蒸汽出口预测值")
    plt.title("蒸汽出口温度预测")
    plt.xlabel("步数")
    plt.ylabel("温度")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(pred_df["Tg_out_true"].values, label="烟气出口真实值")
    plt.plot(pred_df["Tg_out_pred"].values, label="烟气出口预测值")
    plt.title("烟气出口温度预测")
    plt.xlabel("步数")
    plt.ylabel("温度")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 9. 主程序
# =========================================================
if __name__ == "__main__":
    # -----------------------------------------------------
    # 9.1 读数据
    # -----------------------------------------------------
    file_path = r"HRSG\jyrd高压余热锅炉过热器1_滑窗平均数据.csv"
    df = pd.read_csv(file_path)

    # -----------------------------------------------------
    # 9.2 需要的列
    # -----------------------------------------------------
    cols_needed = [
        "高压过热器1入口烟温",
        "高压过热器1出口烟温",
        "烟气流量",
        "高压过热器1入口蒸汽温度",
        "高压主蒸汽流量",
        "高压过热器1出口蒸汽温度",
    ]

    # -----------------------------------------------------
    # 9.3 参数网格
    #     直接按你想要的格式写
    # -----------------------------------------------------
    kg_grid = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    ag_grid = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    ks_grid = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    cm_grid = [1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]

    # -----------------------------------------------------
    # 9.4 网格搜索
    #     seg_len=12 表示：
    #     前2点估计Tm0，后10步预测
    # -----------------------------------------------------
    result_df, segment_list = grid_search_params(
        df=df,
        kg_grid=kg_grid,
        ag_grid=ag_grid,
        ks_grid=ks_grid,
        cm_grid=cm_grid,
        a_s=0.8,
        seg_len=12,
        stride=6,
        min_len=12,
        max_segments=300,
        cols_needed=cols_needed
    )

    # -----------------------------------------------------
    # 9.5 输出结果
    # -----------------------------------------------------
    print("\n搜索结果前10名：")
    print(result_df.head(10))

    # 保存结果
    result_df.to_csv(
        r"HRSG\grid_search_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # 9.6 取最优参数并绘图
    # -----------------------------------------------------
    if len(result_df) > 0 and len(segment_list) > 0:
        best = result_df.iloc[0]

        print("\n最优参数：")
        print(best)

        plot_one_segment_prediction(
            seg_df=segment_list[0],
            K_g=best["K_g"],
            a_g=best["a_g"],
            K_s=best["K_s"],
            a_s=best["a_s"],
            C_m=best["C_m"],
            pred_steps=10
        )