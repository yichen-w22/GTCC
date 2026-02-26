import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def remove_outlier_df(df, cols, n=3):
    df = df.copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")  # 保证可算
    for c in cols:
        x = df[c]
        mu, sigma = x.mean(skipna=True), x.std(skipna=True)
        df[c] = x.where(x.between(mu - n*sigma, mu + n*sigma))  # 离群点 -> NaN
    return df

def steady_state_index(
    series: pd.Series,
    time_dt: pd.Series,
    window: int = 10,
    std_quantile: float = 0.7,
    slope_quantile: float = 0.7,
    power_low: float = 100,
    power_high: float = 300,
):
    # 只保留必要列，保留原 index
    tmp = pd.DataFrame({
        "value": pd.to_numeric(series, errors="coerce"),
        "time": pd.to_datetime(time_dt, errors="coerce")
    }).dropna(subset=["value", "time"])

    v = tmp["value"]

    # rolling 统计量
    rolling_std = v.rolling(window=window, center=True).std()
    rolling_slope = v.diff().rolling(window=window, center=True).mean().abs()

    # 自适应阈值（分位数）
    std_thr = rolling_std.quantile(std_quantile)
    slope_thr = rolling_slope.quantile(slope_quantile)

    # 稳态判据
    steady_mask = (rolling_std < std_thr) & (rolling_slope < slope_thr)

    # 功率区间筛选
    if power_low is not None:
        steady_mask &= (v >= power_low)
    if power_high is not None:
        steady_mask &= (v <= power_high)

    # 返回原 df 的 index
    steady_index = tmp.index[steady_mask.fillna(False)]

    return steady_index, std_thr, slope_thr


def compute_isentropic_efficiency(df: pd.DataFrame,
                                  k: float = 1.4,
                                  T1_col: str = "comp_inlet_T_K",
                                  P1_col: str = "comp_inlet_P_MPa",
                                  T2_col: str = "comp_exit_T_K",
                                  P2_col: str = "comp_exit_P_MPa",
                                  out_col: str = "eta_c") -> pd.DataFrame:
    df = df.copy()

    T1 = df[T1_col].astype(float)
    P1 = df[P1_col].astype(float)
    T2 = df[T2_col].astype(float)
    P2 = df[P2_col].astype(float)

    pr = P2 / P1
    exp = (k - 1.0) / k
    T2s = T1 * np.power(pr, exp)

    # 基本有效性
    valid = (
        (T1 > 0) & (P1 > 0) &
        (pr > 1.0) &
        (T2 > T1) &
        np.isfinite(T2s) & np.isfinite(T2)
    )

    eta = np.full(len(df), np.nan, dtype=float)
    eta[valid.values] = ((T2s[valid] - T1[valid]) / (T2[valid] - T1[valid])).values

    eta[(eta < 0) | (eta > 1.2)] = np.nan

    df[out_col] = eta
    return df

def compute_corrected(df: pd.DataFrame) -> pd.DataFrame:
    
    """
      Wc = W * sqrt(T1/Tref) / (P1/Pref)
      Nc = N * sqrt(Tref/T1)
    """
    df["Wc"] = df["air_inlet_flow_kg_s"] * np.sqrt(df["comp_inlet_T_K"] / 288.15) / (df["comp_inlet_P_MPa"] / 0.101325)
    df["Nc"] = df["shaft_rpm"] * np.sqrt(288.15 / df["comp_inlet_T_K"])
    
    df = remove_outlier_df(df, ["Wc"], n=3)
    df = remove_outlier_df(df, ["Nc"], n=3)

    return df

def plot_3d_map_scatter_and_binned_surface(
    df: pd.DataFrame,
    bins_x: int = 40,
    bins_y: int = 40):
    
    d = df[["Wc", "Nc", "eta_c"]].dropna().copy()

    # 分箱边界
    x_edges = np.linspace(d["Wc"].quantile(0.01), d["Wc"].quantile(0.99), bins_x + 1)
    y_edges = np.linspace(d["Nc"].quantile(0.01), d["Nc"].quantile(0.99), bins_y + 1)
    d["xb"] = pd.cut(d["Wc"], x_edges, include_lowest=True)
    d["yb"] = pd.cut(d["Nc"], y_edges, include_lowest=True)

    g = d.groupby(["xb", "yb"], observed=True)["eta_c"].mean().reset_index()
    # 网格中心点
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    Z = np.full((bins_x, bins_y), np.nan)

    # 映射到矩阵
    xb_codes = g["xb"].cat.codes.values
    yb_codes = g["yb"].cat.codes.values
    Z[xb_codes, yb_codes] = g["eta_c"].values

    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")  # X: Wc, Y: Nc

    mask = np.isfinite(Z)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 1) 画分箱均值面（先画面再画点，点更清楚）
    ax.plot_trisurf(
        X[mask].ravel(),   # Wc
        Y[mask].ravel(),   # Nc
        Z[mask].ravel(),   # eta mean
        linewidth=0.2,
        antialiased=True,
        alpha=0.7,
        color='red'
    )

    # 2) 画原始散点
    ax.scatter(
        d["Wc"].values, d["Nc"].values, d["eta_c"].values,
        s=1, alpha=0.5
    )

    ax.set_xlabel("Wc")
    ax.set_ylabel("Nc")
    ax.set_zlabel("eta_c")
    ax.set_title("Compressor Map (scatter + binned mean surface)")
    plt.tight_layout()
    plt.show()


def compute_delta_eta_by_operating_bin(df: pd.DataFrame,
                                       bins_x: int = 30,
                                       bins_y: int = 30,) -> pd.DataFrame:

    x_edges = np.linspace(df["Wc"].quantile(0.01), df["Wc"].quantile(0.99), bins_x + 1)
    y_edges = np.linspace(df["Nc"].quantile(0.01), df["Nc"].quantile(0.99), bins_y + 1)

    df["xb"] = pd.cut(df["Wc"], x_edges, include_lowest=True)
    df["yb"] = pd.cut(df["Nc"], y_edges, include_lowest=True)
    # 统计每个工况箱子的全年平均效率 + 样本数
    bin_stats = (
        df.groupby(["xb", "yb"], observed=True)["eta_c"]
         .agg(eta_mean_bin="mean")
         .reset_index()
    )

    # 回填到每条记录
    df = df.merge(bin_stats[["xb", "yb", "eta_mean_bin"]], on=["xb", "yb"], how="inner")
    df["delta_eta"] = df["eta_c"] - df["eta_mean_bin"]
    
    return df

def plot_delta_trend(df: pd.DataFrame):
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])

    g = (
        d.set_index("timestamp")["delta_eta"]
         .resample("1h")
         .mean()
         .dropna()
    )

    t = (g.index - g.index[0]).total_seconds().values
    y = g.values

    k, b = np.polyfit(t, y, 1)
    y_trend = k * t + b

    plt.figure(figsize=(10, 4.8))
    plt.scatter(g.index, y, s=1, alpha=0.7, label="hourly Δη")
    plt.plot(g.index, y_trend, "r-", linewidth=2,
             label=f"slope = {k*3600*24*365:.3e} / year")
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("delta_eta")
    plt.title("Δη trend")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return k


# =========================
# 4) 主流程（你直接改 df 名字即可）
# =========================
def run_all(df: pd.DataFrame,
            k: float = 1.4,
            bins_x: int = 30,
            bins_y: int = 30):

    # 0) 时间列
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1) 等熵效率
    df = compute_isentropic_efficiency(df, k=k, out_col="eta_c")

    # 2) 折合量
    df = compute_corrected(df)

    # （可选）简单清洗：只保留关键列有效的点
    core = df.dropna(subset=["Wc", "Nc", "eta_c"]).copy()

    # 3) 特性曲线展示：先散点，再表面（数据多时建议散点采样 + 表面网格）
    plot_3d_map_scatter_and_binned_surface(core, bins_x, bins_y)

    # 4) 排除工况影响：计算 Δη 并看趋势
    d_delta = compute_delta_eta_by_operating_bin(df, bins_x, bins_y)

    # 趋势图：每周（你也可以改成 "1MS" 看每月）
    plot_delta_trend(d_delta)

    return df, d_delta


# =========================
# 5) 用法示例
# =========================
# 假设你的原始数据叫 df：

df = pd.read_csv(r"data_processing\for_degradation\outcome\jqrd_compressor.csv")

df = df.loc[
    steady_state_index(
        df["unit_power_MW"],
        pd.to_datetime(df["timestamp"]),
        window=10,
        std_quantile=0.9,
        slope_quantile=0.9,
    )[0]
]

df_out, d_delta = run_all(df, k=1.4, bins_x=7, bins_y=7)
