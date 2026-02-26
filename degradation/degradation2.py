import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


CSV_PATH = r"data_processing\for_degradation\outcome\jqrd_compressor_degradation.csv"
OUT_ALL = r"temp\eta_residual_all.csv"
OUT_TREND = r"temp\eta_residual_trend_by_bins.csv"


def fit_avg_surface(df, y="eta_isentropic",
                    x=("comp_inlet_T_K", "comp_inlet_P_MPa", "IGV_deg", "unit_power_MW"),
                    degree=2, alpha=1.0):
    d = df[list(x) + [y, "timestamp"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"] + list(x) + [y]).sort_values("timestamp")

    X = d[list(x)].to_numpy()
    Y = d[y].to_numpy()

    m = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=alpha))
    m.fit(X, Y)

    d["eta_hat"] = m.predict(X)
    d["delta_eta"] = d[y] - d["eta_hat"]
    return d, m


def residual_trend_by_bins(d, bins_col="unit_power_MW", q=10, freq="7D"):
    dd = d.copy()
    dd["timestamp"] = pd.to_datetime(dd["timestamp"], errors="coerce")
    dd = dd.dropna(subset=["timestamp", bins_col, "delta_eta"])

    dd["bin"] = pd.qcut(dd[bins_col], q=q, duplicates="drop")
    out = (dd.groupby(["bin", pd.Grouper(key="timestamp", freq=freq)])["delta_eta"]
             .median()
             .reset_index()
             .rename(columns={"delta_eta": "delta_eta_med"}))
    return out


def cross_fit_check(df, y="eta_isentropic",
                    x=("comp_inlet_T_K", "comp_inlet_P_MPa", "IGV_deg", "unit_power_MW"),
                    degree=2, alpha=1.0):
    d = df[list(x) + [y, "timestamp"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"] + list(x) + [y]).sort_values("timestamp")

    n = len(d)
    mid = n // 2

    def fit(train):
        m = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=alpha))
        m.fit(train[list(x)].to_numpy(), train[y].to_numpy())
        return m

    m1 = fit(d.iloc[:mid])
    m2 = fit(d.iloc[mid:])

    d1 = d.copy()
    d1["eta_hat_fwd"] = m1.predict(d1[list(x)].to_numpy())
    d1["delta_fwd"] = d1[y] - d1["eta_hat_fwd"]

    d2 = d.copy()
    d2["eta_hat_bwd"] = m2.predict(d2[list(x)].to_numpy())
    d2["delta_bwd"] = d2[y] - d2["eta_hat_bwd"]

    s = pd.DataFrame({
        "fit_front_apply_back_median": [d1.iloc[mid:]["delta_fwd"].median()],
        "fit_back_apply_front_median": [d2.iloc[:mid]["delta_bwd"].median()],
        "front_half_median": [d1.iloc[:mid]["delta_fwd"].median()],
        "back_half_median": [d1.iloc[mid:]["delta_fwd"].median()],
        "n_total": [n],
        "n_front": [mid],
        "n_back": [n - mid],
    })
    return d, d1, d2, s


def visualize_all(d, trend, summary, y="eta_isentropic", bins_col="unit_power_MW",
                  out_dir=None, show=True):
    dd = d.copy()
    dd["timestamp"] = pd.to_datetime(dd["timestamp"], errors="coerce")
    dd = dd.dropna(subset=["timestamp"]).sort_values("timestamp")
    s = dd.set_index("timestamp")

    # 1) eta vs eta_hat
    plt.figure(figsize=(5.6, 5.2))
    plt.scatter(dd["eta_hat"], dd[y], s=3, alpha=0.25)
    lo = np.nanmin([dd["eta_hat"].min(), dd[y].min()])
    hi = np.nanmax([dd["eta_hat"].max(), dd[y].max()])
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("eta_hat (avg surface)")
    plt.ylabel(f"{y} (measured)")
    plt.title("Measured vs Predicted")
    plt.tight_layout()
    if out_dir: plt.savefig(f"{out_dir}/01_eta_vs_hat.png", dpi=200)
    if show: plt.show()
    else: plt.close()

    # 2) residual vs time (scatter + weekly median)
    weekly = s["delta_eta"].resample("7D").median().dropna()
    plt.figure(figsize=(10, 4.6))
    plt.scatter(dd["timestamp"], dd["delta_eta"], s=2, alpha=0.18)
    plt.plot(weekly.index, weekly.values, linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("delta_eta = eta - eta_hat")
    plt.title("Residual vs Time (scatter + weekly median)")
    locator = mdates.AutoDateLocator()
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.tight_layout()
    if out_dir: plt.savefig(f"{out_dir}/02_residual_time.png", dpi=200)
    if show: plt.show()
    else: plt.close()

    # 3) trend lines by bins
    trend2 = trend.copy()
    trend2["timestamp"] = pd.to_datetime(trend2["timestamp"], errors="coerce")
    trend2 = trend2.dropna(subset=["timestamp", "delta_eta_med"])

    plt.figure(figsize=(10, 5.2))
    for b, sub in trend2.groupby("bin"):
        plt.plot(sub["timestamp"], sub["delta_eta_med"], linewidth=1, alpha=0.85)
    plt.axhline(0, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("median delta_eta (per bin)")
    plt.title(f"Residual Trend by {bins_col} bins")
    locator = mdates.AutoDateLocator()
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.tight_layout()
    if out_dir: plt.savefig(f"{out_dir}/03_trend_by_bins_lines.png", dpi=200)
    if show: plt.show()
    else: plt.close()

    # 4) heatmap
    heat = (trend2.pivot(index="bin", columns="timestamp", values="delta_eta_med")
                  .sort_index())
    plt.figure(figsize=(10, 4.8))
    im = plt.imshow(heat.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="median delta_eta")
    plt.yticks(np.arange(len(heat.index)), [str(b) for b in heat.index])
    plt.xticks([])
    plt.xlabel("time (increasing →)")
    plt.ylabel(f"{bins_col} bins")
    plt.title("Residual Trend Heatmap (bins × time)")
    plt.tight_layout()
    if out_dir: plt.savefig(f"{out_dir}/04_trend_heatmap.png", dpi=200)
    if show: plt.show()
    else: plt.close()

    # 5) residual distribution
    plt.figure(figsize=(6.2, 4.2))
    x = dd["delta_eta"].dropna().to_numpy()
    plt.hist(x, bins=80)
    plt.axvline(0, linewidth=1)
    plt.xlabel("delta_eta")
    plt.ylabel("count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    if out_dir: plt.savefig(f"{out_dir}/05_residual_hist.png", dpi=200)
    if show: plt.show()
    else: plt.close()

    # 6) cross-fit bar
    cols = ["fit_front_apply_back_median", "fit_back_apply_front_median"]
    vals = summary.iloc[0][cols].to_numpy()
    labs = ["fit front → apply back", "fit back → apply front"]

    plt.figure(figsize=(6.6, 3.8))
    plt.bar(labs, vals)
    plt.axhline(0, linewidth=1)
    plt.ylabel("median residual")
    plt.title("Cross-fit sanity check")
    plt.tight_layout()
    if out_dir: plt.savefig(f"{out_dir}/06_cross_fit.png", dpi=200)
    if show: plt.show()
    else: plt.close()

    # 7) residual vs load
    if bins_col in dd.columns:
        plt.figure(figsize=(7.2, 4.4))
        plt.scatter(dd[bins_col], dd["delta_eta"], s=2, alpha=0.2)
        plt.axhline(0, linewidth=1)
        plt.xlabel(bins_col)
        plt.ylabel("delta_eta")
        plt.title("Residual vs Load (check remaining dependence)")
        plt.tight_layout()
        if out_dir: plt.savefig(f"{out_dir}/07_residual_vs_load.png", dpi=200)
        if show: plt.show()
        else: plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    d, model = fit_avg_surface(df, degree=2, alpha=1.0)
    trend = residual_trend_by_bins(d, bins_col="unit_power_MW", q=10, freq="30D")
    d_all, d_fwd, d_bwd, summary = cross_fit_check(df, degree=2, alpha=1.0)

    print(summary)

    d.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    trend.to_csv(OUT_TREND, index=False, encoding="utf-8-sig")

    visualize_all(d, trend, summary, y="eta_isentropic", bins_col="unit_power_MW",
                  out_dir=None, show=True)


if __name__ == "__main__":
    main()