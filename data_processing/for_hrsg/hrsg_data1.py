import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def add_group_mean_or_split(
    df_out: pd.DataFrame,
    df_raw: pd.DataFrame,
    cols: list[str],
    base_name: str,
    cv_thr: float,
    eps: float = 1e-9,
    keep_stats: bool = True,
):
    cols_exist = [c for c in cols if c in df_raw.columns]
    if len(cols_exist) == 0:
        print(f"[WARN] {base_name}: 没有找到任何列")
        return

    X = df_raw[cols_exist].apply(pd.to_numeric, errors="coerce")
    mean = X.mean(axis=1, skipna=True)
    std = X.std(axis=1, skipna=True)
    cv = std / (mean.abs() + eps)

    split_mask = cv > cv_thr

    # 合并列：一致时用均值；不一致时置空
    df_out[base_name] = mean
    df_out.loc[split_mask, base_name] = np.nan

    # 分列：不一致时保留各测点；一致时置空
    for c in cols_exist:
        out_c = f"{base_name}__{c}"
        df_out[out_c] = X[c]
        df_out.loc[~split_mask, out_c] = np.nan

    if keep_stats:
        df_out[f"{base_name}_mean"] = mean
        df_out[f"{base_name}_std"] = std
        df_out[f"{base_name}_cv"] = cv
        df_out[f"{base_name}_split"] = split_mask.astype(int)


# =========================
# 读取数据
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
p = BASE_DIR / Path(r"datareader_new\jyrd\outcome\jyrd高压余热锅炉_1min.pkl")
df_raw = pd.read_pickle(p)

# 处理空列名：用上一列名填充
cols = pd.Series(df_raw.columns)
df_raw.columns = cols.mask(cols.isna() | (cols == ""), cols.shift())

# 重名列加后缀 .1 .2 ...
cols = pd.Series(df_raw.columns)
df_raw.columns = cols + cols.groupby(cols).cumcount().replace(0, "").astype(str).radd(".").replace(".", "")

# 0 视为缺失
df_raw = df_raw.replace(0, np.nan)

# 输出表
df = pd.DataFrame(index=df_raw.index)
df["timestamp"] = df_raw.index


# =========================
# 阈值（我给的默认值）
# =========================
# 温度类：1%（传感器应较一致；>1% 基本可判为有空间梯度/测点问题）
CV_T = 0.01
# 压力/水位：2%（有工况波动和测量噪声，稍放宽）
CV_P = 0.02
CV_L = 0.02
# 流量：5%（电厂流量测点常见差异更大）
CV_F = 0.05


# =========================
# 烟温（6点）
# =========================
add_group_mean_or_split(
    df,
    df_raw,
    [f"#1炉高压过热器1入口烟温{i}" for i in range(1, 7)],
    "高压过热器1入口烟温",
    cv_thr=CV_T,
)
add_group_mean_or_split(
    df,
    df_raw,
    [f"#1炉高压蒸发器入口烟温{i}" for i in range(1, 7)],
    "高压蒸发器入口烟温",
    cv_thr=CV_T,
)
add_group_mean_or_split(
    df,
    df_raw,
    [f"#1炉高压省煤器3入口烟温{i}" for i in range(1, 7)],
    "高压省煤器3入口烟温",
    cv_thr=CV_T,
)
add_group_mean_or_split(
    df,
    df_raw,
    [f"#1炉中压过热器入口烟温{i}" for i in range(1, 7)],
    "中压过热器入口烟温",
    cv_thr=CV_T,
)

# =========================
# 烟气压力/流量/流速（单列直接取）
# =========================
for k_src, k_dst in [
    ("#1炉烟道入口烟气压力", "烟道入口烟气压力"),
    ("#1炉出口烟气压力", "烟道出口烟气压力"),
    ("#1炉烟囱出口烟气压力", "烟囱出口烟气压力"),
    ("#1炉烟囱出口烟气流量", "烟囱出口烟气流量"),
    ("#1燃机烟气流速", "燃机烟气流速"),
]:
    if k_src in df_raw.columns:
        df[k_dst] = pd.to_numeric(df_raw[k_src], errors="coerce")
    else:
        print(f"[WARN] 缺列: {k_src}")

# =========================
# 给水泵（单列）
# =========================
for k_src, k_dst in [
    ("#1炉#1高压给水泵出口流量", "#1高压给水泵出口流量"),
    ("#1炉#1高压给水泵入口压力", "#1高压给水泵入口压力"),
    ("#1炉#1高压给水泵出口压力", "#1高压给水泵出口压力"),
    ("#1炉#2高压给水泵出口流量", "#2高压给水泵出口流量"),
    ("#1炉#2高压给水泵入口压力", "#2高压给水泵入口压力"),
    ("#1炉#2高压给水泵出口压力", "#2高压给水泵出口压力"),
]:
    if k_src in df_raw.columns:
        df[k_dst] = pd.to_numeric(df_raw[k_src], errors="coerce")
    else:
        print(f"[WARN] 缺列: {k_src}")

# =========================
# 高压给水流量（多列：一致则合并，不一致则分列保存）
# =========================
高压给水流量_cols = [
    "#1炉高压给水流量A",
    "#1炉高压给水流量A.1",
    "#1炉高压给水流量B",
    "#1炉高压给水流量B.1",
    "#1炉高压给水流量C",
    "#1炉高压给水流量C.1",
    "#1锅炉高压给水流量",
]
add_group_mean_or_split(df, df_raw, 高压给水流量_cols, "高压给水流量", cv_thr=CV_F)

# =========================
# 给水温度/压力（单列）
# =========================
for k_src, k_dst in [
    ("#1炉高压给水温度", "高压给水温度"),
    ("#1炉高压给水母管压力", "高压给水母管压力"),
    ("#1炉高压省煤器1入口压力", "高压省煤器1入口压力"),
    ("#1炉高压省煤器1出口压力", "高压省煤器1出口压力"),
]:
    if k_src in df_raw.columns:
        df[k_dst] = pd.to_numeric(df_raw[k_src], errors="coerce")
    else:
        print(f"[WARN] 缺列: {k_src}")

# =========================
# 省煤器出口给水温度（多列）
# =========================
高压省煤器2出口给水温度_cols = [
    "#1炉高压省煤器2出口给水温度A",
    "#1炉高压省煤器2出口给水温度B",
    "#1炉高压省煤器2出口给水温度C",
    "#1炉高压省煤器2出口给水温度D",
]
add_group_mean_or_split(df, df_raw, 高压省煤器2出口给水温度_cols, "高压省煤器2出口给水温度", cv_thr=CV_T)

高压省煤器3出口给水温度_cols = [
    "#1炉高压省煤器3出口给水温度A",
    "#1炉高压省煤器3出口给水温度B",
    "#1炉高压省煤器3出口给水温度C",
    "#1炉高压省煤器3出口给水温度D",
]
add_group_mean_or_split(df, df_raw, 高压省煤器3出口给水温度_cols, "高压省煤器3出口给水温度", cv_thr=CV_T)

# =========================
# 汽包水位/压力/壁温（多列）
# =========================
高压汽包水位_cols = [
    "#1炉高压汽包水位A.1",
    "#1炉高压汽包水位B.1",
    "#1炉高压汽包水位C.1",
    "#1炉高压汽包水位",
]
add_group_mean_or_split(df, df_raw, 高压汽包水位_cols, "高压汽包水位", cv_thr=CV_L)

锅炉高压汽包压力_cols = [
    "#1炉高压汽包压力A",
    "#1炉高压汽包压力B",
    "#1炉高压汽包压力C",
    "#1锅炉高压汽包压力",
]
add_group_mean_or_split(df, df_raw, 锅炉高压汽包压力_cols, "锅炉高压汽包压力", cv_thr=CV_P)

高压汽包上壁温_cols = [
    "#1炉高压汽包上壁温A",
    "#1炉高压汽包上壁温B",
    "#1炉高压汽包上壁温C",
]
add_group_mean_or_split(df, df_raw, 高压汽包上壁温_cols, "高压汽包上壁温", cv_thr=CV_T)

高压汽包下壁温_cols = [
    "#1炉高压汽包下壁温A",
    "#1炉高压汽包下壁温B",
    "#1炉高压汽包下壁温C",
]
add_group_mean_or_split(df, df_raw, 高压汽包下壁温_cols, "高压汽包下壁温", cv_thr=CV_T)

# =========================
# 减温器/主蒸汽（混合：单列+多列）
# =========================
for k_src, k_dst in [
    ("#1炉高压过热蒸汽减温水流量", "高压过热蒸汽减温水流量"),
    ("#1炉高压过热汽减温器入口蒸汽温度", "高压过热汽减温器入口蒸汽温度"),
    ("#1炉高压过热蒸汽减温器出口压力", "高压过热蒸汽减温器出口压力"),
    ("#1炉高压过热汽减温器出口疏水温度", "高压过热汽减温器出口疏水温度"),
    ("#1炉高压主蒸汽流量", "高压主蒸汽流量"),
]:
    if k_src in df_raw.columns:
        df[k_dst] = pd.to_numeric(df_raw[k_src], errors="coerce")
    else:
        print(f"[WARN] 缺列: {k_src}")

高压过热汽减温器出口蒸汽温度_cols = [
    "#1炉高压过热汽减温器出口蒸汽温度A",
    "#1炉高压过热汽减温器出口蒸汽温度B",
    "#1炉高过减温器出口蒸汽温度",
]
add_group_mean_or_split(df, df_raw, 高压过热汽减温器出口蒸汽温度_cols, "高压过热汽减温器出口蒸汽温度", cv_thr=CV_T)

高压主汽压力_cols = [
    "#1炉高压主蒸汽压力A",
    "#1炉高压主蒸汽压力B",
    "#1炉高压主蒸汽压力C",
    "#1锅炉高压主汽压力",
]
add_group_mean_or_split(df, df_raw, 高压主汽压力_cols, "高压主汽压力", cv_thr=CV_P)

高压主蒸汽温度_cols = [
    "#1炉高压主蒸汽温度A",
    "#1炉高压主蒸汽温度B",
    "#1炉高压主蒸汽温度C",
    "#1锅炉高压主汽温度",
]
add_group_mean_or_split(df, df_raw, 高压主蒸汽温度_cols, "高压主蒸汽温度", cv_thr=CV_T)


# =========================
# 导出
# =========================
out_path = BASE_DIR / Path(r"data_processing\for_hrsg\outcome\jyrd_hrgs_#1.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(out_path, encoding="utf-8-sig", index=False)
print(f"[OK] saved to: {out_path}")