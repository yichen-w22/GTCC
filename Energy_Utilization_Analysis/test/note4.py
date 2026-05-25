import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Energy_Utilization_Analysis" / "data_precessing" / "continuous_data_10min.csv"
OUT_PATH = BASE_DIR / "residual_clustered2.csv"

X_COL = "燃机出力_2"
Y_COL = "烟气质量流量_2"

N_CLUSTERS = 2
POLY_DEGREE = 3


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[X_COL, Y_COL]).copy()

    x = df[[X_COL]].to_numpy()
    y = df[Y_COL].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=POLY_DEGREE),
        Ridge(alpha=1.0),
    )

    model.fit(x, y)

    df["fit_y"] = model.predict(x)
    df["residual"] = df[Y_COL] - df["fit_y"]

    residual_scaled = StandardScaler().fit_transform(df[["residual"]])

    raw_labels = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=0,
        n_init=50,
    ).fit_predict(residual_scaled)

    df["raw_cluster"] = raw_labels

    # 让标签稳定：残差较小的一类为 0，残差较大的一类为 1
    order = (
        df.groupby("raw_cluster")["residual"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    label_map = {raw_label: label for label, raw_label in enumerate(order)}
    df["cluster"] = df["raw_cluster"].map(label_map).astype(int)

    df = df.drop(columns=["raw_cluster"])

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"有效数据量: {len(df)}")
    print("各类数量:")
    print(df["cluster"].value_counts().sort_index())
    print()
    print(f"已输出: {OUT_PATH}")


if __name__ == "__main__":
    main()