import matplotlib.pyplot as plt
import pandas as pd


def plot_cluster_scatter(df, x_param, y_param, save_path=None):
    """
    x_param, y_param 可以是列名，也可以是列序号 idx。
    如果 x_param="idx"，则横坐标使用 DataFrame 的 index。
    """

    if x_param == "idx":
        x_data = df.index
        x_label = "idx"
    elif isinstance(x_param, int):
        x_label = df.columns[x_param]
        x_data = df[x_label]
    else:
        x_label = x_param
        x_data = df[x_label]

    if y_param == "idx":
        y_data = df.index
        y_label = "idx"
    elif isinstance(y_param, int):
        y_label = df.columns[y_param]
        y_data = df[y_label]
    else:
        y_label = y_param
        y_data = df[y_label]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(8, 6))

    for label in sorted(df["cluster"].unique()):
        mask = df["cluster"] == label
        plt.scatter(
            x_data[mask],
            y_data[mask],
            s=10,
            alpha=0.45,
            label=f"cluster {label}",
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160)

    plt.show()

df = pd.read_csv(r"C:\MyFolder\Projects\GTCC\Test\residual_clustered2.csv")

plot_cluster_scatter(df, x_param="燃机出力_2", y_param="烟气质量流量_2")