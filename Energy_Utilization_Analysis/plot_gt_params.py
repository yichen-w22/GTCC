import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.patch.set_facecolor('white')

tables_data = {
    "压气机": [
        ["等熵效率"],
        ["㶲效率"],
        ["等熵损失"],
        ["耗功"],
        ["比焓升"],
    ],
    "燃烧室": [
        ["燃料低位热值"],
        ["释放热量"],
        ["压力损失"],
    ],
    "透平": [
        ["等熵效率"],
        ["㶲效率"],
        ["等熵损失"],
        ["输出功"],
    ],
    "整机": [
        ["净出力"],
        ["燃料输入能量"],
        ["热效率"],
        ["发电功率份额"],
        ["压气机耗功份额"],
        ["排气能量份额"],
    ],
}

positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
header_color = "#4472C4"
row_colors = ["#D9E2F3", "#EDF2FA"]

for (title, rows), (r, c) in zip(tables_data.items(), positions):
    ax = axes[r][c]
    ax.set_axis_off()

    col_labels = ["性能参数"]
    cell_colors = [[row_colors[i % 2]] for i in range(len(rows))]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=[header_color],
        cellColours=cell_colors,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.6)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#B4C7E7")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_text_props(color="white", fontweight="bold", fontsize=12)
            cell.set_facecolor(header_color)
        else:
            cell.set_text_props(color="#333333")

    ax.set_title(title, fontsize=14, fontweight="bold", color="#2F5496", pad=12)

plt.tight_layout(pad=2.0)
plt.savefig(
    "Energy_Utilization_Analysis/GT_performance_params.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("saved → Energy_Utilization_Analysis/GT_performance_params.png")
