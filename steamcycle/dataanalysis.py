import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


p = '主蒸汽压力平均绝压@jyrd(MPa)'
T = '主蒸汽温度平均(K)@jyrd_10min'
m_dot = '主蒸汽流量(kg-s)@jyrd_10min'
eff = '高压缸等熵效率(%)@jyrd_10min'

# p = '再热蒸汽绝对压力@jyrd(MPa)'
# T = '再热主温度平均(K)@jyrd_10min'
# m_dot = '热再热蒸汽流量(kg-s)@jyrd_10min'
# eff = '中压缸抽汽前透平级等熵效率(%)@jyrd_10min'

# p = '中压排汽压力(MPa)@jyrd_10min'
# T = '低压缸进汽温度(K)@jyrd_10min'
# m_dot = '中压缸抽后流量(kg-s)@jyrd_10min'
# eff = '低压缸等熵效率(%)@jyrd_10min'

# df = pd.read_pickle("data/steamcycle.pickle")
df = pd.read_pickle("data/steamcycle_all.pickle")



df = df.sample(frac=0.1, random_state=42)

print(df.columns)

# 过滤异常值
T_mean = df[T].mean()
T_std = df[T].std()
df = df[(df[T] > T_mean - 1.5 * T_std) & (df[T] < T_mean + 1.5 * T_std)]

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue']

x = df[T]
y = df[p]
z = df[eff]
clusters = df['cluster']

ax.scatter(x, y, z, s=5, c=[colors[label] for label in clusters])

ax.set_xlabel(T)
ax.set_ylabel(p)
ax.set_zlabel(eff)

plt.show()
