import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
from scipy.signal import butter, filtfilt
from iapws import IAPWS97

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

water = IAPWS97(T=650, P=6.5)

print(water.cp)   # kJ/(kg·K)

water = IAPWS97(T=798, P=6.5)

print(water.cp)   # kJ/(kg·K)