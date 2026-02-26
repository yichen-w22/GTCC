import pandas as pd
# from statepoint import StatePoint
import steamturbine


class StatePoint():
    def __init__(self, p=None, T=None, m_dot=None, name=""):
        self.name = name
        self.p = p        # 压力 [MPa]
        self.T = T        # 温度 [K]
        self.m_dot = m_dot # 质量流量 [kg/s]

if __name__ == "__main__":
    # 创建蒸汽涡轮对象
    hp = steamturbine.Steamturbine(T0=298.15, mech_eff=0.99, name="高压缸")
    ip = steamturbine.Steamturbine(T0=298.15, mech_eff=0.99, name="中压缸")
    lp = steamturbine.Steamturbine(T0=298.15, mech_eff=0.99, name="低压缸")
    
    df = pd.read_pickle("data/steamcycle.pickle").head(5)
    
    sp1 = StatePoint(p=df['主蒸汽压力平均绝压@jyrd(MPa)'], 
                        T=df['主蒸汽温度平均(K)@jyrd_10min'], 
                        m_dot=df['主蒸汽流量(kg-s)@jyrd_10min'],
                        name="高压缸进口")
    
    sp2 = StatePoint(p=df['高压排汽绝对压力@jyrd(MPa)'], 
                        T=df['高中压外缸高压排汽温度(K)@jyrd_10min'], 
                        m_dot=df['主蒸汽流量(kg-s)@jyrd_10min'],
                        name="高压缸出口")
    
    sp3 = StatePoint(p=df['再热蒸汽绝对压力@jyrd(MPa)'], 
                        T=df['再热主温度平均(K)@jyrd_10min'], 
                        m_dot=df['热再热蒸汽流量(kg-s)@jyrd_10min'],
                        name="中压缸入口")
    
    sp4 = StatePoint(p=df['一段抽汽压力(MPa)@jyrd_10min'], 
                        T=df['一段抽汽温度(K)@jyrd_10min'], 
                        m_dot=df['一抽流量(kg-s)'],
                        name="一抽入口")
    
    sp5 = StatePoint(p=df['一段抽汽压力(MPa)@jyrd_10min'], 
                        T=df['一段抽汽温度(K)@jyrd_10min'], 
                        m_dot=df['中压缸抽后流量(kg-s)@jyrd_10min'],
                        name="中压缸抽后")
    
    sp6 = StatePoint(p=df['中压排汽压力(MPa)@jyrd_10min'], 
                        T=df['高中压外缸中压排汽温度(K)@jyrd_10min'], 
                        m_dot=df['二抽流量(kg-s)'],
                        name="二抽入口")
    
    sp7 = StatePoint(p=df['中压排汽压力(MPa)@jyrd_10min'], 
                        T=df['低压缸进汽温度(K)@jyrd_10min'], 
                        m_dot=df['低压缸凝结水流量(kg-s)@jyrd_10min'],
                        name="低压缸入口")
    
    sp8 = StatePoint(p=df['低压缸排汽压力(MPa)@jyrd_10min'], 
                        T=df['低压缸排汽温度(K)@jyrd_10min'], 
                        m_dot=df['低压缸凝结水流量(kg-s)@jyrd_10min'],
                        name="低压缸出口")
    
    # 显式循环
    hp_eta_list = []
    hp_W_list = []
    
    ip_eta_list1 = []
    ip_W_list1 = []
    
    ip_eta_list2 = []
    ip_W_list2 = []
    
    lp_eta_list = []
    lp_W_list = []
    
    for idx in df.index:
        # 高压缸性能计算
        hp_eta_i, hp_W_i = hp.asses_eta_W_out(sp1.p.loc[idx], sp1.T.loc[idx], 
                                        sp2.p.loc[idx], sp2.T.loc[idx], 
                                        sp1.m_dot.loc[idx])
        
        hp_eta_list.append(hp_eta_i)
        hp_W_list.append(hp_W_i)
        
        # 中压缸抽前性能计算
        ip_eta_i1, ip_W_i1 = ip.asses_eta_W_out(sp3.p.loc[idx], sp3.T.loc[idx], 
                                        sp4.p.loc[idx], sp4.T.loc[idx], 
                                        sp3.m_dot.loc[idx])
        
        ip_eta_list1.append(ip_eta_i1)
        ip_W_list1.append(ip_W_i1)
        
        # 中压缸抽后性能计算
        ip_eta_i2, ip_W_i2 = ip.asses_eta_W_out(sp5.p.loc[idx], sp5.T.loc[idx], 
                                        sp6.p.loc[idx], sp6.T.loc[idx], 
                                        sp5.m_dot.loc[idx])
        
        ip_eta_list2.append(ip_eta_i2)
        ip_W_list2.append(ip_W_i2)
        
        # 低压缸性能计算
        lp_eta_i, lp_W_i = lp.asses_eta_W_out(sp7.p.loc[idx]*0.9, sp7.T.loc[idx], 
                                        sp8.p.loc[idx], sp8.T.loc[idx], 
                                        sp7.m_dot.loc[idx])
        
        lp_eta_list.append(lp_eta_i)
        lp_W_list.append(lp_W_i)


    # 转为 Series
    eta_hp = pd.Series(hp_eta_list, index=df.index, name="高压缸等熵效率")
    W_hp = pd.Series(hp_W_list, index=df.index, name="高压缸输出功率(kW)")
    
    eta_ip1 = pd.Series(ip_eta_list1, index=df.index, name="中压缸抽前等熵效率")
    W_ip1 = pd.Series(ip_W_list1, index=df.index, name="中压缸抽前输出功率(kW)")
    
    eta_ip2 = pd.Series(ip_eta_list2, index=df.index, name="中压缸抽后等熵效率")
    W_ip2 = pd.Series(ip_W_list2, index=df.index, name="中压缸抽后输出功率(kW)")
    
    eta_lp = pd.Series(lp_eta_list, index=df.index, name="低压缸等熵效率")
    W_lp = pd.Series(lp_W_list, index=df.index, name="低压缸输出功率(kW)")
    
    eta = (W_hp + W_ip1 + W_ip2 + W_lp) / (W_hp / eta_hp + W_ip1 / eta_ip1 + W_ip2 / eta_ip2 + W_lp / eta_lp)
    eta.name = "总等熵效率"
    
    W = W_hp + W_ip1 + W_ip2 + W_lp
    W.name = "总输出功率(kW)"

print(eta_hp.index, eta_hp.name, eta_hp)
print(W_hp.index, W_hp.name, W_hp)
print(eta_ip1.index, eta_ip1.name, eta_ip1)
print(W_ip1.index, W_ip1.name, W_ip1)
print(eta_ip2.index, eta_ip2.name, eta_ip2)
print(W_ip2.index, W_ip2.name, W_ip2)
print(eta_lp.index, eta_lp.name, eta_lp)
print(W_lp.index, W_lp.name, W_lp)
print(eta.index, eta.name, eta)
print(W.index, W.name, W)

# def print_by_index(index, data_dict):
#     """
#     按 index 输出所有传入的 Series 对应行的结果，data_dict 为字典形式：
#     {label: series}
#     """
#     print(f"Index: {index}")
#     for label, series in data_dict.items():
#         value = series.loc[index]
#         print(f"  {label}: {value}")
#     print("-" * 40)

# # 创建一个字典，映射标签到对应的 Series
# data_dict = {
#     "高压缸等熵效率": eta_hp,
#     "高压缸输出功率(kW)": W_hp,
#     "中压缸等熵效率（段1）": eta_ip1,
#     "中压缸输出功率(kW）（段1）": W_ip1,
#     "中压缸等熵效率（段2）": eta_ip2,
#     "中压缸输出功率(kW）（段2）": W_ip2,
#     "低压缸等熵效率": eta_lp,
#     "低压缸输出功率(kW)": W_lp,
#     "总等熵效率": eta,
#     "总输出功率(kW)": W,
# }

# # 输出前 5 个 index 的结果
# for idx in df.index[:]:
#     print_by_index(idx, data_dict)
    
