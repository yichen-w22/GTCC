import pandas as pd
from iapws import IAPWS97

class Steamturbine():
    def __init__(self, T0, mech_eff, name):
        self.T0 = T0
        self.mech_eff = mech_eff
        self.name = name
    
    def asses_eta_W_out(self, p_in, T_in, p_out, T_out, m_dot):
        '''
        基于给定的进出口参数
        计算蒸汽涡轮的等熵效率和输出功率
        '''
        # 进口、出口状态
        st_in = IAPWS97(P=p_in, T=T_in)
        st_out = IAPWS97(P=p_out, T=T_out)

        h_in = st_in.h   # kJ/kg
        s_in = st_in.s   # kJ/kgK
        h_out = st_out.h
        s_out = st_out.s

        # 等熵出口状态
        st_out_s = IAPWS97(P=p_out, s=s_in)
        h_out_s = st_out_s.h

        # 等熵做功与实际做功
        w_is = h_in - h_out_s
        w_actual = h_in - h_out

        # 功率（kW）
        W_out = w_actual * m_dot * self.mech_eff

        # 等熵效率
        eta_isentropic = w_actual / w_is

        # （可选）火用损失
        exergy_loss = self.T0 * (s_out - s_in)
        eta_exergy = (w_is - exergy_loss) / w_is

        return eta_isentropic, W_out
        
        # eta_list = []
        # W_list = []

        # # 使用 Series 的索引进行遍历
        # for idx in p_in.index:
        #     # 提取当前索引对应的工况
        #     p_in_i = float(p_in.loc[idx])
        #     T_in_i = float(T_in.loc[idx])
        #     p_out_i = float(p_out.loc[idx])
        #     T_out_i = float(T_out.loc[idx])
        #     m_dot_i = float(m_dot.loc[idx])

        #     # 获取状态
        #     st_in = IAPWS97(P=p_in_i, T=T_in_i)
        #     st_out = IAPWS97(P=p_out_i, T=T_out_i)

        #     h_in = st_in.h   # kJ/kg
        #     s_in = st_in.s

        #     h_out = st_out.h
        #     s_out = st_out.s

        #     # 等熵出口状态
        #     st_out_s = IAPWS97(P=p_out_i, s=s_in)
        #     h_out_s = st_out_s.h

        #     # 等熵做功与实际做功
        #     w_is = h_in - h_out_s
        #     w_actual = h_in - h_out

        #     # 输出功率（kW）
        #     W_out = w_actual * m_dot_i * self.mech_eff

        #     # 等熵效率
        #     eta_is = w_actual / w_is

        #     # 加入列表
        #     eta_list.append(eta_is)
        #     W_list.append(W_out)

        # # 构造 Series 并保留原 index
        # eta_series = pd.Series(eta_list, index=p_in.index, name=f"{self.name}_eta_is")
        # W_series = pd.Series(W_list, index=p_in.index, name=f"{self.name}_W_out")

        # return eta_series, W_series
    
    # def predict_W_out(self, p_in, T_in, p_out, eta_isentropic, m_dot):
    #     '''
    #     基于给定的进出口参数和等熵效率
    #     预测蒸汽涡轮的输出功率
    #     '''
        
    #     # 获取进出口状态
    #     st_in = IAPWS97(P=p_in, T=T_in)
        
    #     # 获取进出口的焓熵
    #     h_in = st_in.h   # kJ/kg
    #     s_in = st_in.s   # kJ/kgK

    #     # 等熵条件下的出口状态
    #     st_out_s = IAPWS97(P=p_out, s=s_in)
    #     h_out_s = st_out_s.h
        
    #     # 计算等熵做功
    #     w_is = h_in - h_out_s
        
    #     # 计算实际做功
    #     w_actual = w_is * eta_isentropic
    #     W_out = w_actual * m_dot * self.mech_eff
        
    #     return W_out
    
    
    
# 示例参数
if __name__ == "__main__":
    p_in = 9.736
    T_in = 838.6
    p_out = 2.361
    T_out = 649.6
    m_dot = 133.48

    turbine = Steamturbine(T0=298.15, mech_eff=1, name="示例涡轮")
    result = turbine.asses_eta_W_out(p_in, T_in, p_out, T_out, m_dot)
        
    print(result)