from iapws import IAPWS97

class Hrsg():
    def __init__(self, T0, name):
        self.T0 = T0
        self.name = name
    
    def asses_eta_HRSG(self, gas_p_in, gas_T_in, gas_p_out, gas_T_out, 
                       steam_p_in, steam_T_in, steam_p_out, steam_T_out,
                       gas_m_dot, steam_m_dot):
        '''
        基于给定的进出口参数
        计算热回收蒸汽发生器的热效率
        '''
        
        # 获取烟气进出口状态
        gas_in = IAPWS97(P=gas_p_in, T=gas_T_in)
        gas_out = IAPWS97(P=gas_p_out, T=gas_T_out)
        
        # 获取蒸汽进出口状态
        steam_in = IAPWS97(P=steam_p_in, T=steam_T_in)
        steam_out = IAPWS97(P=steam_p_out, T=steam_T_out)
        
        # 计算烟气的焓变化
        delta_h_gas = gas_in.h - gas_out.h  # kJ/kg
        
        # 计算蒸汽的焓变化
        delta_h_steam = steam_out.h - steam_in.h  # kJ/kg
        
        # 计算热回收效率
        eta_HRSG = (delta_h_steam * steam_m_dot) / (delta_h_gas * gas_m_dot)
        
        return eta_HRSG
    
    def predict_steam_T_out(self, gas_p_in, gas_T_in, gas_p_out, gas_T_out,
                            steam_p_in, steam_T_in, 
                            steam_m_dot, eta_HRSG, gas_m_dot):
        '''
        基于给定的进出口参数和热回收效率
        预测热回收蒸汽发生器的蒸汽出口温度
        '''
        
        # 获取烟气进出口状态
        gas_in = IAPWS97(P=gas_p_in, T=gas_T_in)
        gas_out = IAPWS97(P=gas_p_out, T=gas_T_out)
        
        # 获取蒸汽进口状态
        steam_in = IAPWS97(P=steam_p_in, T=steam_T_in)
        
        # 计算烟气的焓变化
        delta_h_gas = gas_in.h - gas_out.h  # kJ/kg
        
        # 计算蒸汽的焓变化
        delta_h_steam = (delta_h_gas * gas_m_dot * eta_HRSG) / steam_m_dot  # kJ/kg
        
        # 预测蒸汽出口焓
        h_steam_out = steam_in.h + delta_h_steam
        
        # 预测蒸汽出口状态
        steam_out = IAPWS97(P=steam_p_in, h=h_steam_out)
        
        return steam_out.T, steam_out.p