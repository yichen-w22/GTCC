class StatePoint():
    def __init__(self, p=None, T=None, m_dot=None, name=""):
        self.name = name
        self.p = p        # 压力 [MPa]
        self.T = T        # 温度 [K]
        self.m_dot = m_dot # 质量流量 [kg/s]