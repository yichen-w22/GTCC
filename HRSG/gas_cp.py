from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP

def cp_flue_gas(T, P):
    """
    烟气定压比热
    T : K
    P : Pa
    return : J/(kg·K)
    """

    comp = {
        "Nitrogen": 0.75,
        "Oxygen": 0.12,
        "CarbonDioxide": 0.08,
        "Water": 0.05
    }

    cp = 0.0

    for gas, x in comp.items():
        cp_i = PropsSI("Cpmass", "T", T, "P", P, gas)
        cp += x * cp_i

    return cp

T = 75+273.15  # K
P = 0.101325e6  # Pa
print(cp_flue_gas(T, P))

def rho_flue_gas_ideal(T, P):
    """
    理想气体烟气密度
    T : K
    P : Pa
    return : kg/m3
    """

    comp = {
        "Nitrogen": 0.75,
        "Oxygen": 0.12,
        "CarbonDioxide": 0.08,
        "Water": 0.05
    }

    R_mix = 0.0

    for gas, x in comp.items():
        M = CP.PropsSI("M", gas)      # molar mass
        R_i = 8.314462618 / M         # specific gas constant
        R_mix += x * R_i

    rho = P / (R_mix * T)

    return rho

print(rho_flue_gas_ideal(T, P))