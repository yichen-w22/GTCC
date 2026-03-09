from CoolProp.CoolProp import PropsSI

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

T = 810  # K
P = 6.5e6  # Pa
print(cp_flue_gas(T, P))