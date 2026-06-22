from energy_analysis.working_fluid.gas import (
    GasComposition,
    GasState,
    build_air_composition,
    build_flue_gas_composition,
    create_gas_reference_env,
)
from energy_analysis.working_fluid.steam_water import WaterSteamState, create_water_reference_env


MIN_WATER_REF_T = 273.15
GT_POWER_STOP_THRESHOLD = 10_000_000.0


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _ambient_temperature(row, unit):
    return row[f"环境温度_{unit}"]


def _valid_state(state):
    return (
        not any(_is_missing(value) for value in (state.P, state.m_dot, state.h))
        and state.m_dot > 0
    )


def _is_unit_running(row, unit):
    power = row.get(f"燃机出力_{unit}")
    return not _is_missing(power) and power > GT_POWER_STOP_THRESHOLD


def build_streams_from_row(df, idx):

    LP_Turbine_Exhaust_Steam_Quality = 0.97

    """
    从 df 第 idx 行构造全场流股字典。
    返回:
        streams: dict[str, FlowState]
    """
    row = df.iloc[idx]
    streams = {}

    unit_running = {
        1: _is_unit_running(row, 1),
        2: _is_unit_running(row, 2),
    }

    def mix_streams(source_keys, output_key):
        running_sources = [
            streams[key]
            for unit, key in source_keys
            if unit_running[unit]
        ]
        if not running_sources:
            return WaterSteamState(name=output_key)
        if any(not _valid_state(source) for source in running_sources):
            return WaterSteamState(name=output_key)
        sources = running_sources
        m_mix = sum(source.m_dot for source in sources)
        h_mix = sum(source.m_dot * source.h for source in sources) / m_mix
        P_mix = min(source.P for source in sources)
        return WaterSteamState.from_Ph(P=P_mix, h=h_mix, m_dot=m_mix, name=output_key)

    # =============================
    # 0. 动态参考环境（关键！）
    # =============================
    water_ref_T0 = max(_ambient_temperature(row, 1), MIN_WATER_REF_T)
    water_ref = create_water_reference_env(
        T0=water_ref_T0,   # IAPWS97 does not support water below 273.15 K
        P0=101325.0
    )

    # =========================================================
    # 2. 1号余热锅炉汽水侧
    # =========================================================
    streams["1号炉低压汽包"] = WaterSteamState.from_Px(
        P=row["低压汽包压力_1"],
        x=0,
        m_dot=row["低压给水流量_1"],
        name="1号炉低压汽包"
    )

    # 高压
    streams["1号炉低压汽包高压给水泵后"] = WaterSteamState.from_PT(
        P=row["高压省煤器进水压力_1"],
        T=streams["1号炉低压汽包"].T,
        m_dot=row["高压给水流量_1"],
        name="1号炉低压汽包高压给水泵后"
    )

    streams["1号炉高压主蒸汽"] = WaterSteamState.from_PT(
        P=row["高压主蒸汽压力_1"],
        T=row["高压主蒸汽温度_1"],
        m_dot=row["高压主蒸汽流量_1"],
        name="1号炉高压主蒸汽"
    )

    # 中压
    streams["1号炉低压汽包中压给水泵后"] = WaterSteamState.from_PT(
        P=row["中压省煤器进水压力_1"],
        T=streams["1号炉低压汽包"].T,
        m_dot=row["中压给水流量_1"],
        name="1号炉低压汽包中压给水泵后"
    )

    # 低压省煤器
    streams["1号炉低压省煤器入口"] = WaterSteamState.from_PT(
        P=row["凝结水泵出口母管压力"],
        T=row["冷凝水温度_1"],
        m_dot=row["低压给水流量_1"],
        name="1号炉低压省煤器入口"
    )

    streams["1号炉低压省煤器出口"] = WaterSteamState.from_PT(
        P=row["凝结水泵出口母管压力"],
        T=row["低压省煤器出水温度_1"],
        m_dot=row["低压给水流量_1"],
        name="1号炉低压省煤器出口"
    )

    # 低压汽水
    streams["1号炉低压汽包给水调节阀出口"] = WaterSteamState.from_Ph(
        P=row["低压汽包压力_1"],
        h=streams["1号炉低压省煤器出口"].h,
        m_dot=row["低压给水流量_1"],
        name="1号炉低压汽包给水调节阀出口"
    )

    streams["1号炉低压汽包至低压主蒸汽"] = WaterSteamState.from_Px(
        P=row["低压汽包压力_1"],
        x=0,
        m_dot=row["低压主蒸汽流量_1"],
        name="1号炉低压汽包至低压主蒸汽"
    )

    streams["1号炉低压主蒸汽"] = WaterSteamState.from_PT(
        P=row["低压主蒸汽压力_1"],
        T=row["低压主蒸汽温度_1"],
        m_dot=row["低压主蒸汽流量_1"],
        name="1号炉低压主蒸汽"
    )

    # 再热
    streams["1号炉高压缸排汽"] = WaterSteamState.from_PT(
        P=row["高压缸排汽压力"],
        T=row["高压缸排汽温度"],
        m_dot=row["高压缸排汽进入锅炉_1"],
        name="1号炉高压缸排汽"
    )

    streams["1号炉热再热出口"] = WaterSteamState.from_PT(
        P=row["热再热蒸汽压力_1"],
        T=row["热再热蒸汽温度_1"],
        m_dot=row["热再热蒸汽流量_1"],
        name="1号炉热再热出口"
    )

    # =========================================================
    # 3. 2号余热锅炉汽水侧
    # =========================================================
    streams["2号炉低压汽包"] = WaterSteamState.from_Px(
        P=row["低压汽包压力_2"],
        x=0,
        m_dot=row["低压给水流量_2"],
        name="2号炉低压汽包"
    )
    
    # 高压
    streams["2号炉低压汽包高压给水泵后"] = WaterSteamState.from_PT(
        P=row["高压省煤器进水压力_2"],
        T=streams["2号炉低压汽包"].T,
        m_dot=row["高压给水流量_2"],
        name="2号炉低压汽包高压给水泵后"
    )

    streams["2号炉高压主蒸汽"] = WaterSteamState.from_PT(
        P=row["高压主蒸汽压力_2"],
        T=row["高压主蒸汽温度_2"],
        m_dot=row["高压主蒸汽流量_2"],
        name="2号炉高压主蒸汽"
    )

    # 中压
    streams["2号炉低压汽包中压给水泵后"] = WaterSteamState.from_PT(
        P=row["中压省煤器进水压力_2"],
        T=streams["2号炉低压汽包"].T,
        m_dot=row["中压给水流量_2"],
        name="2号炉低压汽包中压给水泵后"
    )

    # 低压省煤器
    streams["2号炉低压省煤器入口"] = WaterSteamState.from_PT(
        P=row["凝结水泵出口母管压力"],
        T=row["冷凝水温度_2"],
        m_dot=row["低压给水流量_2"],
        name="2号炉低压省煤器入口"
    )

    streams["2号炉低压省煤器出口"] = WaterSteamState.from_PT(
        P=row["凝结水泵出口母管压力"],
        T=row["低压省煤器出水温度_2"],
        m_dot=row["低压给水流量_2"],
        name="2号炉低压省煤器出口"
    )
  

    # 低压汽水
    streams["2号炉低压汽包给水调节阀出口"] = WaterSteamState.from_Ph(
        P=row["低压汽包压力_2"],
        h=streams["2号炉低压省煤器出口"].h,
        m_dot=row["低压给水流量_2"],
        name="2号炉低压汽包给水调节阀出口"
    )

    streams["2号炉低压汽包至低压主蒸汽"] =  WaterSteamState.from_Px(
        P=row["低压汽包压力_2"],
        x=0,
        m_dot=row["低压主蒸汽流量_2"],
        name="2号炉低压汽包至低压主蒸汽"
    )

    streams["2号炉低压主蒸汽"] = WaterSteamState.from_PT(
        P=row["低压主蒸汽压力_2"],
        T=row["低压主蒸汽温度_2"],
        m_dot=row["低压主蒸汽流量_2"],
        name="2号炉低压主蒸汽"
    )

    # 再热
    streams["2号炉高压缸排汽"] = WaterSteamState.from_PT(
        P=row["高压缸排汽压力"],
        T=row["高压缸排汽温度"],
        m_dot=row["高压缸排汽进入锅炉_2"],
        name="2号炉高压缸排汽"
    )

    streams["2号炉热再热出口"] = WaterSteamState.from_PT(
        P=row["热再热蒸汽压力_2"],
        T=row["热再热蒸汽温度_2"],
        m_dot=row["热再热蒸汽流量_2"],
        name="2号炉热再热出口"
    )


    # =========================================================
    # 4. 汽轮机
    # =========================================================
    streams["高压缸入口"] = mix_streams(
        [(1, "1号炉高压主蒸汽"), (2, "2号炉高压主蒸汽")],
        "高压缸入口",
    )
    m_mix = streams["高压缸入口"].m_dot

    streams["高压缸出口"] = WaterSteamState.from_PT(
        P=row["高压缸排汽压力"],
        T=row["高压缸排汽温度"],
        m_dot=m_mix,
        name="高压缸出口"
    )

    streams["中压缸入口"] = mix_streams(
        [(1, "1号炉热再热出口"), (2, "2号炉热再热出口")],
        "中压缸入口",
    )
    m_mix = streams["中压缸入口"].m_dot

    streams["中压缸出口"] = WaterSteamState.from_PT(
        P=row["中压缸排汽压力"],
        T=row["中压缸排汽温度"],
        m_dot=m_mix,
        name="中压缸出口"
    )

    running_lp_steam_sources = [
        streams[f"{unit}号炉低压主蒸汽"].m_dot
        for unit in (1, 2)
        if unit_running[unit]
    ]
    m_lp = (
        None
        if (
            _is_missing(m_mix)
            or _is_missing(row["热网抽汽流量"])
            or any(_is_missing(m_dot) for m_dot in running_lp_steam_sources)
        )
        else m_mix + sum(running_lp_steam_sources) - row["热网抽汽流量"]
    )
    streams["低压缸入口"] = WaterSteamState.from_PT(
        P=row["低压缸进汽压力"],
        T=row["低压缸进汽温度"],
        m_dot=m_lp,
        name="低压缸入口"
    )

    streams["低压缸出口"] = WaterSteamState.from_Px(
        P=row["凝汽器压力"],
        x = LP_Turbine_Exhaust_Steam_Quality,
        m_dot=m_lp,
        name="低压缸出口"
    )

    streams["热网抽汽"] = WaterSteamState.from_PT(
        P=row["中压缸排汽压力"],
        T=row["中压缸排汽温度"],
        m_dot=row["热网抽汽流量"],
        name="热网抽汽"
    )

    for s in streams.values():
        s.ref = water_ref    
    
    return streams


def build_fuel_composition_from_row(df, idx):
    row = df.iloc[idx]
    compositions = {}

    compositions["H2"] = row["H2"]
    compositions["N2"] = row["N2"]
    compositions["CO2"] = row["CO2"]
    compositions["CH4"] = row["CH4"]
    compositions["CO"] = row["CO"]
    compositions["O2+Ar"] = row["O2+Ar"]
    compositions["C2H6"] = row["C2H6"]
    compositions["C3H8"] = row["C3H8"]
    compositions["iC4H10"] = row["iC4H10"]
    compositions["nC4H10"] = row["nC4H10"]
    compositions["iC5H12"] = row["iC5H12"]
    compositions["nC5H12"] = row["nC5H12"]

    return compositions

def _row_value(row, *keys, default=None):
    for key in keys:
        if key in row.index:
            return row[key]
    if default is not None:
        return default
    raise KeyError(keys[0])

def build_gases_from_row(df, idx):
    row = df.iloc[idx]
    gases = {}
    unit_running = {
        1: _is_unit_running(row, 1),
        2: _is_unit_running(row, 2),
    }
    fuel_composition = GasComposition.from_dict(build_fuel_composition_from_row(df, idx))

    def unit_gas_names(unit):
        return {
            f"{unit}号炉燃料": f"{unit}号炉燃料",
            f"{unit}号燃机入口空气": f"{unit}号燃机入口空气",
            f"{unit}号燃机压气机出口": f"{unit}号燃机压气机出口",
            f"{unit}号余热锅炉入口烟气": f"{unit}号余热锅炉进口烟气",
            f"{unit}号余热锅炉出口烟气": f"{unit}号余热锅炉出口烟气",
        }

    def add_empty_unit_gases(unit):
        for key, name in unit_gas_names(unit).items():
            gases[key] = GasState(name=name)

    def corrected_air_flow(unit):
        fuel_flow = row.get(f"燃料质量流量_{unit}")
        air_flow = row.get(f"燃机进口空气流量_{unit}")
        if _is_missing(fuel_flow) or _is_missing(air_flow) or fuel_flow == 0:
            return None
        if unit == 1:
            corr_coff = (
                1.6034403778
                + 1.1384305006e-01 * fuel_flow
                - 3.9913329318e-03 * air_flow
                + 2.7900005990e-03 * air_flow / fuel_flow
            )
        else:
            corr_coff = (
                -1.4147760372
                + 3.5387273740e-01 * fuel_flow
                - 1.0364906629e-02 * air_flow
                + 8.2414971724e-02 * air_flow / fuel_flow
            )
        return air_flow * corr_coff

    required_fields_by_unit = {
        unit: [
            f"环境温度_{unit}",
            f"压气机入口压力_{unit}",
            f"大气相对湿度_{unit}",
            f"燃料温度_{unit}",
            f"燃料压力_{unit}",
            f"燃料质量流量_{unit}",
            f"燃机进口空气流量_{unit}",
            f"压气机出口温度_{unit}",
            f"压气机出口压力_{unit}",
            f"进口烟温_{unit}",
            f"进口烟压_{unit}",
            f"排烟烟温_{unit}",
            f"排烟烟压_{unit}",
        ]
        for unit in (1, 2)
    }

    valid_ref_units = [
        unit
        for unit in (1, 2)
        if (
            unit_running[unit]
            and not _is_missing(row.get(f"环境温度_{unit}"))
            and not _is_missing(row.get(f"压气机入口压力_{unit}"))
        )
    ]
    gas_ref = None
    if valid_ref_units:
        gas_ref = create_gas_reference_env(
            T0=sum(row[f"环境温度_{unit}"] for unit in valid_ref_units) / len(valid_ref_units),
            P0=sum(row[f"压气机入口压力_{unit}"] for unit in valid_ref_units) / len(valid_ref_units),
        )

    def add_unit_gases(unit):
        if not unit_running[unit]:
            add_empty_unit_gases(unit)
            return
        if any(_is_missing(row.get(field)) for field in required_fields_by_unit[unit]):
            add_empty_unit_gases(unit)
            return

        air_flow = corrected_air_flow(unit)
        if _is_missing(air_flow):
            add_empty_unit_gases(unit)
            return

        try:
            air_composition = build_air_composition(
                T=row[f"环境温度_{unit}"],
                P=row[f"压气机入口压力_{unit}"],
                RH=row[f"大气相对湿度_{unit}"],
            )
            fuel_gas_composition = build_flue_gas_composition(
                fuel_composition,
                air_composition,
                m_dot_fuel=row[f"燃料质量流量_{unit}"],
                m_dot_air=air_flow,
            )
        except Exception:
            add_empty_unit_gases(unit)
            return

        flue_gas_flow = row[f"燃料质量流量_{unit}"] + air_flow
        gases[f"{unit}号炉燃料"] = GasState.from_TP(
            T=row[f"燃料温度_{unit}"],
            P=row[f"燃料压力_{unit}"],
            m_dot=row[f"燃料质量流量_{unit}"],
            composition=fuel_composition,
            name=f"{unit}号炉燃料",
            ref=gas_ref,
        )
        gases[f"{unit}号燃机入口空气"] = GasState.from_TP(
            T=row[f"环境温度_{unit}"],
            P=row[f"压气机入口压力_{unit}"],
            m_dot=air_flow,
            composition=air_composition,
            name=f"{unit}号燃机入口空气",
            ref=gas_ref,
        )
        gases[f"{unit}号燃机压气机出口"] = GasState.from_TP(
            T=row[f"压气机出口温度_{unit}"],
            P=row[f"压气机出口压力_{unit}"],
            m_dot=air_flow,
            composition=air_composition,
            name=f"{unit}号燃机压气机出口",
            ref=gas_ref,
        )
        gases[f"{unit}号余热锅炉入口烟气"] = GasState.from_TP(
            T=row[f"进口烟温_{unit}"],
            P=row[f"进口烟压_{unit}"],
            m_dot=flue_gas_flow,
            composition=fuel_gas_composition,
            name=f"{unit}号余热锅炉进口烟气",
            ref=gas_ref,
        )
        gases[f"{unit}号余热锅炉出口烟气"] = GasState.from_TP(
            T=row[f"排烟烟温_{unit}"],
            P=row[f"排烟烟压_{unit}"],
            m_dot=flue_gas_flow,
            composition=fuel_gas_composition,
            name=f"{unit}号余热锅炉出口烟气",
            ref=gas_ref,
        )

    add_unit_gases(1)
    add_unit_gases(2)

    return gases
