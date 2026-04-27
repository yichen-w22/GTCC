from dataclasses import dataclass
from .converter import EnergyConverter


@dataclass
class GasWaterHeatExchangerResult:
    gas_heat_release: float | None
    water_heat_absorption: float | None
    heat_loss: float | None
    heat_balance_ratio: float | None
    heat_effectiveness: float | None
    gas_exergy_release: float | None
    water_exergy_absorption: float | None
    exergy_loss: float | None
    exergy_balance_ratio: float | None
    exergy_effectiveness: float | None
    mass_balance: float | None
    energy_balance: float | None
    exergy_balance: float | None


@dataclass
class GasWaterHeatExchanger(EnergyConverter):
    def gas_heat_release(self):
        if self.inlets[0].energy_flow is None or self.outlets[0].energy_flow is None:
            return None
        return self.inlets[0].energy_flow - self.outlets[0].energy_flow

    def water_heat_absorption(self):
        q = 0.0
        for s_in, s_out in zip(self.inlets[1:], self.outlets[1:]):
            if s_in.energy_flow is None or s_out.energy_flow is None:
                return None
            q += s_out.energy_flow - s_in.energy_flow
        return q

    def heat_loss(self):
        gas_heat_release = self.gas_heat_release()
        water_heat_absorption = self.water_heat_absorption()
        if gas_heat_release is None or water_heat_absorption is None:
            return None
        return gas_heat_release - water_heat_absorption
    
    # 吸热量占据放热量的比例
    def heat_balance_ratio(self):
        gas_heat_release = self.gas_heat_release()
        water_heat_absorption = self.water_heat_absorption()
        if gas_heat_release in (None, 0) or water_heat_absorption is None:
            return None
        return water_heat_absorption / gas_heat_release
    
    # 吸热量占最大吸热量的比例
    def heat_effectiveness(self):
        max_absorption = self.gas_heat_release()
        water_heat_absorption = self.water_heat_absorption()
        if max_absorption in (None, 0) or water_heat_absorption is None:
            return None
        return water_heat_absorption / max_absorption

    def gas_exergy_release(self):
        if self.inlets[0].exergy_flow is None or self.outlets[0].exergy_flow is None:
            return None
        return self.inlets[0].exergy_flow - self.outlets[0].exergy_flow

    def water_exergy_absorption(self):
        ex = 0.0
        for s_in, s_out in zip(self.inlets[1:], self.outlets[1:]):
            if s_in.exergy_flow is None or s_out.exergy_flow is None:
                return None
            ex += s_out.exergy_flow - s_in.exergy_flow
        return ex

    # 吸火用占据放火用的比例
    def exergy_balance_ratio(self):
        gas_exergy_release = self.gas_exergy_release()
        water_exergy_absorption = self.water_exergy_absorption()
        if gas_exergy_release in (None, 0) or water_exergy_absorption is None:
            return None
        return water_exergy_absorption / gas_exergy_release
    
    # 吸火用占最大吸火用的比例
    def exergy_effectiveness(self):
        max_absorption = self.gas_exergy_release()
        water_exergy_absorption = self.water_exergy_absorption()
        if max_absorption in (None, 0) or water_exergy_absorption is None:
            return None
        return water_exergy_absorption / max_absorption

    def solve(self) -> GasWaterHeatExchangerResult:
        return GasWaterHeatExchangerResult(
            gas_heat_release=self.gas_heat_release(),
            water_heat_absorption=self.water_heat_absorption(),
            heat_loss=self.heat_loss(),
            heat_balance_ratio=self.heat_balance_ratio(),
            heat_effectiveness=self.heat_effectiveness(),
            gas_exergy_release=self.gas_exergy_release(),
            water_exergy_absorption=self.water_exergy_absorption(),
            exergy_loss=self.exergy_balance(),
            exergy_balance_ratio=self.exergy_balance_ratio(),
            exergy_effectiveness=self.exergy_effectiveness(),
            mass_balance=self.mass_balance(),
            energy_balance=self.energy_balance(),
            exergy_balance=self.exergy_balance(),
        )
