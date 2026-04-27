from dataclasses import dataclass


@dataclass
class GTModelConfig:
    compressor_bleeding_mass_fraction: float = 0.08
    compressor_bleeding_pressure_fraction: float = 0.8
    compressor_bleeding_energy_fraction: float = 0.8
    total_pressure_recovery: float = 0.95
    combustion_efficiency: float = 0.99
