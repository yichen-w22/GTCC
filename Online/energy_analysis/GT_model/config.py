from dataclasses import dataclass


@dataclass
class GTModelConfig:
    compressor_bleeding_mass_fraction: float = 0.15 
    compressor_bleeding_pressure_fraction: float = 0.9
    compressor_bleeding_energy_fraction: float = 0.9
    total_pressure_recovery: float = 0.95
    combustion_efficiency: float = 0.99
