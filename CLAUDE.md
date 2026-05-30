# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GTCC (Gas Turbine Combined Cycle) energy utilization analysis system. Models the thermodynamic cycles of a combined cycle power plant with gas turbines (GT), heat recovery steam generators (HRSG), and steam turbines (ST). Includes a companion LaTeX thesis.

The codebase is in **Chinese research context** — commit messages, comments, and thesis are in Chinese.

## Environment Setup

```bash
conda env create -f environment.yml    # Python 3.10, CoolProp, IAPWS, PyTorch, etc.
```

Key dependencies: CoolProp (gas properties), IAPWS (water/steam via IAPWS-97), PyTorch (deep learning models), NumPy, SciPy, Pandas, Matplotlib.

## Running Tests & Analysis

```bash
# Unit tests (run from repo root)
python -m pytest Energy_Utilization_Analysis/test/test_gt.py
python -m pytest Energy_Utilization_Analysis/test/test_st.py

# Individual analysis scripts
python Energy_Utilization_Analysis/test/<script_name>.py

# Deep learning training pipeline
python dl/final.py
```

## Architecture

### Data Pipeline

```
Raw AVRO data → datareader_new/ (AVRO→pickle) → data_processing/ (averaging, dedup)
→ streams.py builders (create FlowState objects) → GT_model + ST_model → plant_model.py
→ analysis scripts (plots, efficiency studies)
```

### Core Package: `Energy_Utilization_Analysis/energy_analysis/`

**Working Fluid** (`working_fluid/`): Thermodynamic state classes.
- `FlowState` — base class (T, P, m_dot, h, s, exergy)
- `GasState` — multi-species gas via CoolProp. Factory methods: `from_TP()`, `from_Ph()`, `from_Ps()`
- `WaterSteamState` — water/steam via IAPWS-97
- `GasComposition` — mole fractions, LHV calculation
- `streams.py` — `build_streams_from_row()` and `build_gases_from_row()` create all states from a data row

**Gas Turbine** (`GT_model/`): Three-component cycle.
- Compressor → Combustion Chamber → Turbine
- `GTModel` orchestrates the cycle, calculates net power and thermal efficiency
- `GTModelConfig` dataclass holds defaults (15% bleeding, 95% pressure recovery, 99% combustion efficiency)

**Steam Turbine** (`ST_model/`): All components inherit from `EnergyConverter` base class.
- Components: Turbine, Pump, Condenser, HeatExchanger (HRSG), Mixer, ThrottleValve
- `plant.py` — `build_plant()` configures dual HRSG units and turbine stages (HP/IP/LP)

**Plant Model** (`PLANT_model.py/plant_model.py`): Top-level orchestration combining GT and ST models, calculates overall plant metrics.

### Supporting Modules

- `degradation/` — Steady-state detection (`steady_state_index()`), outlier removal, efficiency tracking
- `dl/` — Neural network models (LSTM, MLP, PINN, Transformer) in `final.py` unified pipeline
- `HRSG/` — Standalone HRSG superheater modeling
- `thesis/` — LaTeX thesis (`main.tex`, chapters 1-6, figures organized by chapter)

## Key Code Patterns

- **Lazy imports** via `__getattr__()` in package `__init__.py` files
- **`@cached_property`** for expensive thermodynamic calculations
- **`@dataclass`** for all state and component classes
- **Factory methods** (`from_TP`, `from_Ph`, `from_Ps`) instead of constructors for different property combinations
- **Dual-unit architecture** — everything supports Unit 1 and Unit 2 configurations
- **Dynamic reference environment** — reference state created per data row from actual ambient conditions (critical for exergy)

## Important Constants

- Reference: T_REF = 298.15 K, P_REF = 101325 Pa
- IAPWS-97 range: 273.15–863.15 K
- Plant thresholds: GT power stop > 1.0 MW, fuel flow > 5.0 kg/s, flue gas 100–1000 kg/s

## Thesis

LaTeX source in `thesis/`. Master file is `main.tex`, chapters in `chapters/`, figures in `figures/chapter{N}/`.