import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Energy_Utilization_Analysis.energy_analysis.working_fluid.streams import build_streams_from_row, build_gases_from_row


df = pd.read_csv(r'C:\MyFolder\Projects\GTCC\Test\Energy_Utilization_Analysis\data_precessing\continuous_data_10min.csv')
gases = build_gases_from_row(df, idx=200)

print(gases["2号余热锅炉出口烟气"].exergy )



