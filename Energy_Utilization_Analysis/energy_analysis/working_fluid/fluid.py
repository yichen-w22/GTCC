import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class FlowState:
    name: str = ""
    T: Optional[float] = None
    P: Optional[float] = None
    m_dot: Optional[float] = None
    h: Optional[float] = None
    s: Optional[float] = None
    ref: Optional[object] = None

    @property
    def energy_flow(self):
        if self.m_dot is None or self.h is None:
            return None
        return self.m_dot * self.h

    @property
    def exergy(self):
        raise NotImplementedError

    @property
    def exergy_flow(self):
        if self.m_dot is None or self.exergy is None:
            return None
        return self.m_dot * self.exergy




