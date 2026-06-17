import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

from iapws import IAPWS97

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_analysis.working_fluid.fluid import FlowState


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _invalid_state(cls, *, P=None, T=None, m_dot=None, name="", ref=None, x=None):
    return cls(name=name, T=T, P=P, m_dot=m_dot, h=None, s=None, x=x, ref=ref)


@dataclass
class ReferenceEnv:
    T0: float
    P0: float
    h0: float
    s0: float


def create_water_reference_env(T0=298.15, P0=101325.0):
    w0 = IAPWS97(P=P0 / 1e6, T=T0)
    h0 = w0.h * 1000.0
    s0 = w0.s * 1000.0

    return ReferenceEnv(
        T0=T0,
        P0=P0,
        h0=h0,
        s0=s0,
    )


def create_reference_env(T0=298.15, P0=101325.0):
    return create_water_reference_env(T0=T0, P0=P0)


@dataclass
class WaterSteamState(FlowState):
    x: Optional[float] = None

    @classmethod
    def from_PT(cls, P, T, m_dot=None, name="", ref=None):
        if _is_missing(P) or _is_missing(T):
            return _invalid_state(cls, P=P, T=T, m_dot=m_dot, name=name, ref=ref)
        try:
            w = IAPWS97(P=P / 1e6, T=T)
        except Exception:
            return _invalid_state(cls, P=P, T=T, m_dot=m_dot, name=name, ref=ref)
        if _is_missing(w.h) or _is_missing(w.s):
            return _invalid_state(cls, P=P, T=T, m_dot=m_dot, name=name, ref=ref, x=getattr(w, "x", None))
        return cls(
            name=name,
            T=T,
            P=P,
            m_dot=m_dot,
            h=w.h * 1000.0,
            s=w.s * 1000.0,
            x=getattr(w, "x", None),
            ref=ref,
        )

    @classmethod
    def from_Px(cls, P, x, m_dot=None, name="", ref=None):
        if _is_missing(P) or _is_missing(x):
            return _invalid_state(cls, P=P, m_dot=m_dot, name=name, ref=ref, x=x)
        try:
            w = IAPWS97(P=P / 1e6, x=x)
        except Exception:
            return _invalid_state(cls, P=P, m_dot=m_dot, name=name, ref=ref, x=x)
        if _is_missing(w.h) or _is_missing(w.s):
            return _invalid_state(cls, P=P, T=getattr(w, "T", None), m_dot=m_dot, name=name, ref=ref, x=x)
        return cls(
            name=name,
            T=w.T,
            P=P,
            m_dot=m_dot,
            h=w.h * 1000.0,
            s=w.s * 1000.0,
            x=getattr(w, "x", x),
            ref=ref,
        )

    @classmethod
    def from_Ps(cls, P, s, m_dot=None, name="", ref=None):
        if _is_missing(P) or _is_missing(s):
            return _invalid_state(cls, P=P, m_dot=m_dot, name=name, ref=ref)
        try:
            w = IAPWS97(P=P / 1e6, s=s / 1000.0)
        except Exception:
            return _invalid_state(cls, P=P, m_dot=m_dot, name=name, ref=ref)
        if _is_missing(w.h) or _is_missing(w.s):
            return _invalid_state(cls, P=P, T=getattr(w, "T", None), m_dot=m_dot, name=name, ref=ref, x=getattr(w, "x", None))
        return cls(
            name=name,
            T=w.T,
            P=P,
            m_dot=m_dot,
            h=w.h * 1000.0,
            s=w.s * 1000.0,
            x=getattr(w, "x", None),
            ref=ref,
        )

    @classmethod
    def from_Ph(cls, P, h, m_dot=None, name="", ref=None):
        if _is_missing(P) or _is_missing(h):
            return _invalid_state(cls, P=P, m_dot=m_dot, name=name, ref=ref)
        try:
            w = IAPWS97(P=P / 1e6, h=h / 1000.0)
        except Exception:
            return _invalid_state(cls, P=P, m_dot=m_dot, name=name, ref=ref)
        if _is_missing(w.h) or _is_missing(w.s):
            return _invalid_state(cls, P=P, T=getattr(w, "T", None), m_dot=m_dot, name=name, ref=ref, x=getattr(w, "x", None))
        return cls(
            name=name,
            T=w.T,
            P=P,
            m_dot=m_dot,
            h=w.h * 1000.0,
            s=w.s * 1000.0,
            x=getattr(w, "x", None),
            ref=ref,
        )

    @cached_property
    def exergy(self):
        if self.h is None or self.s is None or self.ref is None:
            return None
        return (self.h - self.ref.h0) - self.ref.T0 * (self.s - self.ref.s0)
