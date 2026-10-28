"""
Regime-Switching Simulation Framework (Section 8 Architecture).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class MarketRegime:
    name: str
    annual_drift: Dict[str, float]
    annual_volatility: Dict[str, float]
    correlation_matrix: Optional[np.ndarray] = None
    description: Optional[str] = None

@dataclass
class RegimeConfig:
    regimes: Dict[str, MarketRegime] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None
    initial_regime: str = "normal"
    enabled: bool = False

    @classmethod
    def create_default_template(cls, assets: List[str]) -> "RegimeConfig":
        regimes = {
            "bull": MarketRegime(name="bull", annual_drift={a: 0.18 for a in assets}, annual_volatility={a: 0.14 for a in assets}),
            "normal": MarketRegime(name="normal", annual_drift={a: 0.09 for a in assets}, annual_volatility={a: 0.18 for a in assets}),
            "bear": MarketRegime(name="bear", annual_drift={a: -0.12 for a in assets}, annual_volatility={a: 0.28 for a in assets}),
            "crash": MarketRegime(name="crash", annual_drift={a: -0.45 for a in assets}, annual_volatility={a: 0.55 for a in assets}),
        }
        trans_matrix = np.array([
            [0.85, 0.12, 0.02, 0.01],
            [0.10, 0.80, 0.08, 0.02],
            [0.05, 0.15, 0.75, 0.05],
            [0.15, 0.35, 0.40, 0.10],
        ])
        return cls(regimes=regimes, transition_matrix=trans_matrix, initial_regime="normal", enabled=False)
