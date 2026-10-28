"""
Non-Parametric Historical Bootstrap Engine.
Implements IID and Circular Block Bootstrap to simulate future asset return paths
directly from empirical distribution without imposing normal distribution assumptions.
Preserves joint cross-sectional asset correlations and serial dependence.
"""

from enum import Enum
from typing import Tuple
import numpy as np


class BootstrapType(str, Enum):
    IID = "iid"
    CIRCULAR_BLOCK = "circular_block"


def generate_iid_bootstrap_returns(
    historical_returns: np.ndarray,
    n_steps: int,
    n_sims: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Generate synthetic return paths by IID resampling cross-sectional rows.
    """
    T, n_assets = historical_returns.shape
    if T < 2:
        raise ValueError("Historical returns must have at least 2 observations for bootstrap.")

    indices = rng.randint(low=0, high=T, size=(n_sims, n_steps))
    simulated_returns = historical_returns[indices]
    return simulated_returns


def generate_circular_block_bootstrap_returns(
    historical_returns: np.ndarray,
    n_steps: int,
    n_sims: int,
    block_size: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Generate synthetic return paths via Circular Block Bootstrap.
    Preserves time-series serial correlation and joint cross-sectional asset correlation.
    """
    T, n_assets = historical_returns.shape
    if T < 2:
        raise ValueError("Historical returns must have at least 2 observations for block bootstrap.")

    eff_block_size = max(1, min(block_size, T))
    n_blocks = int(np.ceil(n_steps / eff_block_size))

    start_indices = rng.randint(low=0, high=T, size=(n_sims, n_blocks))
    block_offsets = np.arange(eff_block_size)
    expanded_indices = (start_indices[:, :, None] + block_offsets[None, None, :]) % T

    flat_indices = expanded_indices.reshape(n_sims, n_blocks * eff_block_size)[:, :n_steps]
    simulated_returns = historical_returns[flat_indices]
    return simulated_returns