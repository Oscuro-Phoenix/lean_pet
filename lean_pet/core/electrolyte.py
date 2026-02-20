"""Electrolyte transport properties."""

import numpy as np


def electrolyte_conductivity(c_e: float | np.ndarray, T: float = 298.15) -> float | np.ndarray:
    """
    Electrolyte ionic conductivity κ(c, T) based on a polynomial fit.

    Parameters
    ----------
    c_e : float or array_like
        Electrolyte concentration [mol m⁻³].
    T : float
        Temperature [K] (default 298.15 K).

    Returns
    -------
    kappa : float or ndarray
        Electrolyte conductivity [S m⁻¹].
    """
    k00, k01, k02 = -8.2488, 0.053248, -0.000029871
    k10, k11, k12 = 0.26235, -0.0093063, 0.000008069
    k20, k21 = 0.22002, -0.0001765

    c = np.asarray(c_e, dtype=float) / 1000.0  # mol m⁻³ → mol L⁻¹
    kappa_mS_cm = c * (
        k00 + k01 * T + k02 * T ** 2
        + k10 * c + k11 * c * T + k12 * c * T ** 2
        + k20 * c ** 2 + k21 * c ** 2 * T
    ) ** 2
    return kappa_mS_cm * 0.1  # mS cm⁻¹ → S m⁻¹

