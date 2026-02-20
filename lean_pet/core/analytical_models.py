"""
Analytical porous-electrode models for the three protocols:

* **Discharge (VQ)**  — voltage vs. capacity at constant current.
* **Pulsing (I-t)**   — current vs. time after a voltage step.
* **EIS**             — complex impedance vs. frequency.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from lean_pet.core.kinetics import ecd_mhc, ecd_mhc_df_dclyte
from lean_pet.core.ocv import NMC532_Colclasure20, NMC532_Colclasure20_deriv
from lean_pet.core.parameters import V_T


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Discharge  —  Voltage vs. Capacity  (VQ)
# ═══════════════════════════════════════════════════════════════════════════

def predict_vq(
    ocv_function: Callable,
    X: np.ndarray,
    Da_w: float,
    Da_w_sigma: float,
    Da_w_kappa: float,
    Da_p: float,
    Da_lim: float,
    J_P: float,
    k0: float = 5e-6,
    R_film: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the voltage–capacity curve using the CIET/MHC analytical model.

    Parameters
    ----------
    ocv_function : callable
        OCV(x) returning voltage [V] for stoichiometry *x*.
    X : array_like
        Stoichiometry (filling-fraction) grid.
    Da_w, Da_w_sigma, Da_w_kappa, Da_p, Da_lim, J_P : float
        Dimensionless numbers.
    k0 : float
        Rate-constant prefactor [A m⁻²].
    R_film : float
        Film resistance [Ω m²].

    Returns
    -------
    X, V : ndarray
        Stoichiometry and corresponding voltage arrays.
    """
    X = np.asarray(X, dtype=float)
    ec = ecd_mhc(X, c_lyte=1.0, k0=k0, R_film=R_film)
    alpha = ecd_mhc_df_dclyte(X, c_lyte=1.0, k0=k0, R_film=R_film)

    Lambda = np.sqrt(Da_w * ec + alpha * Da_p / J_P)
    beta = Da_w_sigma / Da_w if Da_w != 0 else 0.5

    Xi = np.zeros_like(X)
    mask = ec > 1e-10
    Lam = Lambda[mask]
    Z = (Lam ** 2) * (
        2.0 * beta * (1.0 - beta) * (0.5 + 1.0 / (Lam * np.sinh(Lam)))
        + (beta ** 2 + (1.0 - beta) ** 2) * np.cosh(Lam) / (Lam * np.sinh(Lam))
    )
    Xi[mask] = Z / (J_P * ec[mask])

    V = ocv_function(X) - np.abs(Xi) * V_T
    return X, V


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Pulsing  —  Current vs. Time  (I-t)
# ═══════════════════════════════════════════════════════════════════════════

def _get_AB(
    ocv_function: Callable,
    ocv_deriv_function: Callable,
    X: float,
    V_app: float,
    Da_w: float,
    J_P: float,
    k0: float = 5e-6,
    R_film: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute the linearised amplitude *A* and decay rate *B* for the
    current-transient model  I(t) = −A exp(B t).
    """
    eps = 1e-6

    ecd_X = ecd_mhc(X, k0=k0, R_film=R_film)
    Lambda_X = np.sqrt(Da_w * ecd_X)
    h_X = np.tanh(Lambda_X) / Lambda_X
    g_X = -V_app / V_T
    A = J_P * ecd_X * g_X * h_X

    # Logarithmic derivatives
    docv_dX = ocv_deriv_function(X)

    ecd_eps = ecd_mhc(X + eps, k0=k0, R_film=R_film)
    Lambda_eps = np.sqrt(Da_w * ecd_eps)
    h_eps = np.tanh(Lambda_eps) / Lambda_eps

    d_ln_f = (ecd_eps - ecd_X) / (eps * ecd_X)
    d_ln_g = -docv_dX / V_app
    d_ln_h = (h_eps - h_X) / (eps * h_X)

    B = A * (d_ln_f + d_ln_g + d_ln_h)
    return A, B


def predict_current_vs_time(
    ocv_function: Callable,
    ocv_deriv_function: Callable,
    X: float,
    V_app: float,
    Da_w: float,
    J_P: float,
    k0: float = 5e-6,
    R_film: float = 0.0,
    t_max: float = 1000.0,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the current transient after a voltage step.

    Parameters
    ----------
    ocv_function, ocv_deriv_function : callable
        OCV and its derivative.
    X : float
        Average stoichiometry at the time of the step.
    V_app : float
        Applied overpotential (V_step − OCV) [V].
    Da_w, J_P : float
        Dimensionless numbers.
    k0, R_film : float
        Kinetic parameters.
    t_max : float
        Maximum time [s].
    n_points : int
        Number of time points.

    Returns
    -------
    t, I : ndarray
        Time [s] and current arrays.
    """
    A, B = _get_AB(ocv_function, ocv_deriv_function, X, V_app, Da_w, J_P, k0, R_film)
    t = np.linspace(0, t_max, n_points)
    I = -A * np.exp(B * t)
    return t, I


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EIS  —  Complex Impedance
# ═══════════════════════════════════════════════════════════════════════════

def calculate_eis_impedance(
    Dac: float,
    Dap: float,
    Daw: float,
    omega: np.ndarray,
    stoichiometry: float = 0.3,
    R_hf: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical EIS impedance using the CIET/MHC kinetics model.

    Parameters
    ----------
    Dac : float
        Capacitance Damköhler number.
    Dap : float
        Particle-diffusion Damköhler number.
    Daw : float
        Wiring Damköhler number.
    omega : array_like
        Angular frequency [rad s⁻¹].
    stoichiometry : float
        Cathode stoichiometry at which to evaluate (default 0.3).
    R_hf : float
        High-frequency resistance ratio σ_eff / κ_eff.

    Returns
    -------
    Z_real, Z_imag : ndarray
        Real and (Nyquist-convention negative) imaginary impedance.
    """
    from lean_pet.core.parameters import F as F_const

    L_c = 100e-6
    poros_c = 0.5
    P_L_c = 0.69
    c_s_max = 2.9869e28 * 1.6e-19 / F_const
    eps_solid = (1.0 - poros_c) * P_L_c

    dphidc = NMC532_Colclasure20_deriv(stoichiometry) / V_T
    f = ecd_mhc(stoichiometry, c_lyte=1.0, k0=5.0, R_film=0.0)

    Lambda = np.sqrt(
        Daw * (f + 1j * omega / Dac - dphidc * Dap * f / Dac)
        / (1.0 - dphidc * Dap * f / (1j * omega))
    )

    Z_ref = Daw / Dap * V_T / (L_c * eps_solid * F_const * c_s_max)
    Z_complex = Z_ref * (
        R_hf * (1.0 + 2.0 / (Lambda * np.sinh(Lambda)))
        + (1.0 + R_hf ** 2) / (np.tanh(Lambda) * Lambda)
    ) / (1.0 + R_hf) ** 2

    return np.real(Z_complex), -np.imag(Z_complex)

