"""
CIET / Marcus-Hush-Chidsey (MHC) kinetics.

Provides both NumPy (for analytical models) and PyBaMM (for simulations)
implementations of the exchange-current-density function and the full
CIET kinetics sub-model.
"""

from __future__ import annotations

import numpy as np
import scipy.special as spl


# ---------------------------------------------------------------------------
# NumPy implementations (used by analytical models)
# ---------------------------------------------------------------------------

# Reorganization energy in thermal-voltage units  (λ / V_T)
_LAMBDA_DIM = 0.112 / 0.0257


def _activity_correction(c_lyte: float | np.ndarray) -> float | np.ndarray:
    """Electrolyte activity correction  c̃ = c·1.9·e⁻¹ / (1 + c·1.9·e⁻¹)."""
    fac = c_lyte * 1.9 * np.exp(-1)
    return fac / (1.0 + fac)


def ecd_mhc(
    c_sld: np.ndarray,
    c_lyte: float = 1.0,
    k0: float = 5e-6,
    R_film: float = 0.0,
) -> np.ndarray:
    """
    Exchange-current-density factor *f* for CIET/MHC kinetics.

    Parameters
    ----------
    c_sld : array_like
        Dimensionless solid concentration (0, 1).
    c_lyte : float
        Dimensionless electrolyte concentration (default 1).
    k0 : float
        Rate-constant prefactor [A m⁻²].
    R_film : float
        Film resistance [Ω m²].

    Returns
    -------
    f : ndarray
        Dimensionless exchange-current-density factor.
    """
    c_lyte = _activity_correction(c_lyte)
    eta = np.log(c_lyte / c_sld)
    a = 1.0 + np.sqrt(_LAMBDA_DIM)
    erf_term = 1.0 - spl.erf(
        (_LAMBDA_DIM - np.sqrt(a + eta ** 2)) / (2.0 * np.sqrt(_LAMBDA_DIM))
    )
    f = (1.0 - c_sld) * c_sld * c_lyte / (c_sld + c_lyte) * erf_term / 2.0
    return f / (k0 * f * R_film / 0.0257 + 1.0)


def ecd_mhc_df_dclyte(
    c_sld: np.ndarray,
    c_lyte: float = 1.0,
    k0: float = 5e-6,
    R_film: float = 0.0,
) -> np.ndarray:
    """
    Derivative of the electrolyte-concentration factor α = d ln(c̃/(c̃+c_s)) / d c̃.

    Used in the VQ analytical model for the electrolyte-transport correction.
    """
    c_lyte_corr = _activity_correction(c_lyte)
    g = (
        (1.0 / (c_sld + c_lyte_corr)
         - c_lyte_corr / (c_sld + c_lyte_corr) ** 2)
        / (c_lyte_corr / (c_sld + c_lyte_corr))
    )
    return g


# ---------------------------------------------------------------------------
# PyBaMM-symbolic implementations (used only inside simulations)
# ---------------------------------------------------------------------------

def MHC_kfunc_pybamm(eta, lmbda):
    """
    Marcus-Hush-Chidsey rate function using PyBaMM symbolic operations.

    Parameters
    ----------
    eta : pybamm.Symbol
        Dimensionless overpotential η / V_T.
    lmbda : pybamm.Symbol
        Dimensionless reorganisation energy λ / V_T.

    Returns
    -------
    k : pybamm.Symbol
        MHC rate factor.
    """
    import pybamm  # deferred so the module loads without pybamm installed

    a = 1.0 + pybamm.sqrt(lmbda)
    term1 = 1.0 / (1.0 + pybamm.exp(-eta))
    arg_erf = (lmbda - pybamm.sqrt(a + eta ** 2)) / (2.0 * pybamm.sqrt(lmbda))
    term2 = 1.0 - pybamm.erf(arg_erf)
    return term1 * term2


def _make_cathode_kinetics_ciet():
    """
    Factory that returns the CathodeKineticsCIET class.

    Deferred import of ``pybamm`` so that the rest of the package works
    without it being installed.
    """
    import pybamm  # noqa: F811

    class CathodeKineticsCIET(pybamm.interface.kinetics.BaseKinetics):
        """Custom PyBaMM kinetics sub-model implementing CIET/MHC."""

        def __init__(self, param, domain, reaction, options=None, phase="primary"):
            super().__init__(param, domain, reaction, options, phase)

        def get_coupled_variables(self, variables):
            self.temp_variables = variables
            return super().get_coupled_variables(variables)

        def _get_kinetics(self, j0, ne, eta_r, T, u):
            domain = self.domain.capitalize()

            # --- Electrolyte concentration ---
            c_e_m3 = self.temp_variables[f"{domain} electrolyte concentration [mol.m-3]"]

            # --- Solid concentration ---
            for key_template in (
                "{d} particle concentration [mol.m-3]",
                "{d} primary particle concentration [mol.m-3]",
            ):
                key = key_template.format(d=domain)
                if key in self.temp_variables:
                    c_s_bulk_m3 = self.temp_variables[key]
                    break
            else:
                raise KeyError(
                    f"No particle concentration variable found for {domain}"
                )

            if hasattr(c_s_bulk_m3, "domain") and any(
                "particle" in d for d in c_s_bulk_m3.domain
            ):
                c_s_bulk_m3 = pybamm.r_average(c_s_bulk_m3)

            c_e0_m3 = 1000.0
            c_s_max_m3 = pybamm.Parameter(
                f"Maximum concentration in {domain.lower()} electrode [mol.m-3]"
            )

            # Dimensionless concentrations
            c_lyte = c_e_m3 / c_e0_m3
            c_lyte = (c_lyte * 1.9 * np.exp(-1)) / (1 + c_lyte * 1.9 * np.exp(-1))
            c_sld = c_s_bulk_m3 / c_s_max_m3

            eps = 1e-12
            c_lyte = pybamm.maximum(c_lyte, eps)
            c_sld = pybamm.minimum(pybamm.maximum(c_sld, eps), 1.0 - eps)

            V_thermal = pybamm.constants.R * T / pybamm.constants.F
            eta_dim = eta_r / V_thermal

            lmbda_eV = pybamm.Parameter(
                f"{domain} electrode reorganization energy [eV]"
            )
            lmbda_dim = lmbda_eV / V_thermal

            eta_f_dim = eta_dim + pybamm.log(c_lyte / c_sld)
            ecd_extras = (1.0 - c_sld) / 2.0

            krd = MHC_kfunc_pybamm(-eta_f_dim, lmbda_dim)
            kox = MHC_kfunc_pybamm(eta_f_dim, lmbda_dim)

            return -j0 * ecd_extras * (krd * c_lyte - kox * c_sld)

    return CathodeKineticsCIET


# Lazy singleton so ``from lean_pet.core.kinetics import CathodeKineticsCIET``
# works but pybamm is only imported when the class is actually *used*.
class _CathodeKineticsCIETProxy:
    """Transparent proxy that builds the real class on first attribute access."""

    _cls = None

    def __getattr__(self, name):
        if _CathodeKineticsCIETProxy._cls is None:
            _CathodeKineticsCIETProxy._cls = _make_cathode_kinetics_ciet()
        return getattr(_CathodeKineticsCIETProxy._cls, name)

    def __call__(self, *args, **kwargs):
        if _CathodeKineticsCIETProxy._cls is None:
            _CathodeKineticsCIETProxy._cls = _make_cathode_kinetics_ciet()
        return _CathodeKineticsCIETProxy._cls(*args, **kwargs)

    def __instancecheck__(cls, instance):
        if _CathodeKineticsCIETProxy._cls is None:
            _CathodeKineticsCIETProxy._cls = _make_cathode_kinetics_ciet()
        return isinstance(instance, _CathodeKineticsCIETProxy._cls)


CathodeKineticsCIET = _CathodeKineticsCIETProxy()

