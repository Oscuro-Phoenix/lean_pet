"""
Electrode parameters and dimensionless-number calculations.

Centralises the physical constants and electrode geometry that every
protocol script needs, so they are defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from lean_pet.core.electrolyte import electrolyte_conductivity

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
F = 96485.0        # Faraday constant  [C mol⁻¹]
R_GAS = 8.314      # Universal gas constant [J mol⁻¹ K⁻¹]
V_T = 0.0257       # Thermal voltage at 298.15 K  [V]


@dataclass
class ElectrodeParameters:
    """
    Collects all physical parameters for a porous-electrode half-cell and
    derives the dimensionless groups (Da numbers) used by the analytical
    models.

    Default values correspond to the NMC532 cathode used throughout this
    project.
    """

    # --- Geometry ---
    L: float = 100e-6          # Electrode thickness [m]
    R_p: float = 500e-9        # Particle radius [m]
    poros: float = 0.5         # Porosity
    P_L: float = 0.69          # Active-material loading factor
    BruggExp: float = 1.5      # Bruggeman exponent

    # --- Material ---
    sigma_s: float = 1e-1      # Solid electronic conductivity [S m⁻¹]
    c_s_max: float = field(default=None)  # type: ignore[assignment]
    c0: float = 1000.0         # Initial electrolyte concentration [mol m⁻³]
    t_plus: float = 0.38       # Cation transference number
    T_ref: float = 298.15      # Reference temperature [K]

    # --- Kinetics ---
    k0: float = 5.0            # Exchange-current-density prefactor [A m⁻²]
    R_film: float = 0.0        # Film resistance [Ω m²]
    C_DL: float = 0.2          # Double-layer capacitance [F m⁻²]

    # --- Diffusivities ---
    Dp: float = 2.2e-10        # Cation diffusivity [m² s⁻¹]
    Dm: float = 2.94e-10       # Anion diffusivity [m² s⁻¹]

    # --- Cell geometry (for absolute-current scaling) ---
    height: float = 1e-2       # Electrode height [m]
    width: float = 1e-2        # Electrode width [m]

    def __post_init__(self):
        if self.c_s_max is None:
            self.c_s_max = 2.9869e28 * 1.6e-19 / F  # NMC532 default [mol m⁻³]

    # --- Derived quantities ------------------------------------------------

    @property
    def electrode_area(self) -> float:
        """Geometric electrode area [m²]."""
        return self.height * self.width

    @property
    def eps_am(self) -> float:
        """Active-material volume fraction ε_am = (1 − ε) · P_L."""
        return (1.0 - self.poros) * self.P_L

    @property
    def a_p(self) -> float:
        """Specific interfacial area  a_p = 3 ε_am / R_p  [m⁻¹]."""
        return 3.0 * self.eps_am / self.R_p

    @property
    def D_eff(self) -> float:
        """Effective binary electrolyte diffusivity (harmonic mean) [m² s⁻¹]."""
        return 2.0 * self.Dp * self.Dm / (self.Dp + self.Dm)

    @property
    def kappa_ref(self) -> float:
        """Bulk electrolyte conductivity at (c0, T_ref) [S m⁻¹]."""
        return float(electrolyte_conductivity(self.c0, self.T_ref))

    @property
    def kappa_eff(self) -> float:
        """Effective electrolyte conductivity (Bruggeman) [S m⁻¹]."""
        return self.kappa_ref * self.poros ** self.BruggExp

    @property
    def sigma_s_eff(self) -> float:
        """Effective solid electronic conductivity (Bruggeman) [S m⁻¹]."""
        return self.sigma_s * (1.0 - self.poros) ** self.BruggExp

    @property
    def sigma_eff(self) -> float:
        """Harmonic-mean effective conductivity [S m⁻¹]."""
        return (self.sigma_s_eff ** -1 + self.kappa_eff ** -1) ** -1

    @property
    def R_hf(self) -> float:
        """High-frequency resistance ratio σ_eff / κ_eff."""
        return self.sigma_eff / self.kappa_eff

    @property
    def nominal_capacity(self) -> float:
        """Nominal capacity [A h]."""
        return (
            self.electrode_area * self.L * self.eps_am
            * F * self.c_s_max / 3600.0
        )

    # --- Dimensionless numbers ---------------------------------------------

    def Da_w(self) -> float:
        """Wiring Damköhler number  Da_w = Da_w,σ + Da_w,κ."""
        return self.Da_w_sigma() + self.Da_w_kappa()

    def Da_w_sigma(self) -> float:
        """Electronic-wiring contribution to Da_w."""
        return self.k0 * self.a_p * self.L ** 2 / (self.sigma_s_eff * V_T)

    def Da_w_kappa(self) -> float:
        """Ionic-wiring contribution to Da_w."""
        return self.k0 * self.a_p * self.L ** 2 / (self.kappa_eff * V_T)

    def Da_p(self) -> float:
        """Transport Damköhler number."""
        return (
            self.k0 * self.a_p * self.L ** 2 * (1.0 - self.t_plus)
            / (self.poros ** self.BruggExp * self.c0 * F * self.D_eff)
        )

    def J_P(self, C_rate: float = 1.0) -> float:
        """
        Dimensionless current density (process number).

        Parameters
        ----------
        C_rate : float
            C-rate (default 1 C).
        """
        return (
            self.k0 * self.a_p * 3600.0
            / (self.P_L * C_rate * (1.0 - self.poros) * F * self.c_s_max)
        )

    def Da_lim(self, C_rate: float = 1.0) -> float:
        """Limiting Damköhler number  Da_p / J_P."""
        return self.Da_p() / self.J_P(C_rate)

    # --- EIS-specific dimensionless numbers --------------------------------

    def Dac(self, j0: float | None = None) -> float:
        """Capacitance Damköhler number for EIS."""
        j0 = j0 if j0 is not None else self.k0 / V_T
        return j0 / self.C_DL

    def Dap_eis(self, j0: float | None = None) -> float:
        """Particle-diffusion Damköhler number for EIS."""
        j0 = j0 if j0 is not None else self.k0 / V_T
        return j0 * self.a_p * V_T / (F * self.eps_am * self.c_s_max)

    def Daw_eis(self, j0: float | None = None) -> float:
        """Wiring Damköhler number for EIS."""
        j0 = j0 if j0 is not None else self.k0 / V_T
        return j0 * self.a_p * self.L ** 2 / self.sigma_eff

    # --- Convenience -------------------------------------------------------

    def beta(self) -> float:
        """Wiring-resistance split  β = Da_w,σ / Da_w."""
        daw = self.Da_w()
        return self.Da_w_sigma() / daw if daw > 0 else 0.5

    def dimensionless_summary(self, C_rate: float = 1.0) -> dict:
        """Return a dict of all dimensionless numbers at a given C-rate."""
        return {
            "Da_w": self.Da_w(),
            "Da_w_sigma": self.Da_w_sigma(),
            "Da_w_kappa": self.Da_w_kappa(),
            "Da_p": self.Da_p(),
            "J_P": self.J_P(C_rate),
            "Da_lim": self.Da_lim(C_rate),
            "beta": self.beta(),
        }

