#!/usr/bin/env python3
"""
Chi-square landscape analysis for parameter identifiability.

Supports two data sources:
  * **Synthetic** — model-generated noisy data (self-consistency test).
  * **MPET**      — full-scale MPET simulation data.

Both produce a 2-D contour plot of log₁₀(χ²) in (Da_w, Da_p) space.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lean_pet.core.ocv import NMC532_Colclasure20
from lean_pet.core.analytical_models import predict_vq
from lean_pet.core.electrolyte import electrolyte_conductivity
from lean_pet.core.parameters import F, V_T
from lean_pet.core.plotting import apply_publication_style_serif


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _predict_vq_wrapper(
    Da_w: float,
    Da_p: float,
    J_P: float,
    capacity_range: np.ndarray,
    beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around :func:`predict_vq` with ``beta`` split."""
    capacity_grid = np.linspace(capacity_range.min(), capacity_range.max(), 100)
    Da_lim = Da_p / max(J_P, 1e-30)
    Da_w_sigma = beta * Da_w
    Da_w_kappa = (1.0 - beta) * Da_w
    try:
        return predict_vq(
            NMC532_Colclasure20, capacity_grid,
            Da_w, Da_w_sigma, Da_w_kappa, Da_p, Da_lim, J_P,
        )
    except Exception:
        return capacity_grid, np.full_like(capacity_grid, np.nan)


def calculate_chi_square(
    Da_w: float,
    Da_p: float,
    J_P: float,
    capacity_true: np.ndarray,
    voltage_true: np.ndarray,
    beta: float = 0.5,
) -> float:
    """Sum of squared residuals between data and the VQ prediction."""
    cap_pred, volt_pred = _predict_vq_wrapper(Da_w, Da_p, J_P, capacity_true, beta)
    if np.any(np.isnan(volt_pred)):
        return np.inf
    volt_interp = interp1d(cap_pred, volt_pred, bounds_error=False, fill_value="extrapolate")(capacity_true)
    return float(np.sum((voltage_true - volt_interp) ** 2))


def find_optimal_parameters(
    capacity_true: np.ndarray,
    voltage_true: np.ndarray,
    J_P: float,
    initial_guess: Tuple[float, float] | None = None,
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """Minimise χ² over (Da_w, Da_p) with Nelder-Mead."""
    x0 = list(initial_guess) if initial_guess else [1.0, 0.1]

    def objective(params):
        Da_w, Da_p = params
        if Da_w <= 0 or Da_p <= 0:
            return np.inf
        return calculate_chi_square(Da_w, Da_p, J_P, capacity_true, voltage_true, beta)

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-8})
    Da_w_opt, Da_p_opt = result.x
    print(f"  Optimal: Da_w={Da_w_opt:.6f}, Da_p={Da_p_opt:.6f}, χ²={result.fun:.4e}")
    return Da_w_opt, Da_p_opt, float(result.fun)


# ═══════════════════════════════════════════════════════════════════════════
# Contour-plot builder
# ═══════════════════════════════════════════════════════════════════════════

def plot_chisquare_landscape(
    capacity_true: np.ndarray,
    voltage_true: np.ndarray,
    J_P: float,
    Da_w_opt: float,
    Da_p_opt: float,
    *,
    true_Da_w: float | None = None,
    true_Da_p: float | None = None,
    beta: float = 0.5,
    grid_resolution: int = 50,
    range_factor: float = 3.0,
    save_path: str | None = None,
):
    """
    2-D filled contour of log₁₀(χ²) around the optimum.
    """
    apply_publication_style_serif()

    Da_w_grid = np.linspace(Da_w_opt / range_factor, Da_w_opt * range_factor, grid_resolution)
    Da_p_grid = np.linspace(Da_p_opt / range_factor, Da_p_opt * range_factor, grid_resolution)
    Dw, Dp = np.meshgrid(Da_w_grid, Da_p_grid)

    chi2 = np.zeros_like(Dw)
    total = grid_resolution ** 2
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            chi2[i, j] = calculate_chi_square(Dw[i, j], Dp[i, j], J_P,
                                              capacity_true, voltage_true, beta)
        done = (i + 1) * grid_resolution
        if done % 500 < grid_resolution:
            print(f"  {done}/{total}")

    max_fin = np.nanmax(chi2[np.isfinite(chi2)])
    chi2[~np.isfinite(chi2)] = max_fin * 10
    chi2_log = np.log10(chi2 + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 10))
    cf = ax.contourf(Dw, Dp, chi2_log, levels=20, cmap="coolwarm", alpha=0.9)
    ax.contour(Dw, Dp, chi2_log, levels=10, colors="black", alpha=0.3, linewidths=1)

    ax.plot(Da_w_opt, Da_p_opt, "k*", markersize=20, markeredgecolor="white",
            markeredgewidth=1.5,
            label=f"Optimal: Da_w={Da_w_opt:.4f}, Da_p={Da_p_opt:.4f}")

    if true_Da_w is not None and true_Da_p is not None:
        ax.plot(true_Da_w, true_Da_p, "g^", markersize=15, markeredgecolor="white",
                markeredgewidth=1.5,
                label=f"True: Da_w={true_Da_w:.4f}, Da_p={true_Da_p:.4f}")

    ax.set_xlabel(r"$Da_{\mathrm{wiring}}$", fontsize=26)
    ax.set_ylabel(r"$Da_{\mathrm{process}}$", fontsize=26)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(\chi^2)$", fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    cbar.locator = MaxNLocator(nbins=20)
    cbar.update_ticks()

    ax.legend(loc="upper right", fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"✓ χ² landscape saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic-data entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    x_axis: np.ndarray,
    true_Da_w: float,
    true_Da_p: float,
    true_J_P: float,
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturb model parameters multiplicatively and generate a clean VQ curve
    from the noisy parameters (parameter-noise test).
    """
    rng = np.random.default_rng(seed)
    Da_w_n = max(true_Da_w * (1 + rng.normal(0, noise_sigma)), 1e-6)
    Da_p_n = max(true_Da_p * (1 + rng.normal(0, noise_sigma)), 1e-6)
    J_P_n = max(true_J_P * (1 + rng.normal(0, noise_sigma)), 1e-6)
    Da_lim_n = Da_p_n / max(J_P_n, 1e-30)

    X, V = predict_vq(
        NMC532_Colclasure20, x_axis,
        Da_w_n, Da_w_n / 2, Da_w_n / 2, Da_p_n, Da_lim_n, J_P_n,
    )
    return X, V


def main_synthetic():
    """CLI: chi-square landscape from model-generated data."""
    from pathlib import Path

    np.random.seed(42)
    true_Da_w, true_Da_p, true_J_P = 5.0, 0.1, 0.2
    x_axis = np.linspace(0.3, 0.95, 100)

    cap, volt = generate_synthetic_data(x_axis, true_Da_w, true_Da_p, true_J_P)

    Da_w_opt, Da_p_opt, _ = find_optimal_parameters(
        cap, volt, true_J_P,
        initial_guess=(true_Da_w * 1.5, true_Da_p * 0.7),
    )

    out = Path("chisquare_landscape_model.png")
    plot_chisquare_landscape(
        cap, volt, true_J_P, Da_w_opt, Da_p_opt,
        true_Da_w=true_Da_w, true_Da_p=true_Da_p,
        save_path=str(out),
    )


# ═══════════════════════════════════════════════════════════════════════════
# MPET-data entry point
# ═══════════════════════════════════════════════════════════════════════════

def main_mpet():
    """CLI: chi-square landscape from MPET simulation data."""
    sim_folder = (
        "/home/shakulp/Desktop/leanpet/NMC_grid/"
        "sigma_s_c=1.0000e-01_Crate=1.0556e+00_c0=1.0556e+03"
    )
    if not os.path.exists(sim_folder):
        print(f"Simulation folder not found: {sim_folder}")
        return

    mat_file = os.path.join(sim_folder, "output_data.mat")
    data = scipy.io.loadmat(mat_file)
    capacity = np.squeeze(data["ffrac_c"])
    voltage = NMC532_Colclasure20(capacity[0]) - V_T * np.squeeze(data["phi_applied"])

    # Parse physical parameters from folder name
    folder_name = os.path.basename(sim_folder)
    sigma_s_c = float(re.search(r"sigma_s_c=([\d\.e\-\+]+)", folder_name).group(1))
    C_rate = float(re.search(r"Crate=([\d\.e\-\+]+)", folder_name).group(1))
    c0 = float(re.search(r"c0=([\d\.e\-\+]+)", folder_name).group(1))

    k0_ref = 1e-1
    R_p = 500e-9
    P_L = 0.69
    L = 100e-6
    poros = 0.5
    a_p = (3 / R_p) * (1 - poros)

    kappa_eff = 0.13
    sigma_s_c_eff = sigma_s_c * ((1 - poros) ** 1.5)
    beta = kappa_eff / (kappa_eff + sigma_s_c_eff)
    sigma_eff = (sigma_s_c_eff ** -1 + kappa_eff ** -1) ** -1

    c_s_max = 2.9869e28 * 1.6e-19 / F
    J_P = k0_ref * a_p * 3600 / (P_L * C_rate * (1 - poros) * c_s_max * 1.602e-19 * F / F)
    # Simplified: use the compare_mpet formula
    J_P = k0_ref * a_p * 3600 / (P_L * C_rate * (1 - poros) * 2.9869e28 * 1.602e-19)

    Da_p_init = k0_ref * a_p * L ** 2 * (1 - 0.38) / (poros ** 1.5 * c0 * F * 10 ** (-9.65))
    Da_w_init = k0_ref * a_p * L ** 2 / (sigma_eff * V_T) + 2 * (1 - 0.38) * Da_p_init

    Da_w_opt, Da_p_opt, _ = find_optimal_parameters(
        capacity, voltage, J_P,
        initial_guess=(Da_w_init, Da_p_init),
        beta=beta,
    )

    plot_chisquare_landscape(
        capacity, voltage, J_P, Da_w_opt, Da_p_opt,
        beta=beta,
        save_path="chisquare_landscape_mpet.png",
    )


if __name__ == "__main__":
    main_synthetic()

