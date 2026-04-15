#!/usr/bin/env python3
"""
Chi-square landscape analysis using model-generated synthetic noisy data.
Creates a 2D contour plot showing the chi-square landscape around the optimal fit.

Unlike chisquare_landscape_mpet.py which uses MPET simulation data, this script
generates synthetic noisy data from the VQ model itself (similar to
sensitivity_based_identifiability_Da.py) to test parameter identifiability.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from typing import Tuple
from pathlib import Path

from lean_pet.core.analytical_models import predict_vq
from lean_pet.core.ocv import NMC532_Colclasure20
from lean_pet.core.plotting import apply_publication_style_serif
from lean_pet.core.parameters import V_T


def generate_synthetic_data(
    x_axis: np.ndarray,
    true_Da_w: float,
    true_Da_p: float,
    true_J_P: float,
    noise_sigma: float = 0.05,
    beta: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic noisy V-Q data by perturbing model parameters.

    Applies multiplicative Gaussian noise to (Da_w, Da_p, J_P) and generates
    a clean VQ curve from the perturbed values.

    Parameters
    ----------
    x_axis : array
        Fractional capacity points.
    true_Da_w, true_Da_p, true_J_P : float
        True dimensionless numbers.
    noise_sigma : float
        Standard deviation of multiplicative noise on each parameter.
    beta : float
        Wiring-resistance split Da_w_sigma / Da_w.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X, V : ndarray
        Capacity and voltage arrays.
    """
    rng = np.random.default_rng(seed)

    Da_w_n = max(true_Da_w * (1.0 + rng.normal(0.0, noise_sigma)), 1e-6)
    Da_p_n = max(true_Da_p * (1.0 + rng.normal(0.0, noise_sigma)), 1e-6)
    Da_lim_n = Da_p_n / max(true_J_P, 1e-30)

    print(f"\nGenerating synthetic data:")
    print(f"  True parameters: Da_w={true_Da_w:.4f}, Da_p={true_Da_p:.4f}, J_P={true_J_P:.4f}")
    print(f"  Noisy parameters: Da_w={Da_w_n:.4f}, Da_p={Da_p_n:.4f} (J_P held fixed)")

    X, V = predict_vq(
        NMC532_Colclasure20, x_axis,
        Da_w_n, beta * Da_w_n, (1.0 - beta) * Da_w_n,
        Da_p_n, Da_lim_n, true_J_P,
    )
    return X, V


def predict_vq_curve(
    Da_w: float,
    Da_p: float,
    J_P: float,
    capacity_range: np.ndarray,
    beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict voltage-capacity curve using the lean_pet VQ model."""
    capacity_grid = np.linspace(capacity_range.min(), capacity_range.max(), 100)
    Da_lim = Da_p / max(J_P, 1e-30)
    try:
        return predict_vq(
            NMC532_Colclasure20, capacity_grid,
            Da_w, beta * Da_w, (1.0 - beta) * Da_w,
            Da_p, Da_lim, J_P,
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
    cap_pred, volt_pred = predict_vq_curve(Da_w, Da_p, J_P, capacity_true, beta)
    if np.any(np.isnan(volt_pred)):
        return np.inf
    volt_interp = interp1d(
        cap_pred, volt_pred, bounds_error=False, fill_value="extrapolate",
    )(capacity_true)
    return float(np.sum((voltage_true - volt_interp) ** 2))


def find_optimal_parameters(
    capacity_true: np.ndarray,
    voltage_true: np.ndarray,
    J_P: float,
    initial_guess: Tuple[float, float] | None = None,
    beta: float = 0.5,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
) -> Tuple[float, float, float]:
    """
    Minimise chi-square over (Da_w, Da_p).

    Uses differential evolution (global) in log-space for the initial sweep,
    then polishes with L-BFGS-B.  Falls back to multi-start Nelder-Mead if
    the global search is skipped (bounds=None and no guess).
    """
    if bounds is None:
        g = initial_guess or (1.0, 0.1)
        bounds = ((g[0] * 0.01, g[0] * 100.0),
                  (g[1] * 0.01, g[1] * 100.0))

    log_bounds = [(np.log10(lo), np.log10(hi)) for lo, hi in bounds]

    def objective_log(log_params):
        Da_w, Da_p = 10.0 ** log_params[0], 10.0 ** log_params[1]
        return calculate_chi_square(Da_w, Da_p, J_P,
                                    capacity_true, voltage_true, beta)

    # --- Stage 1: differential evolution (global) in log-space ---
    de_result = differential_evolution(
        objective_log, log_bounds,
        seed=42, maxiter=300, tol=1e-10, polish=False,
        mutation=(0.5, 1.5), recombination=0.9, popsize=20,
    )
    best_log = de_result.x
    best_fun = de_result.fun
    print(f"  DE global:  Da_w={10**best_log[0]:.6f}, "
          f"Da_p={10**best_log[1]:.6f}, chi2={best_fun:.4e}")

    # --- Stage 2: L-BFGS-B polish in log-space ---
    polish = minimize(
        objective_log, best_log, method="L-BFGS-B",
        bounds=log_bounds,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
    )
    if polish.fun < best_fun:
        best_log = polish.x
        best_fun = polish.fun

    Da_w_opt = 10.0 ** best_log[0]
    Da_p_opt = 10.0 ** best_log[1]
    print(f"  Polished:   Da_w={Da_w_opt:.6f}, Da_p={Da_p_opt:.6f}, "
          f"chi2={best_fun:.4e}")
    return Da_w_opt, Da_p_opt, float(best_fun)


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
    """2-D filled contour of log10(chi2) around the optimum."""
    apply_publication_style_serif()

    Da_w_grid = np.linspace(Da_w_opt / range_factor, Da_w_opt * range_factor, grid_resolution)
    Da_p_grid = np.linspace(Da_p_opt / range_factor, Da_p_opt * range_factor, grid_resolution)
    Dw, Dp = np.meshgrid(Da_w_grid, Da_p_grid)

    chi2 = np.zeros_like(Dw)
    total = grid_resolution ** 2
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            chi2[i, j] = calculate_chi_square(
                Dw[i, j], Dp[i, j], J_P, capacity_true, voltage_true, beta,
            )
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
        print(f"  Saved: {save_path}")
    plt.show()


def plot_data_comparison(
    capacity_true: np.ndarray,
    voltage_true: np.ndarray,
    Da_w_opt: float,
    Da_p_opt: float,
    J_P: float,
    true_Da_w: float,
    true_Da_p: float,
    true_J_P: float,
    beta: float = 0.5,
    save_path: str | None = None,
):
    """Plot comparison of synthetic data vs optimal fit."""
    apply_publication_style_serif()

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(capacity_true, voltage_true, "ko", markersize=6,
            label="Synthetic data (noisy params)", alpha=0.6)

    cap_pred, volt_pred = predict_vq_curve(Da_w_opt, Da_p_opt, J_P, capacity_true, beta)
    ax.plot(cap_pred, volt_pred, "r-", linewidth=2.5,
            label=f"Optimal fit: $Da_w$={Da_w_opt:.4f}, $Da_p$={Da_p_opt:.4f}")

    Da_lim_true = true_Da_p / max(true_J_P, 1e-30)
    cap_true_curve, volt_true_curve = predict_vq(
        NMC532_Colclasure20, capacity_true,
        true_Da_w, beta * true_Da_w, (1.0 - beta) * true_Da_w,
        true_Da_p, Da_lim_true, true_J_P,
    )
    ax.plot(cap_true_curve, volt_true_curve, "g--", linewidth=2.5,
            label=f"True params: $Da_w$={true_Da_w:.4f}, $Da_p$={true_Da_p:.4f}",
            alpha=0.8)

    ax.set_xlabel("Fractional Capacity", fontsize=24)
    ax.set_ylabel("Voltage (V)", fontsize=24)
    ax.legend(loc="best", fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        print(f"  Saved: {save_path}")
    plt.show()


def main(C_rate: float = 1.0):
    """CLI: chi-square landscape from model-generated data."""
    from lean_pet.core.parameters import ElectrodeParameters

    np.random.seed(42)

    params = ElectrodeParameters()
    true_Da_w = params.Da_w()
    true_Da_p = params.Da_p()
    true_J_P = params.J_P(C_rate)
    beta = params.beta()

    print(f"  ElectrodeParameters @ {C_rate}C:")
    print(f"    Da_w={true_Da_w:.4f}  Da_p={true_Da_p:.4f}  "
          f"J_P={true_J_P:.4f}  beta={beta:.4f}")

    x_axis = np.linspace(0.3, 0.95, 100)

    cap, volt = generate_synthetic_data(
        x_axis, true_Da_w, true_Da_p, true_J_P, beta=beta,
    )

    Da_w_opt, Da_p_opt, _ = find_optimal_parameters(
        cap, volt, true_J_P,
        initial_guess=(true_Da_w * 1.5, true_Da_p * 0.7),
        beta=beta,
    )

    print(f"\n  True Da_w={true_Da_w:.4f}  Recovered={Da_w_opt:.4f}  "
          f"Error={abs(Da_w_opt - true_Da_w) / true_Da_w * 100:.1f}%")
    print(f"  True Da_p={true_Da_p:.4f}  Recovered={Da_p_opt:.4f}  "
          f"Error={abs(Da_p_opt - true_Da_p) / true_Da_p * 100:.1f}%")

    out = Path("chisquare_landscape_model.png")
    plot_chisquare_landscape(
        cap, volt, true_J_P, Da_w_opt, Da_p_opt,
        true_Da_w=true_Da_w, true_Da_p=true_Da_p,
        beta=beta, save_path=str(out),
    )

    plot_data_comparison(
        cap, volt, Da_w_opt, Da_p_opt, true_J_P,
        true_Da_w, true_Da_p, true_J_P,
        beta=beta, save_path="chisquare_model_data_comparison.png",
    )


if __name__ == "__main__":
    main()

