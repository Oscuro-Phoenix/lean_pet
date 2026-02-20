#!/usr/bin/env python3
"""
MCMC-based sensitivity / identifiability analysis.

Two modes:
  * **3-D** (``main_3d``) — sample (Da_w, Da_p, J_base) jointly.
  * **2-D** (``main_2d``) — fix Da_p ≈ 0 and sample (Da_w, J_base).

Uses the ``emcee`` affine-invariant ensemble sampler with optional
multiprocessing.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from lean_pet.core.ocv import NMC532_Colclasure20
from lean_pet.core.analytical_models import predict_vq
from lean_pet.core.parameters import V_T

try:
    import emcee  # type: ignore
except ImportError:
    emcee = None


# ═══════════════════════════════════════════════════════════════════════════
# Shared statistics
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_loglike(y_obs: np.ndarray, y_pred: np.ndarray, sigma: float) -> float:
    """Gaussian log-likelihood (constant σ)."""
    resid = y_obs - y_pred
    var = sigma ** 2
    return -0.5 * (np.sum(resid ** 2) / var + y_obs.size * np.log(2.0 * np.pi * var))


def _predict_vq_for_mcmc(
    ocv_fn: Callable,
    x_axis: np.ndarray,
    Da_w: float,
    Da_p: float,
    J_P: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for the VQ model used inside log-prob."""
    Da_lim = Da_p / max(J_P, 1e-30)
    return predict_vq(
        ocv_fn, x_axis,
        Da_w, Da_w / 2, Da_w / 2, Da_p, Da_lim, J_P,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    ocv_fn: Callable,
    x_axis: np.ndarray,
    true_params_by_crate: Dict[float, Tuple[float, float, float, float]],
    noise_sigma: float = 0.005,
    multiplicative: bool = False,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic VQ datasets for one or more C-rates.

    Parameters
    ----------
    multiplicative : bool
        If *True*, perturb the **parameters** multiplicatively (no voltage
        noise).  If *False* (default), add i.i.d. Gaussian noise to the
        voltage curve.
    """
    datasets: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(42)

    for crate, (Da_w, Da_p, Da_lim, J_P) in true_params_by_crate.items():
        if multiplicative:
            Da_w_n = Da_w * (1 + rng.normal(0, noise_sigma))
            Da_p_n = Da_p * (1 + rng.normal(0, noise_sigma))
            J_P_n = J_P * (1 + rng.normal(0, noise_sigma)) / crate
            Da_lim_n = Da_p_n / max(J_P_n, 1e-30)
            X, V = predict_vq(
                ocv_fn, x_axis,
                Da_w_n, Da_w_n / 2, Da_w_n / 2, Da_p_n, Da_lim_n, J_P_n,
            )
        else:
            X, V = predict_vq(
                ocv_fn, x_axis,
                Da_w, Da_w / 2, Da_w / 2, Da_p, Da_lim, J_P,
            )
            V = V + rng.normal(0, noise_sigma, size=V.shape)

        datasets[crate] = (X, V)
    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# 3-D log-prob  (Da_w, Da_p, J_base)
# ═══════════════════════════════════════════════════════════════════════════

class _LogProb3D:
    """Picklable log-probability for 3-D (Da_w, Da_p, J_base) sampling."""

    def __init__(self, ocv_fn, x_axis, crates, X_obs, V_obs, bounds, sigma):
        self.ocv_fn = ocv_fn
        self.x_axis = x_axis
        self.crates = crates
        self.X_obs = X_obs
        self.V_obs = V_obs
        self.bounds = bounds
        self.sigma = sigma

    def __call__(self, theta: np.ndarray) -> float:
        if np.any(~np.isfinite(theta)):
            return -np.inf
        Da_w, Da_p, J_base = float(theta[0]), float(theta[1]), float(theta[2])
        (lw, hw), (lp, hp), (lj, hj) = self.bounds
        if not (lw <= Da_w <= hw and lp <= Da_p <= hp and lj <= J_base <= hj):
            return -np.inf

        ll = 0.0
        for i, c in enumerate(self.crates):
            J_P = J_base * c
            X_pred, V_pred = _predict_vq_for_mcmc(self.ocv_fn, self.x_axis, Da_w, Da_p, J_P)
            V_fit = np.interp(self.X_obs[i], X_pred, V_pred, left=V_pred[0], right=V_pred[-1])
            ll += gaussian_loglike(self.V_obs[i], V_fit, self.sigma)
        return ll


# ═══════════════════════════════════════════════════════════════════════════
# 2-D log-prob  (Da_w, J_base)  with Da_p fixed
# ═══════════════════════════════════════════════════════════════════════════

class _LogProb2D:
    """Picklable log-probability for 2-D (Da_w, J_base) sampling."""

    def __init__(self, ocv_fn, x_axis, crates, X_obs, V_obs, bounds, sigma, da_p_fixed):
        self.ocv_fn = ocv_fn
        self.x_axis = x_axis
        self.crates = crates
        self.X_obs = X_obs
        self.V_obs = V_obs
        self.bounds = bounds
        self.sigma = sigma
        self.da_p = da_p_fixed

    def __call__(self, theta: np.ndarray) -> float:
        if np.any(~np.isfinite(theta)):
            return -np.inf
        Da_w, J_base = float(theta[0]), float(theta[1])
        (lw, hw), (lj, hj) = self.bounds
        if not (lw <= Da_w <= hw and lj <= J_base <= hj):
            return -np.inf

        ll = 0.0
        for i, c in enumerate(self.crates):
            J_P = J_base / c
            X_pred, V_pred = _predict_vq_for_mcmc(self.ocv_fn, self.x_axis, Da_w, self.da_p, J_P)
            V_fit = np.interp(self.X_obs[i], X_pred, V_pred, left=V_pred[0], right=V_pred[-1])
            ll += gaussian_loglike(self.V_obs[i], V_fit, self.sigma)
        return ll


# ═══════════════════════════════════════════════════════════════════════════
# Sampler
# ═══════════════════════════════════════════════════════════════════════════

def run_emcee(
    log_prob,
    ndim: int,
    bounds: list[Tuple[float, float]],
    n_walkers: int = 32,
    n_steps: int = 10000,
    burn_in: int = 2000,
    use_pool: bool = False,
) -> np.ndarray:
    """
    Run the ``emcee`` ensemble sampler and return the flat chain
    (post burn-in).
    """
    if emcee is None:
        raise RuntimeError("emcee not installed — pip install emcee")

    rng = np.random.default_rng(1234)
    p0 = np.empty((n_walkers, ndim))
    for d, (lo, hi) in enumerate(bounds):
        p0[:, d] = rng.uniform(lo, hi, size=n_walkers)

    pool = None
    try:
        if use_pool:
            ctx = mp.get_context("fork")
            pool = ctx.Pool(processes=min(n_walkers, max(2, mp.cpu_count() - 1)))
            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob, pool=pool)
        else:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)

        state = sampler.run_mcmc(p0, burn_in, progress=False)
        sampler.reset()
        sampler.run_mcmc(state, n_steps, progress=False)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"  Acceptance rate: {np.mean(sampler.acceptance_fraction):.3f}")
    return sampler.get_chain(flat=True)


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def corner_plot(
    samples: Dict[str, np.ndarray],
    labels: Dict[str, str] | None = None,
    true_values: Dict[str, float] | None = None,
    log_scale: bool = False,
    bins: int = 40,
    figsize: Tuple[int, int] = (9, 9),
    out_path: Path | str | None = None,
):
    """Simple corner / marginal histogram plot."""
    keys = list(samples.keys())
    if labels is None:
        labels = {k: k for k in keys}

    data = {}
    for k in keys:
        s = np.asarray(samples[k])
        if log_scale:
            s = np.log10(np.where(s > 0, s, np.nan))
        data[k] = s

    if log_scale and true_values:
        true_values = {k: np.log10(v) for k, v in true_values.items() if v > 0}
    if log_scale:
        labels = {k: f"log₁₀({v})" for k, v in labels.items()}

    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[keys[i]], bins=bins, density=True, alpha=0.8, edgecolor="white")
                ax.set_title(labels[keys[i]])
                if true_values and keys[i] in true_values:
                    ax.axvline(true_values[keys[i]], color="k", ls="--", lw=1.5)
            elif i > j:
                ax.scatter(data[keys[j]], data[keys[i]], s=1, alpha=0.15, rasterized=True)
                if true_values:
                    if keys[j] in true_values:
                        ax.axvline(true_values[keys[j]], color="k", ls="--", lw=1)
                    if keys[i] in true_values:
                        ax.axhline(true_values[keys[i]], color="k", ls="--", lw=1)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(labels[keys[j]])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[keys[i]])

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"✓ Corner plot saved: {out_path}")
    return fig, axes


def plot_noise_vs_mcmc(
    noise_dists: Dict[str, list],
    mcmc_chain: Dict[str, np.ndarray],
    true_values: Dict[str, float],
    out_path: Path | str | None = None,
):
    """Side-by-side histograms: input-noise spread vs. MCMC posterior."""
    param_keys = [k for k in noise_dists if k.endswith("_noisy")]
    n = len(param_keys)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, nk in zip(axes, param_keys):
        base = nk.replace("_noisy", "")
        mcmc_key = f"samples_{base}" if f"samples_{base}" in mcmc_chain else None

        ax.hist(np.asarray(noise_dists[nk]), bins=40, alpha=0.6, color="#ff7f0e",
                edgecolor="white", density=True, histtype="stepfilled", label="Input noise")
        if mcmc_key:
            ax.hist(mcmc_chain[mcmc_key], bins=40, alpha=0.6, color="#1f77b4",
                    edgecolor="white", density=True, histtype="stepfilled", label="MCMC posterior")
        if base in true_values:
            ax.axvline(true_values[base], color="black", ls="--", lw=2, label="True value")

        ax.set_xlabel(base, fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"✓ Noise-vs-MCMC plot saved: {out_path}")
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# Noise-distribution generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_input_noise_distributions(
    true_params_by_crate: Dict[float, Tuple[float, float, float, float]],
    noise_sigma: float,
    n_realizations: int = 100,
) -> Dict[str, list]:
    """Multiple realisations of multiplicative parameter noise."""
    dists: Dict[str, list] = {"Da_w_noisy": [], "J_P_noisy": []}
    for seed in range(n_realizations):
        rng = np.random.default_rng(42 + seed)
        for _, (Da_w, _, _, J_P) in true_params_by_crate.items():
            dists["Da_w_noisy"].append(Da_w * (1 + rng.normal(0, noise_sigma)))
            dists["J_P_noisy"].append(J_P * (1 + rng.normal(0, noise_sigma)))
    return dists


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry points
# ═══════════════════════════════════════════════════════════════════════════

def main_3d():
    """Sample (Da_w, Da_p, J_base) jointly."""
    np.random.seed(42)
    x_axis = np.linspace(0.3, 0.95, 300)

    true_Da_w, true_Da_p = 5.0, 2.0
    JP_base = 0.2
    crates = [0.5, 1.0, 2.0]
    true_params = {c: (true_Da_w, true_Da_p, true_Da_p / (JP_base * c), JP_base * c) for c in crates}

    noise_sigma = 0.01
    datasets = generate_synthetic_dataset(NMC532_Colclasure20, x_axis, true_params, noise_sigma)

    priors = {"Da_w": (1, 10), "Da_p": (1, 10), "J_P": (0.05, 0.5)}

    # Build log-prob
    sorted_c = np.array(sorted(datasets.keys()))
    X_obs = [datasets[c][0] for c in sorted_c]
    V_obs = [datasets[c][1] for c in sorted_c]
    bounds = [priors["Da_w"], priors["Da_p"], priors["J_P"]]
    lp = _LogProb3D(NMC532_Colclasure20, x_axis, sorted_c, X_obs, V_obs,
                    tuple(bounds), noise_sigma)

    chain = run_emcee(lp, ndim=3, bounds=bounds, n_walkers=48, n_steps=6000,
                      burn_in=1500, use_pool=True)

    out_dir = Path("fit_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {"Da_w": chain[:, 0], "Da_p": chain[:, 1], "J_P": chain[:, 2]}
    corner_plot(samples, labels={"Da_w": "Da_wiring", "Da_p": "Da_transport", "J_P": "Da_process"},
                log_scale=True, out_path=out_dir / "mcmc_corner_3d.png")

    np.savez(out_dir / "mcmc_chain_3d.npz", **samples)
    plt.show()


def main_2d():
    """Fix Da_p ≈ 0 and sample (Da_w, J_base)."""
    np.random.seed(42)
    x_axis = np.linspace(0.3, 0.95, 300)

    true_Da_w = 5.0
    true_Da_p = 0.0
    JP_base = 0.2
    crates = [1.0]
    true_params = {c: (true_Da_w, true_Da_p, 0.0, JP_base) for c in crates}

    noise_sigma = 0.05
    datasets = generate_synthetic_dataset(
        NMC532_Colclasure20, x_axis, true_params, noise_sigma, multiplicative=True,
    )

    # Input-noise distributions
    noise_dists = generate_input_noise_distributions(true_params, noise_sigma, n_realizations=100)

    priors = {"Da_w": (1, 10), "J_P": (0.05, 0.5)}

    sorted_c = np.array(sorted(datasets.keys()))
    X_obs = [datasets[c][0] for c in sorted_c]
    V_obs = [datasets[c][1] for c in sorted_c]
    bounds = [priors["Da_w"], priors["J_P"]]
    lp = _LogProb2D(NMC532_Colclasure20, x_axis, sorted_c, X_obs, V_obs,
                    tuple(bounds), noise_sigma, da_p_fixed=0.0)

    chain = run_emcee(lp, ndim=2, bounds=bounds, n_walkers=48, n_steps=6000,
                      burn_in=1500, use_pool=True)

    out_dir = Path("fit_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    mcmc_chain = {"samples_Da_w": chain[:, 0], "samples_J_P": chain[:, 1]}
    true_vals = {"Da_w": true_Da_w, "J_P": JP_base}

    plot_noise_vs_mcmc(noise_dists, mcmc_chain, true_vals,
                       out_path=out_dir / "noise_vs_mcmc_2d.png")

    corner_plot(
        {"Da_w": chain[:, 0], "J_P": chain[:, 1]},
        labels={"Da_w": "Da_wiring", "J_P": "Da_process"},
        true_values=true_vals,
        log_scale=False,
        out_path=out_dir / "mcmc_corner_2d.png",
    )

    np.savez(out_dir / "mcmc_chain_2d.npz", **mcmc_chain)
    plt.show()


if __name__ == "__main__":
    main_3d()

