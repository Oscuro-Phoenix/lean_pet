"""
MCMC posterior sampling of (Da_w, Da_p) from synthetic VQ data.

Samples the wiring and process Damkohler numbers while holding J_P fixed,
using the lean_pet CIET/MHC analytical VQ model.  Produces corner plots
and overlay plots showing the posterior distributions.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import emcee  # type: ignore
except Exception:
    emcee = None

from lean_pet.core.analytical_models import predict_vq
from lean_pet.core.ocv import NMC532_Colclasure20
from lean_pet.core.plotting import apply_publication_style_serif
from lean_pet.core.parameters import ElectrodeParameters


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _predict_vq_wrap(
    Da_w: float, Da_p: float, J_P: float,
    x_axis: np.ndarray, beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Safe wrapper around the lean_pet VQ model."""
    Da_lim = Da_p / max(J_P, 1e-30)
    try:
        return predict_vq(
            NMC532_Colclasure20, x_axis,
            Da_w, beta * Da_w, (1.0 - beta) * Da_w,
            Da_p, Da_lim, J_P,
        )
    except Exception:
        return x_axis, np.full_like(x_axis, np.nan)


def gaussian_loglike(y_obs: np.ndarray, y_pred: np.ndarray, sigma: float) -> float:
    resid = y_obs - y_pred
    var = sigma ** 2
    return -0.5 * (np.sum(resid * resid) / var + y_obs.size * np.log(2.0 * np.pi * var))


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    x_axis: np.ndarray,
    true_Da_w: float,
    true_Da_p: float,
    J_P: float,
    beta: float,
    crates: list[float],
    noise_sigma: float = 0.05,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic VQ curves for several C-rates by perturbing
    (Da_w, Da_p) multiplicatively.  J_P is scaled as J_P_base / C_rate.
    """
    datasets: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(42)
    for crate in crates:
        Da_w_n = max(true_Da_w * (1.0 + rng.normal(0.0, noise_sigma)), 1e-6)
        Da_p_n = max(true_Da_p * (1.0 + rng.normal(0.0, noise_sigma)), 1e-6)
        J_P_c = J_P / crate
        X, V = _predict_vq_wrap(Da_w_n, Da_p_n, J_P_c, x_axis, beta)
        datasets[crate] = (X, V)
    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# Log-probability (picklable class for emcee + multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

class LogProbDaWDaP:
    """
    Log-posterior for (Da_w, Da_p) with J_P fixed.
    Uniform priors within bounds; -inf outside.
    """

    def __init__(
        self,
        x_axis: np.ndarray,
        crates: np.ndarray,
        X_obs: list[np.ndarray],
        V_obs: list[np.ndarray],
        J_P_base: float,
        beta: float,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        sigma: float,
    ):
        self.x_axis = x_axis
        self.crates = crates
        self.X_obs = X_obs
        self.V_obs = V_obs
        self.J_P_base = J_P_base
        self.beta = beta
        self.bounds = bounds
        self.sigma = sigma

    def __call__(self, theta: np.ndarray) -> float:
        if np.any(~np.isfinite(theta)):
            return -np.inf
        Da_w, Da_p = float(theta[0]), float(theta[1])

        (lo_w, hi_w), (lo_p, hi_p) = self.bounds
        if not (lo_w <= Da_w <= hi_w) or not (lo_p <= Da_p <= hi_p):
            return -np.inf

        ll = 0.0
        for i, crate in enumerate(self.crates):
            J_P = self.J_P_base / crate
            X_pred, V_pred = _predict_vq_wrap(
                Da_w, Da_p, J_P, self.x_axis, self.beta,
            )
            if np.any(np.isnan(V_pred)):
                return -np.inf
            V_interp = np.interp(
                self.X_obs[i], X_pred, V_pred,
                left=V_pred[0], right=V_pred[-1],
            )
            ll += gaussian_loglike(self.V_obs[i], V_interp, self.sigma)
        return ll


# ═══════════════════════════════════════════════════════════════════════════
# emcee sampler
# ═══════════════════════════════════════════════════════════════════════════

def run_emcee(
    x_axis: np.ndarray,
    datasets: Dict[float, Tuple[np.ndarray, np.ndarray]],
    J_P_base: float,
    beta: float,
    priors: Dict[str, Tuple[float, float]],
    noise_sigma: float,
    n_walkers: int = 32,
    n_steps: int = 10_000,
    burn_in: int = 2_000,
    use_pool: bool = False,
) -> Dict[str, np.ndarray]:
    """
    emcee affine-invariant ensemble sampler for (Da_w, Da_p).
    """
    if emcee is None:
        raise RuntimeError("emcee is not installed. Install with `pip install emcee`.")

    crates = np.array(sorted(datasets.keys()), dtype=float)
    X_obs = [datasets[c][0] for c in crates]
    V_obs = [datasets[c][1] for c in crates]
    bounds = (priors["Da_w"], priors["Da_p"])

    log_prob = LogProbDaWDaP(
        x_axis, crates, X_obs, V_obs,
        J_P_base, beta, bounds, noise_sigma,
    )

    ndim = 2
    rng = np.random.default_rng(1234)
    lo_w, hi_w = priors["Da_w"]
    lo_p, hi_p = priors["Da_p"]
    p0 = np.column_stack([
        rng.uniform(lo_w, hi_w, n_walkers),
        rng.uniform(lo_p, hi_p, n_walkers),
    ])

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

    chain = sampler.get_chain(flat=True)
    acc = np.mean(sampler.acceptance_fraction)
    print(f"  emcee acceptance rate: {acc:.3f}")

    return {
        "samples_Da_w": chain[:, 0],
        "samples_Da_p": chain[:, 1],
        "accept_rate": np.array([acc]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def corner_plot(
    samples: Dict[str, np.ndarray],
    true_values: Dict[str, float] | None = None,
    bins: int = 40,
    figsize: Tuple[int, int] = (10, 10),
    out_path: Path | None = None,
):
    """2x2 corner plot for (Da_w, Da_p)."""
    apply_publication_style_serif()

    keys = ["samples_Da_w", "samples_Da_p"]
    labels = [r"$Da_{\mathrm{wiring}}$", r"$Da_{\mathrm{process}}$"]
    true_keys = ["Da_w", "Da_p"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[keys[i]], bins=bins, density=True,
                        alpha=0.8, edgecolor="white", color="#1f77b4")
                ax.set_xlabel(labels[i])
                ax.set_ylabel("Density")
                if true_values and true_keys[i] in true_values:
                    ax.axvline(true_values[true_keys[i]], color="k",
                               ls="--", lw=1.5)
            else:
                ax.scatter(
                    samples[keys[j]], samples[keys[i]],
                    s=2, alpha=0.15, c="#1f77b4", rasterized=True,
                )
                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])
                if true_values:
                    if true_keys[j] in true_values and true_keys[i] in true_values:
                        ax.axvline(true_values[true_keys[j]], color="k",
                                   ls="--", lw=1, alpha=0.6)
                        ax.axhline(true_values[true_keys[i]], color="k",
                                   ls="--", lw=1, alpha=0.6)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig, axes


def plot_noise_vs_mcmc(
    mcmc_chain: Dict[str, np.ndarray],
    true_values: Dict[str, float],
    noise_sigma: float,
    n_noise: int = 5000,
    out_path: Path | None = None,
):
    """Side-by-side: input-noise spread vs MCMC posterior for Da_w and Da_p."""
    apply_publication_style_serif()

    rng = np.random.default_rng(99)
    Da_w_noise = true_values["Da_w"] * (1.0 + rng.normal(0, noise_sigma, n_noise))
    Da_p_noise = true_values["Da_p"] * (1.0 + rng.normal(0, noise_sigma, n_noise))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, noise, label in [
        (axes[0], "samples_Da_w", Da_w_noise, r"$Da_{\mathrm{wiring}}$"),
        (axes[1], "samples_Da_p", Da_p_noise, r"$Da_{\mathrm{process}}$"),
    ]:
        ax.hist(noise, bins=40, density=True, alpha=0.55,
                color="#ff7f0e", edgecolor="white", label="Input noise")
        ax.hist(mcmc_chain[key], bins=40, density=True, alpha=0.55,
                color="#1f77b4", edgecolor="white", label="MCMC posterior")
        tv_key = key.replace("samples_", "")
        ax.axvline(true_values[tv_key], color="k", ls="--", lw=2,
                   label="True value")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig, axes


def plot_posterior_overlay(
    x_axis: np.ndarray,
    datasets: Dict[float, Tuple[np.ndarray, np.ndarray]],
    mcmc_chain: Dict[str, np.ndarray],
    J_P_base: float,
    beta: float,
    crates: list[float],
    n_draws: int = 200,
    out_path: Path | None = None,
):
    """Overlay posterior VQ draws on synthetic data."""
    apply_publication_style_serif()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0.5: "tab:blue", 1.0: "tab:orange", 2.0: "tab:green",
              3.0: "tab:red", 5.0: "tab:purple"}

    for c, (X, V) in datasets.items():
        col = colors.get(c, "tab:gray")
        ax.plot(X, V, ".", color=col, alpha=0.6, label=f"{c}C data")

    idxs = np.random.choice(
        len(mcmc_chain["samples_Da_w"]),
        size=min(n_draws, len(mcmc_chain["samples_Da_w"])),
        replace=False,
    )
    for idx in idxs:
        Da_w = mcmc_chain["samples_Da_w"][idx]
        Da_p = mcmc_chain["samples_Da_p"][idx]
        for c in crates:
            J_P = J_P_base / c
            Xp, Vp = _predict_vq_wrap(Da_w, Da_p, J_P, x_axis, beta)
            col = colors.get(c, "tab:gray")
            ax.plot(Xp, Vp, "-", color=col, alpha=0.03)

    ax.set_xlabel("Fractional Capacity")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main(C_rate: float = 1.0):
    np.random.seed(42)

    params = ElectrodeParameters()
    true_Da_w = params.Da_w()
    true_Da_p = params.Da_p()
    J_P_base = params.J_P(C_rate)
    beta = params.beta()

    print(f"  ElectrodeParameters @ {C_rate}C:")
    print(f"    Da_w={true_Da_w:.4f}  Da_p={true_Da_p:.4f}  "
          f"J_P={J_P_base:.4f}  beta={beta:.4f}")

    x_axis = np.linspace(0.3, 0.95, 200)
    crates = [1.0]
    noise_sigma = 0.05

    datasets = generate_synthetic_dataset(
        x_axis, true_Da_w, true_Da_p, J_P_base, beta, crates, noise_sigma,
    )

    prior_margin = 5.0
    priors = {
        "Da_w": (true_Da_w / prior_margin, true_Da_w * prior_margin),
        "Da_p": (true_Da_p / prior_margin, true_Da_p * prior_margin),
    }

    print("  Running emcee ...")
    mcmc_chain = run_emcee(
        x_axis, datasets, J_P_base, beta, priors, noise_sigma,
        n_walkers=48, n_steps=8000, burn_in=2000, use_pool=True,
    )

    true_vals = {"Da_w": true_Da_w, "Da_p": true_Da_p}

    out_dir = Path("fit_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    corner_plot(
        mcmc_chain, true_values=true_vals,
        out_path=out_dir / "mcmc_corner_DawDap.png",
    )

    plot_noise_vs_mcmc(
        mcmc_chain, true_vals, noise_sigma,
        out_path=out_dir / "noise_vs_mcmc_DawDap.png",
    )

    plot_posterior_overlay(
        x_axis, datasets, mcmc_chain, J_P_base, beta, crates,
        out_path=out_dir / "mcmc_overlay_DawDap.png",
    )

    np.savez(
        out_dir / "mcmc_chain_DawDap.npz",
        samples_Da_w=mcmc_chain["samples_Da_w"],
        samples_Da_p=mcmc_chain["samples_Da_p"],
        accept_rate=mcmc_chain["accept_rate"],
    )

    med_w = np.median(mcmc_chain["samples_Da_w"])
    med_p = np.median(mcmc_chain["samples_Da_p"])
    print(f"\n  Posterior median  Da_w={med_w:.4f}  (true {true_Da_w:.4f})")
    print(f"  Posterior median  Da_p={med_p:.4f}  (true {true_Da_p:.4f})")


if __name__ == "__main__":
    main()
