#!/usr/bin/env python3
"""
EIS protocol — Nyquist plots comparing analytical impedance to PyBaMM data.

Supports two sweep modes:
  * **Dap sweep** — vary j₀ prefactor (1×, 2×, 4×) → changes Dac, Dap, Daw together.
  * **Daw sweep** — vary σ_s (1×, 2×, 4×) → changes only Daw (and R_hf).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MultipleLocator

from lean_pet.core.analytical_models import calculate_eis_impedance
from lean_pet.core.parameters import ElectrodeParameters, F, V_T
from lean_pet.core.plotting import (
    apply_publication_style_sans,
    calculate_impedance_rmse,
)


# ── Data loading ──────────────────────────────────────────────────────────

def load_eis_csv(csv_path: str, verbose: bool = False):
    """
    Load EIS data from CSV (columns: ``freq``, ``real``, ``im``).

    Returns ``(freq, Z_real, Z_imag)`` or ``(None, None, None)`` on error.
    """
    try:
        df = pd.read_csv(csv_path)
        freq = df["freq"].values
        Z_real = df["real"].values
        Z_imag = df["im"].values
        if verbose:
            print(f"  Loaded {len(freq)} points from {csv_path}")
        return freq, Z_real, Z_imag
    except Exception as e:
        if verbose:
            print(f"  Error loading {csv_path}: {e}")
        return None, None, None


# ── Shared Nyquist-plot builder ───────────────────────────────────────────

def _nyquist_plot(
    ax: plt.Axes,
    data_entries: list[dict],
    params: ElectrodeParameters,
    freq_range: Tuple[float, float] = (0.001, 10000),
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
):
    """
    Plot one or more (data, model) pairs on a Nyquist diagram.

    Each entry in *data_entries* is a dict with keys:
        ``csv_path``, ``Dac``, ``Dap``, ``Daw``, ``R_hf``,
        ``color``, ``label``.
    """
    electrode_area = params.electrode_area
    min_f, max_f = freq_range

    for entry in data_entries:
        freq, Z_re, Z_im = load_eis_csv(entry["csv_path"], verbose=True)
        if freq is None:
            continue

        mask = (freq >= min_f) & (freq <= max_f)
        freq, Z_re, Z_im = freq[mask], Z_re[mask], Z_im[mask]

        # Sub-sample for scatter
        n_pts = min(40, len(Z_re))
        idx = np.linspace(0, len(Z_re) - 1, n_pts, dtype=int)
        ax.scatter(Z_re[idx], -Z_im[idx], s=80, alpha=0.7,
                   color=entry["color"], edgecolors="black", linewidths=0.5, zorder=3)

        # Analytical
        omega = 2.0 * np.pi * freq
        Zr_th, Zi_th = calculate_eis_impedance(
            entry["Dac"], entry["Dap"], entry["Daw"], omega, R_hf=entry["R_hf"],
        )
        Zr_th /= electrode_area
        Zi_th /= electrode_area
        ax.plot(Zr_th, Zi_th, "-", linewidth=2, color=entry["color"], zorder=2)

        rmse_r, rmse_i, rmse_c = calculate_impedance_rmse(Z_re, Z_im, Zr_th, Zi_th)
        print(f"  {entry['label']}  RMSE combined = {rmse_c:.4e}")

    # Formatting
    ax.set_xlabel(r"$Z^{\prime}$ / $\Omega\text{-}m^2$")
    ax.set_ylabel(r"$-Z^{\prime\prime}$ / $\Omega\text{-}m^2$")
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.set_xlim(xlim or (0, 35.0))
    ax.set_ylim(ylim or (0, 25.0))
    ax.set_aspect("equal", adjustable="box")

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=12, linestyle="None", label="PyBaMM"),
        Line2D([0], [0], color="black", linewidth=2, label="Analytical"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")


# ── Dap sweep (vary j₀ prefactor) ────────────────────────────────────────

def plot_eis_dap_sweep(
    params: ElectrodeParameters,
    data_folder: str,
    prefactors: list[int] | None = None,
    save_path: str | None = None,
):
    """
    Nyquist plot sweeping the j₀ prefactor (1×, 2×, 4×) which scales
    Dac, Dap, and Daw together.
    """
    if prefactors is None:
        prefactors = [1, 2, 4]

    apply_publication_style_sans()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#5B9BD5", "#F39C73", "#C5282F"]

    p = params
    j0_base = p.k0 / V_T

    entries = []
    j0_values = []
    for i, pf in enumerate(prefactors):
        j0 = j0_base * pf
        Dac = j0 / p.C_DL
        Dap = j0 * p.a_p * V_T / (F * p.eps_am * p.c_s_max)
        Daw = j0 * p.a_p * p.L ** 2 / p.sigma_eff
        j0_values.append(j0)
        entries.append({
            "csv_path": os.path.join(data_folder, f"eis_data_{pf}.csv"),
            "Dac": Dac, "Dap": Dap, "Daw": Daw, "R_hf": p.R_hf,
            "color": colors[i % len(colors)],
            "label": f"j0={j0:.1e}",
        })

    _nyquist_plot(ax, entries, params)

    # Annotate j₀ values
    renderer = fig.canvas.get_renderer()
    ax.text(0.1, 0.7, r"$j_0$:", transform=ax.transAxes, fontsize=18, va="top", ha="center")
    x_off = 0.05
    for i, (j0, col) in enumerate(zip(j0_values, colors)):
        txt = f"{j0:.1e}" + ("," if i < len(j0_values) - 1 else "")
        t = ax.text(x_off, 0.6, txt, color=col, fontsize=18, va="top", ha="left",
                    transform=ax.transAxes)
        plt.draw()
        bb = t.get_window_extent(renderer=renderer)
        inv = ax.transAxes.inverted()
        w = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])[1, 0] - inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])[0, 0]
        x_off += w * 1.05

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"✓ Plot saved: {save_path}")
    plt.show()


# ── Daw sweep (vary σ_s) ─────────────────────────────────────────────────

def plot_eis_daw_sweep(
    params: ElectrodeParameters,
    data_folder: str,
    prefactors: list[int] | None = None,
    save_path: str | None = None,
):
    """
    Nyquist plot sweeping σ_s (1×, 2×, 4×) which changes only Daw
    (and the R_hf ratio).  Dac and Dap stay constant.
    """
    if prefactors is None:
        prefactors = [1, 2, 4]

    apply_publication_style_sans()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#5B9BD5", "#F39C73", "#C5282F"]

    p = params
    j0 = p.k0 / V_T
    Dac = j0 / p.C_DL
    Dap = j0 * p.a_p * V_T / (F * p.eps_am * p.c_s_max)

    entries = []
    sigma_values = []
    for i, pf in enumerate(prefactors):
        sigma_s = p.sigma_s * pf
        sig_eff_el = sigma_s * (1 - p.poros) ** p.BruggExp
        sigma_eff = (sig_eff_el ** -1 + p.kappa_eff ** -1) ** -1
        R_hf = sigma_eff / p.kappa_eff
        Daw = j0 * p.a_p * p.L ** 2 / sigma_eff
        sigma_values.append(sigma_s)
        entries.append({
            "csv_path": os.path.join(data_folder, f"eis_data_{pf}.csv"),
            "Dac": Dac, "Dap": Dap, "Daw": Daw, "R_hf": R_hf,
            "color": colors[i % len(colors)],
            "label": f"σ_s={sigma_s:.1e}",
        })

    _nyquist_plot(ax, entries, params)

    # Annotate σ_s values
    renderer = fig.canvas.get_renderer()
    ax.text(0.1, 0.7, r"$\sigma_s$:", transform=ax.transAxes, fontsize=18, va="top", ha="center")
    x_off = 0.05
    for i, (sv, col) in enumerate(zip(sigma_values, colors)):
        txt = f"{sv:.1e}" + ("," if i < len(sigma_values) - 1 else "")
        t = ax.text(x_off, 0.6, txt, color=col, fontsize=18, va="top", ha="left",
                    transform=ax.transAxes)
        plt.draw()
        bb = t.get_window_extent(renderer=renderer)
        inv = ax.transAxes.inverted()
        w = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])[1, 0] - inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])[0, 0]
        x_off += w * 1.05

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"✓ Plot saved: {save_path}")
    plt.show()


# ── CLI entry points ──────────────────────────────────────────────────────

def main_dap():
    """Run the Dap (j₀) sweep EIS comparison."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eis_data = os.path.join(script_dir, "..", "..", "eis", "Dap_data_ECIT")
    params = ElectrodeParameters(k0=2.5)
    plot_eis_dap_sweep(params, data_folder=eis_data,
                       save_path="eis_multi_dap_comparison_ECIT.png")


def main_daw():
    """Run the Daw (σ_s) sweep EIS comparison."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eis_data = os.path.join(script_dir, "..", "..", "eis", "Daw_data_ECIT")
    params = ElectrodeParameters(sigma_s=5e-2, k0=5.0)
    plot_eis_daw_sweep(params, data_folder=eis_data,
                       save_path="eis_multi_daw_comparison_ECIT.png")


if __name__ == "__main__":
    main_dap()
    main_daw()

