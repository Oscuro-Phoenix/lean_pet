#!/usr/bin/env python3
"""
3-D RMSE analysis over (σ_s, C-rate, c₀) using MPET outputs.

Iterates through MPET simulation folders whose names encode the three
swept parameters, computes the analytical VQ prediction for each, and
produces a 3-D iso-surface plot of the RMSE.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import interp1d

from lean_pet.core.ocv import NMC532_Colclasure20
from lean_pet.core.analytical_models import predict_vq
from lean_pet.core.electrolyte import electrolyte_conductivity
from lean_pet.core.parameters import F, V_T
from lean_pet.core.plotting import apply_publication_style_serif, calculate_rmse


# ── Folder-name parsing ───────────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> Tuple[float, float, float]:
    """
    Extract ``(sigma_s_c, C_rate, c0)`` from a folder name like
    ``sigma_s_c=1.0000e-01_Crate=1.0556e+00_c0=1.0556e+03``.
    """
    sigma_match = re.search(r"sigma_s_c=([\d\.e\-\+]+)", folder_name)
    crate_match = re.search(r"Crate=([\d\.e\-\+]+)", folder_name)
    c0_match = re.search(r"c0=([\d\.e\-\+]+)", folder_name)

    if not (sigma_match and crate_match and c0_match):
        raise ValueError(f"Cannot parse parameters from: {folder_name}")

    return (
        float(sigma_match.group(1)),
        float(crate_match.group(1)),
        float(c0_match.group(1)),
    )


# ── Dimensionless-number calculation ──────────────────────────────────────

def calculate_vq_parameters(
    sigma_s_c: float,
    C_rate: float,
    c0: float,
    k0_ref: float = 5.0,
    L: float = 100e-6,
    poros: float = 0.5,
    R_film: float = 0.0,
) -> dict:
    """
    Derive the dimensionless groups from physical parameters.

    Returns a dict with keys:
    ``Da_w, Da_w_sigma, Da_w_kappa, Da_p, J_P, k0, R_film``.
    """
    R_p = 500e-9
    P_L = 0.69
    t_plus = 0.38
    BruggExp = 1.5
    c_s_max = 2.9869e28 * 1.6e-19 / F

    Dp = 2.2e-10
    Dm = 2.94e-10
    D_eff = 2.0 * Dp * Dm / (Dp + Dm)

    kappa_ref = float(electrolyte_conductivity(c0, 298.15))
    kappa_eff = kappa_ref * poros ** BruggExp

    eps_am = (1 - poros) * P_L
    a_p = 3.0 * eps_am / R_p

    sigma_eff_el = sigma_s_c * (eps_am ** BruggExp)

    Da_p = k0_ref * a_p * L ** 2 * (1 - t_plus) / (poros ** BruggExp * c0 * F * D_eff)
    Da_w_sigma = k0_ref * a_p * L ** 2 / ((1 - poros) ** BruggExp * sigma_s_c * V_T)
    Da_w_kappa = k0_ref * a_p * L ** 2 / (poros ** BruggExp * kappa_ref * V_T)
    Da_w = Da_w_sigma + Da_w_kappa

    J_P = k0_ref * a_p * 3600.0 / (P_L * C_rate * (1 - poros) * F * c_s_max)

    return {
        "Da_w": Da_w,
        "Da_w_sigma": Da_w_sigma,
        "Da_w_kappa": Da_w_kappa,
        "Da_p": Da_p,
        "J_P": J_P,
        "k0": k0_ref,
        "R_film": R_film,
    }


# ── Data loading ──────────────────────────────────────────────────────────

def load_simulation_data(data_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MPET ``output_data.mat`` and return ``(capacity, voltage)``."""
    mat_file = os.path.join(data_folder, "output_data.mat")
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Not found: {mat_file}")
    data = scipy.io.loadmat(mat_file)
    capacity = np.squeeze(data["ffrac_c"])
    voltage = NMC532_Colclasure20(capacity[0]) - V_T * np.squeeze(data["phi_applied"])
    return capacity, voltage


# ── Single-folder comparison ──────────────────────────────────────────────

def compare_single_simulation(
    data_folder: str,
    k0_ref: float = 5.0,
    R_film: float = 0.0,
) -> dict | None:
    """
    Compare one MPET simulation to the analytical VQ model.

    Returns a result dict (or ``None`` on failure).
    """
    folder_name = os.path.basename(data_folder)
    try:
        sigma_s_c, C_rate, c0 = parse_folder_name(folder_name)
        params = calculate_vq_parameters(sigma_s_c, C_rate, c0, k0_ref, R_film=R_film)

        capacity_true, voltage_true = load_simulation_data(data_folder)

        capacity_grid = np.linspace(capacity_true.min(), capacity_true.max(), 100)
        Da_lim = params["Da_p"] / params["J_P"]
        capacity_pred, voltage_pred = predict_vq(
            NMC532_Colclasure20,
            capacity_grid,
            params["Da_w"],
            params["Da_w_sigma"],
            params["Da_w_kappa"],
            params["Da_p"],
            Da_lim,
            params["J_P"],
            params["k0"],
            params["R_film"],
        )

        voltage_pred_interp = interp1d(
            capacity_pred, voltage_pred, bounds_error=False, fill_value="extrapolate"
        )(capacity_true)

        rmse = calculate_rmse(voltage_true, voltage_pred_interp)

        return {
            "folder": folder_name,
            "sigma_s_c": sigma_s_c,
            "Crate": C_rate,
            "c0": c0,
            "Da_w": params["Da_w"],
            "Da_p": params["Da_p"],
            "J_P": params["J_P"],
            "rmse": rmse,
        }
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        return None


# ── Batch analysis ────────────────────────────────────────────────────────

def analyze_all_simulations(
    sim_data_folder: str,
    k0_ref: float = 5.0,
    R_film: float = 0.0,
    max_crate: float = 3.0,
) -> pd.DataFrame:
    """Iterate through every subfolder and collect RMSE results."""
    if not os.path.exists(sim_data_folder):
        raise FileNotFoundError(f"Not found: {sim_data_folder}")

    subdirs = sorted(
        d
        for d in os.listdir(sim_data_folder)
        if os.path.isdir(os.path.join(sim_data_folder, d))
    )
    print(f"Found {len(subdirs)} simulation folders …")

    results = []
    nan_count = 0
    for subdir in subdirs:
        result = compare_single_simulation(
            os.path.join(sim_data_folder, subdir), k0_ref=k0_ref, R_film=R_film
        )
        if result is None:
            continue
        if result["Crate"] > max_crate:
            continue
        if np.isnan(result["rmse"]):
            nan_count += 1
        else:
            results.append(result)
            print(f"  {subdir}: RMSE = {result['rmse']:.6f}")

    if nan_count:
        print(f"\n⚠ {nan_count} simulations had NaN RMSE and were excluded.")

    return pd.DataFrame(results)


# ── 3-D plotting ──────────────────────────────────────────────────────────

def plot_3d_rmse(
    df: pd.DataFrame,
    save_path: str | None = None,
    levels: np.ndarray | None = None,
):
    """
    Create a rotatable 3-D iso-surface plot of RMSE in
    (σ_s, C-rate, c₀) space.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import LogLocator
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    if df.empty:
        print("No data to plot.")
        return

    apply_publication_style_serif()
    df = df.dropna(subset=["rmse"]).copy()

    sigmas = np.sort(df["sigma_s_c"].unique())
    crates = np.sort(df["Crate"].unique())
    c0s = np.sort(df["c0"].unique())

    if levels is None:
        qs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99])
        levels = np.percentile(df["rmse"].values, qs * 100)

    # Index mapping
    sigma_idx = {v: i for i, v in enumerate(sigmas)}
    crate_idx = {v: i for i, v in enumerate(crates)}
    c0_idx = {v: i for i, v in enumerate(c0s)}

    df["si"] = df["sigma_s_c"].map(sigma_idx)
    df["ci"] = df["c0"].map(c0_idx)
    df["ri"] = df["Crate"].map(crate_idx)

    # Interpolate onto regular grid
    gres = 10
    xi = np.linspace(0, len(sigmas) - 1, gres)
    yi = np.linspace(0, len(c0s) - 1, gres)
    zi = np.linspace(0, len(crates) - 1, gres)
    XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing="ij")

    points = df[["si", "ci", "ri"]].values
    values = df["rmse"].values
    grid_pts = np.column_stack([XI.ravel(), YI.ravel(), ZI.ravel()])
    rmse_grid = griddata(points, values, grid_pts, method="linear", fill_value=np.nan).reshape(XI.shape)

    nan_mask = np.isnan(rmse_grid)
    if nan_mask.any():
        rmse_nn = griddata(points, values, grid_pts, method="nearest").reshape(XI.shape)
        rmse_grid[nan_mask] = rmse_nn[nan_mask]

    rmse_grid = gaussian_filter(rmse_grid, sigma=2)

    # Plot
    from skimage.measure import marching_cubes

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    vmin = max(levels.min(), 1e-6)
    norm = LogNorm(vmin=vmin, vmax=levels.max())
    cmap = plt.cm.coolwarm

    for level in levels:
        try:
            verts, faces, _, _ = marching_cubes(rmse_grid, level=level, spacing=(1, 1, 1), allow_degenerate=False)
            verts[:, 0] *= (len(sigmas) - 1) / gres
            verts[:, 1] *= (len(c0s) - 1) / gres
            verts[:, 2] *= (len(crates) - 1) / gres
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces, color=cmap(norm(level)), alpha=0.85,
                            edgecolor="none", shade=True, antialiased=True)
        except Exception:
            pass

    ax.set_xlabel(r"Conductivity (S/m)", fontsize=24, labelpad=15)
    ax.set_ylabel(r"Electrolyte Conc. (M)", fontsize=24, labelpad=15)
    ax.set_zlabel(r"$C$-rate", fontsize=24, labelpad=30, rotation=90)
    ax.view_init(elev=25, azim=45)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0)
    ax.grid(True, alpha=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.05)
    cbar.set_label("RMSE", fontsize=22)
    cbar.ax.tick_params(labelsize=14)
    cbar.locator = LogLocator(base=10, subs=np.arange(1, 10), numticks=20)
    cbar.update_ticks()

    plt.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=1.2)
        print(f"✓ 3-D RMSE plot saved: {save_path}")
    plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    sim_data_folder = "/home/shakulp/Desktop/mpet_scaling_test/mpet/store/NMC_grid_CIET_dense"
    k0_ref = float(os.environ.get("K0_REF", "5.0"))
    R_film = float(os.environ.get("R_FILM", "0.0"))

    print("Starting 3-D VQ vs MPET comparison …")
    df = analyze_all_simulations(sim_data_folder, k0_ref=k0_ref, R_film=R_film)

    if df.empty:
        print("No valid simulations found!")
        return

    print(f"\nProcessed {len(df)} simulations — RMSE range: "
          f"{df['rmse'].min():.6f} – {df['rmse'].max():.6f}")

    results_csv = os.path.join(os.path.dirname(__file__), "..", "..", "vq_comparison_results_3d_ciet.csv")
    df.to_csv(results_csv, index=False)
    print(f"Results saved: {results_csv}")

    plot_3d_rmse(df, save_path="rmse_3d_scatter_ciet.png")


if __name__ == "__main__":
    main()

