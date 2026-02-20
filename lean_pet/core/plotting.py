"""
Shared plotting utilities: publication styles, legend builders, RMSE helpers.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def apply_publication_style_serif():
    """
    Configure matplotlib for Journal of Electrochemical Society style
    (serif font, no top/right spines).
    """
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 20,
        "font.family": "serif",
        "text.usetex": False,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.25,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })


def apply_publication_style_sans():
    """
    Configure matplotlib for publication style with sans-serif font
    (Arial / Helvetica).
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
        "axes.linewidth": 1.0,
        "axes.labelsize": 22,
        "axes.labelweight": "normal",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.fontsize": 18,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_impedance_rmse(
    Z_real_exp: np.ndarray,
    Z_imag_exp: np.ndarray,
    Z_real_theory: np.ndarray,
    Z_imag_theory: np.ndarray,
) -> tuple[float, float, float]:
    """
    RMSE between experimental and theoretical impedance spectra.

    Returns ``(rmse_real, rmse_imag, rmse_combined)``.  Experimental
    imaginary data is assumed to follow the Nyquist sign convention
    (positive = capacitive).
    """
    rmse_real = float(np.sqrt(np.mean((Z_real_exp - Z_real_theory) ** 2)))
    rmse_imag = float(np.sqrt(np.mean((-Z_imag_exp - Z_imag_theory) ** 2)))
    rmse_comb = float(np.sqrt(np.mean(
        (Z_real_exp - Z_real_theory) ** 2 + (-Z_imag_exp - Z_imag_theory) ** 2
    )))
    return rmse_real, rmse_imag, rmse_comb


def make_simulation_legend(
    include_mpet: bool = True,
    fontsize: int = 26,
) -> list[Line2D]:
    """
    Build the standard three-entry legend (PyBaMM ○, MPET △, Analytical —).
    """
    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="gray",
            markeredgecolor="black", markersize=14, linestyle="None",
            label="PyBaMM",
        ),
    ]
    if include_mpet:
        handles.append(
            Line2D(
                [0], [0], marker="^", color="w", markerfacecolor="gray",
                markeredgecolor="black", markersize=14, linestyle="None",
                label="MPET",
            )
        )
    handles.append(
        Line2D([0], [0], color="black", lw=3, label="Analytical")
    )
    return handles


def add_colored_labels(
    ax: plt.Axes,
    header: str,
    labels: list[str],
    colors: list,
    x_start: float = 0.58,
    y_header: float = 0.63,
    fontsize: int = 26,
):
    """
    Place a header string and a row of coloured labels on an axes in
    axes-fraction coordinates (used for C-rate / overpotential annotations).
    """
    ax.text(
        x_start, y_header, header, color="black", fontsize=fontsize,
        va="top", ha="left", transform=ax.transAxes, fontweight="normal",
    )
    y_row = y_header - 0.08
    x_offset = x_start
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for idx, (lbl, col) in enumerate(zip(labels, colors)):
        text_str = lbl + ("," if idx < len(labels) - 1 else "")
        t = ax.text(
            x_offset, y_row, text_str, color=col, fontsize=fontsize,
            va="top", ha="left", transform=ax.transAxes, fontweight="normal",
        )
        bbox = t.get_window_extent(renderer=renderer)
        inv = ax.transAxes.inverted()
        bbox_data = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        text_width = bbox_data[1, 0] - bbox_data[0, 0]
        x_offset += text_width * 1.05

