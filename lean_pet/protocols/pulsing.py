#!/usr/bin/env python3
"""
Pulsing protocol — Current vs. Time after a voltage step.

Runs PyBaMM DFN simulations with CIET/MHC kinetics, loads MPET reference
data, and overlays the analytical I-t prediction.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lean_pet.core.ocv import NMC532_Colclasure20, NMC532_Colclasure20_deriv
from lean_pet.core.kinetics import ecd_mhc
from lean_pet.core.analytical_models import predict_current_vs_time, _get_AB
from lean_pet.core.parameters import ElectrodeParameters, F, V_T
from lean_pet.core.plotting import (
    apply_publication_style_sans,
    make_simulation_legend,
    add_colored_labels,
    calculate_rmse,
)


# ── PyBaMM simulation ─────────────────────────────────────────────────────

def run_pybamm_simulations(
    params: ElectrodeParameters,
    kinetics_class=None,
    overpotentials: list[float] | None = None,
    initial_stoichiometry: float = 0.5,
):
    """
    Run PyBaMM voltage-step (hold) simulations.

    Returns ``(results_list, pybamm_params_dict)``.
    """
    import pybamm

    if overpotentials is None:
        overpotentials = [0.025, 0.05, 0.1]

    p = params
    ocv_init = NMC532_Colclasure20(initial_stoichiometry)
    voltage_steps = [ocv_init - eta for eta in overpotentials]

    print("=" * 70)
    kin_name = kinetics_class.__name__ if kinetics_class else "Standard Linear"
    print(f"Running PyBaMM pulsing simulations — {kin_name}")
    print("=" * 70)

    options = {
        "working electrode": "positive",
        "SEI": ("none", "none"),
        "surface form": "differential",
        "particle": "uniform profile",
        "intercalation kinetics": "linear",
    }

    E_r = 13000
    R_GAS = 8.314
    m_ref = p.k0 if kinetics_class else 0.1 / (p.c_s_max * p.c0 ** 0.5)

    def nmc532_ocp(sto):
        return NMC532_Colclasure20(sto)

    def custom_ecd(c_e, c_s_surf, c_s_max, T):
        arrhenius = pybamm.exp(E_r / R_GAS * (1 / 298.15 - 1 / T))
        if kinetics_class:
            return m_ref
        return m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_s_max - c_s_surf) ** 0.5

    def kappa_func(c_e, T):
        # PyBaMM passes symbolic objects — use raw arithmetic, not np.asarray
        k00, k01, k02 = -8.2488, 0.053248, -0.000029871
        k10, k11, k12 = 0.26235, -0.0093063, 0.000008069
        k20, k21 = 0.22002, -0.0001765
        c = c_e / 1000.0
        out = c * (k00 + k01 * T + k02 * T ** 2
                   + k10 * c + k11 * c * T + k12 * c * T ** 2
                   + k20 * c ** 2 + k21 * c ** 2 * T) ** 2
        return out * 0.1  # mS/cm → S/m

    param_vals = pybamm.ParameterValues({
        "Electrode height [m]": p.height,
        "Electrode width [m]": p.width,
        "Negative electrode thickness [m]": 50e-6,
        "Separator thickness [m]": 5e-6,
        "Positive electrode thickness [m]": p.L,
        "Negative current collector thickness [m]": 0,
        "Positive current collector thickness [m]": 0,
        "Negative electrode porosity": 0,
        "Negative electrode exchange-current density [A.m-2]": 1e8,
        "Negative electrode OCP [V]": 0.0,
        "Negative electrode conductivity [S.m-1]": 1e8,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode reorganization energy [eV]": 0.112,
        "Negative electrode reorganization energy [eV]": 0.01,
        "Positive electrode porosity": p.poros,
        "Positive electrode active material volume fraction": p.eps_am,
        "Positive particle radius [m]": p.R_p,
        "Positive electrode Bruggeman coefficient (electrolyte)": p.BruggExp,
        "Positive electrode Bruggeman coefficient (electrode)": p.BruggExp,
        "Positive electrode exchange-current density [A.m-2]": custom_ecd,
        "Positive electrode OCP [V]": nmc532_ocp,
        "Positive electrode conductivity [S.m-1]": p.sigma_s,
        "Positive particle diffusivity [m2.s-1]": 1e-10,
        "Maximum concentration in positive electrode [mol.m-3]": p.c_s_max,
        "Positive electrode density [kg.m-3]": 1000,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        "Positive electrode double-layer capacity [F.m-2]": p.C_DL,
        "Separator porosity": 1.0,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 1000,
        "Separator specific heat capacity [J.kg-1.K-1]": 1000,
        "Separator thermal conductivity [W.m-1.K-1]": 1.0,
        "Initial concentration in electrolyte [mol.m-3]": p.c0,
        "Cation transference number": p.t_plus,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": p.D_eff,
        "Cation diffusivity [m2.s-1]": p.Dp,
        "Anion diffusivity [m2.s-1]": p.Dm,
        "Electrolyte conductivity [S.m-1]": kappa_func,
        "Exchange-current density for lithium metal electrode [A.m-2]": 1e6,
        "Reference temperature [K]": p.T_ref,
        "Ambient temperature [K]": p.T_ref,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 4.5,
        "Open-circuit voltage at 100% SOC [V]": 4.5,
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Initial temperature [K]": p.T_ref,
        "Current function [A]": 1,
        "Nominal cell capacity [A.h]": p.nominal_capacity,
        "SEI partial molar volume [m3.mol-1]": 9.585e-04,
        "Initial concentration in negative electrode [mol.m-3]": 0.99 * 10000,
        "Initial concentration in positive electrode [mol.m-3]": initial_stoichiometry * p.c_s_max,
    })

    results = []
    skip_points = 125

    for eta, v_step in zip(overpotentials, voltage_steps):
        print(f"\n  Hold at {v_step:.3f} V  (η = {eta:.3f} V)")
        model = pybamm.lithium_ion.DFN(options=options, build=False)

        if kinetics_class:
            param_obj = model.param
            new_kin = kinetics_class(param_obj, "positive", "lithium-ion main", model.options, "primary")
            model.submodels["positive primary interface"] = new_kin

        model.build_model()
        experiment = pybamm.Experiment([f"Hold at {v_step}V until C/100"])
        sim = pybamm.Simulation(model, parameter_values=param_vals, experiment=experiment)

        try:
            sol = sim.solve()
            time = sol["Time [s]"].entries[skip_points:]
            current = sol["Current [A]"].entries[skip_points:]
            voltage = sol["Voltage [V]"].entries[skip_points:]
            time = time - time[0]

            filling_raw = sol["Positive particle stoichiometry"].entries
            filling = (filling_raw.mean(axis=tuple(range(filling_raw.ndim - 1)))
                       if filling_raw.ndim > 1 else filling_raw)
            filling = filling[skip_points:]

            results.append({
                "v_step": v_step,
                "overpotential": eta,
                "time": time,
                "current": np.abs(current),
                "voltage": voltage,
                "cathode_filling": np.mean(filling),
            })
            print(f"    ✓ I₀={abs(current[0]):.4e} A → I_end={abs(current[-1]):.4e} A")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    pybamm_params = {
        "L": p.L, "poros": p.poros, "R_p": p.R_p,
        "kappa_eff": p.kappa_ref, "D_eff": p.D_eff,
        "sigma_s_c": p.sigma_s, "P_L_c": p.P_L,
        "m_ref": m_ref, "R_film": p.R_film,
        "c_s_max": p.c_s_max,
    }
    return results, pybamm_params


# ── MPET data loading ─────────────────────────────────────────────────────

def load_mpet_data(folder_path: str) -> Tuple[list[dict], dict]:
    """Load MPET pulsing data from ``.mat`` files."""
    print(f"\nLoading MPET pulsing data from {folder_path}")

    v_set_data = {}
    for dirpath, _, filenames in os.walk(folder_path):
        match = re.search(r"Vset=([\d.]+)", dirpath)
        if match:
            v_set = float(match.group(1))
            for fn in filenames:
                if fn.endswith(".mat"):
                    v_set_data[v_set] = os.path.join(dirpath, fn)
                    break

    L = 100e-6
    c_s_max = 2.9869e28 * 1.602e-19 / F
    P_L_c = 0.69
    area = 1e-4

    mpet_results = []
    for v_set in sorted(v_set_data):
        data = scipy.io.loadmat(v_set_data[v_set])
        if "phi_applied_times" not in data:
            continue

        t = np.squeeze(data["phi_applied_times"]) * (L / 1e-4) ** 2 * 31.168
        I_dim = np.squeeze(data["current"])
        tau = (L ** 2) / (1e-4 ** 2) * 31.168
        I_char = F * 0.5 * c_s_max * P_L_c * L * area / tau
        I = np.abs(I_dim) * I_char

        X = data["ffrac_c"][0]
        X = (X[-1] + X[0]) / 2.0
        V_app = 3.993 - v_set
        t_shifted = t - t[0]

        mpet_results.append({
            "v_set": v_set, "time": t_shifted, "current": I,
            "X": X, "V_app": V_app,
        })
        print(f"  Loaded V_set={v_set:.3f} V")

    poros = 0.5
    R_p = 500e-9
    a_p = 3.0 / R_p
    k0 = 0.1
    mpet_params = {
        "J_P": k0 * a_p / (2.9869e28 * 1.602e-19),
        "Da_w": k0 * a_p * (1 - poros) * L ** 2 / (poros ** 1.5 * 0.1) / V_T,
    }
    return mpet_results, mpet_params


# ── Combined plot ─────────────────────────────────────────────────────────

def create_pulsing_plot(
    pybamm_results: list[dict],
    pybamm_params: dict,
    mpet_results: list[dict] | None = None,
    save_path: str = "combined_It_plot_ECIT.png",
):
    """Create the combined I-t comparison plot."""
    apply_publication_style_sans()
    plt.rcParams.update({"font.size": 24})

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.set_xlabel("Time (s)", fontsize=28)
    ax.set_ylabel("Current (A)", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_xlim(right=1000)

    # Colour map keyed by overpotential
    all_etas = sorted({r["overpotential"] for r in pybamm_results})
    cmap = plt.cm.coolwarm
    n = len(all_etas) + 1
    colors = {eta: cmap((i + 1) / (n - 1)) for i, eta in enumerate(all_etas)}

    pp = pybamm_params

    for result in pybamm_results:
        eta = result["overpotential"]
        v_step = result["v_step"]
        time = result["time"]
        current = result["current"]
        color = colors[eta]
        X_avg = 0.5

        # Scatter
        idx = np.linspace(0, len(time) - 1, 20, dtype=int)
        ax.scatter(time[idx], current[idx], color=color, s=100, alpha=0.7,
                   edgecolors="black", linewidths=1.0, marker="o")

        # Analytical prediction
        V_app = v_step - NMC532_Colclasure20(X_avg)
        sig_eff_el = pp["sigma_s_c"] * ((1 - pp["poros"]) ** 1.5)
        kap_eff = pp["kappa_eff"] * (pp["poros"] ** 1.5)
        sigma_eff = (sig_eff_el ** -1 + kap_eff ** -1) ** -1
        a_p = (3 / pp["R_p"]) * (1 - pp["poros"]) * pp["P_L_c"]
        k0 = pp["m_ref"]

        J_P = k0 * a_p / (pp["P_L_c"] * F * pp["c_s_max"] * (1 - pp["poros"]))
        Da_w = k0 * a_p * pp["L"] ** 2 / sigma_eff / V_T

        t_pred, I_pred = predict_current_vs_time(
            NMC532_Colclasure20, NMC532_Colclasure20_deriv,
            X_avg, V_app, Da_w, J_P, k0, pp["R_film"], t_max=max(time),
        )
        if len(current) > 0 and len(I_pred) > 0:
            scale = max(abs(current)) / max(abs(I_pred))
            ax.plot(t_pred, abs(I_pred) * scale, "-", color=color, linewidth=3, alpha=0.9)

    # MPET overlay
    if mpet_results:
        eta_to_vstep = {r["overpotential"]: round(r["v_step"], 2) for r in pybamm_results}
        for result in mpet_results:
            v_set_r = round(result["v_set"], 2)
            matched_eta = next(
                (eta for eta, vs in eta_to_vstep.items() if abs(vs - v_set_r) < 0.01), None
            )
            color = colors.get(matched_eta, "black")
            idx = np.linspace(0, len(result["time"]) - 1, 30, dtype=int)
            ax.scatter(result["time"][idx], result["current"][idx],
                       color=color, s=120, alpha=0.7, edgecolors="black",
                       linewidths=1.0, marker="^")

    legend_handles = make_simulation_legend(include_mpet=bool(mpet_results))
    ax.legend(handles=legend_handles, fontsize=26, frameon=False, loc="upper right")
    eta_labels = [f"{eta:g}" for eta in all_etas]
    add_colored_labels(ax, "Voltage Step:", eta_labels, [colors[e] for e in all_etas])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"✓ Plot saved: {save_path}")
    plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    from lean_pet.core.kinetics import CathodeKineticsCIET

    params = ElectrodeParameters()
    KINETICS = CathodeKineticsCIET

    pybamm_results, pybamm_params = run_pybamm_simulations(params, kinetics_class=KINETICS)

    mpet_folder = "/home/shakulp/Desktop/mpet_scaling_test/mpet/store/NMC_pulsing_half_CIET/"
    mpet_results, mpet_params = load_mpet_data(mpet_folder) if os.path.exists(mpet_folder) else ([], {})

    if pybamm_results or mpet_results:
        create_pulsing_plot(pybamm_results, pybamm_params, mpet_results)


if __name__ == "__main__":
    main()

