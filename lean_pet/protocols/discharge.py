#!/usr/bin/env python3
"""
Discharge protocol — Voltage vs. Cathode Filling Fraction.

Runs PyBaMM DFN simulations at multiple C-rates with CIET/MHC kinetics,
loads MPET reference data, and overlays the analytical VQ prediction.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from lean_pet.core.ocv import NMC532_Colclasure20
from lean_pet.core.analytical_models import predict_vq
from lean_pet.core.parameters import ElectrodeParameters, F, V_T
from lean_pet.core.plotting import (
    apply_publication_style_sans,
    make_simulation_legend,
    add_colored_labels,
)


# ── PyBaMM simulation ─────────────────────────────────────────────────────

def run_pybamm_simulations(
    params: ElectrodeParameters,
    kinetics_class=None,
    C_rates: list[float] | None = None,
    cutoff_voltage: float = 3.0,
    initial_stoichiometry: float = 0.3,
):
    """
    Run PyBaMM DFN half-cell discharge simulations.

    Returns ``(results_list, pybamm_params_dict)``.
    """
    import pybamm

    if C_rates is None:
        C_rates = [0.5, 1.0, 1.5, 2.0]
    crate_labels = [f"{c}C" for c in C_rates]

    print("=" * 70)
    kin_name = kinetics_class.__name__ if kinetics_class else "Standard Linear"
    print(f"Running PyBaMM discharge simulations — {kin_name}")
    print("=" * 70)

    # --- Model options ---
    options = {
        "working electrode": "positive",
        "SEI": ("none", "constant"),
        "SEI film resistance": "distributed",
        "surface form": "differential",
        "particle": "uniform profile",
        "intercalation kinetics": "linear",
    }

    p = params  # shorthand

    E_r = 39570
    R_GAS = 8.314

    if kinetics_class:
        m_ref = p.k0
    else:
        m_ref = 0.1 / (p.c_s_max * p.c0 ** 0.5)

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

    R_film_cathode = p.R_film
    L_sei = 1e-3
    R_sei = R_film_cathode / L_sei if L_sei > 0 else 0.0

    param_vals = pybamm.ParameterValues({
        "Electrode height [m]": p.height,
        "Electrode width [m]": p.width,
        "Negative electrode thickness [m]": 50e-6,
        "Separator thickness [m]": 5e-6,
        "Positive electrode thickness [m]": p.L,
        "Negative current collector thickness [m]": 1e-6,
        "Positive current collector thickness [m]": 1e-6,
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
        "Positive electrode density [kg.m-3]": 3000,
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
        "Initial temperature [K]": p.T_ref,
        "Current function [A]": 1,
        "Nominal cell capacity [A.h]": p.nominal_capacity,
        "SEI partial molar volume [m3.mol-1]": 9.585e-04,
        "Initial concentration in negative electrode [mol.m-3]": 0.99 * 10000,
        "Initial concentration in positive electrode [mol.m-3]": initial_stoichiometry * p.c_s_max,
    })
    param_vals.update({
        "SEI resistivity [Ohm.m]": R_sei,
        "Initial SEI thickness [m]": L_sei,
    }, check_already_exists=False)

    results = []
    for i, C_rate in enumerate(C_rates):
        print(f"  {crate_labels[i]}: C_rate={C_rate}")
        model = pybamm.lithium_ion.DFN(options=options, build=False)

        if kinetics_class:
            param_obj = model.param
            new_kin = kinetics_class(param_obj, "positive", "lithium-ion main", model.options, "primary")
            model.submodels["positive primary interface"] = new_kin

        model.build_model()
        experiment = pybamm.Experiment([f"Discharge at {C_rate} C until {cutoff_voltage}V"])
        sim = pybamm.Simulation(model, parameter_values=param_vals, experiment=experiment)

        try:
            solution = sim.solve()
            voltage = solution["Voltage [V]"].entries
            time = solution["Time [s]"].entries
            filling_raw = solution["Positive particle stoichiometry"].entries
            filling = filling_raw.mean(axis=tuple(range(filling_raw.ndim - 1))) if filling_raw.ndim > 1 else filling_raw

            results.append({
                "C_rate": C_rate,
                "label": crate_labels[i],
                "voltage": voltage,
                "time": time,
                "cathode_filling": filling,
            })
            print(f"    ✓ Final V: {voltage[-1]:.3f} V")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    pybamm_params = {
        "L": p.L, "poros": p.poros, "R_p": p.R_p,
        "kappa_eff": p.kappa_ref, "D_eff": p.D_eff,
        "Dp": p.Dp, "Dm": p.Dm, "t_plus": p.t_plus,
        "c_s_max": p.c_s_max, "c_e_ref": p.c0,
        "sigma_s_c": p.sigma_s, "P_L_c": p.P_L,
        "m_ref": m_ref, "R_film": R_film_cathode,
    }
    return results, pybamm_params


# ── MPET data loading ─────────────────────────────────────────────────────

def load_mpet_data(folder_path: str) -> Tuple[list[dict], dict]:
    """Load MPET discharge data from ``.mat`` files."""
    print(f"\nLoading MPET discharge data from {folder_path}")

    crate_labels = ["0.5C", "1C", "1.5C", "2C"]
    mpet_results = []

    for crate in crate_labels:
        for dirpath, _, filenames in os.walk(folder_path):
            if crate in dirpath:
                for fn in filenames:
                    if fn.endswith(".mat"):
                        data = scipy.io.loadmat(os.path.join(dirpath, fn))
                        mpet_results.append({
                            "C_rate": float(crate.replace("C", "")),
                            "label": crate,
                            "time": np.squeeze(data["phi_applied_times"]),
                            "ffrac_c": np.squeeze(data["ffrac_c"]),
                            "voltage": np.squeeze(data["phi_applied"]),
                            "L": float(data.get("L_c", 100e-6)),
                            "poros": float(data.get("poros_c", 0.5)),
                        })
                        print(f"  Loaded {crate}")
                        break

    mpet_params = {"R_p": 500e-9, "k0": 10, "P_L_c": 0.69, "t_plus": 0.38,
                   "c_e_ref": 1000, "F": F, "D_eff": 2.5e-10}
    return mpet_results, mpet_params


# ── Combined plot ─────────────────────────────────────────────────────────

def create_discharge_plot(
    pybamm_results: list[dict],
    pybamm_params: dict,
    mpet_results: list[dict] | None = None,
    save_path: str = "combined_vq_plot.png",
):
    """Create the combined V–Q comparison plot."""
    apply_publication_style_sans()
    plt.rcParams.update({"font.size": 24})

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.set_xlabel("Cathode Filling Fraction", fontsize=28)
    ax.set_ylabel("Voltage (V)", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_ylim(top=4.3, bottom=3.3)

    crate_labels = [r["label"] for r in pybamm_results]
    cmap = plt.cm.coolwarm
    n = len(crate_labels)
    colors = {lbl: cmap(i / max(n - 1, 1)) for i, lbl in enumerate(crate_labels)}

    pp = pybamm_params  # shorthand

    for result in pybamm_results:
        lbl = result["label"]
        filling = result["cathode_filling"]
        voltage = result["voltage"]
        color = colors[lbl]

        # Scatter (sub-sampled)
        n_pts = 15
        uf = np.linspace(filling.min(), filling.max(), n_pts)
        uv = interp1d(filling, voltage, kind="linear", fill_value="extrapolate")(uf)
        ax.scatter(uf[1:], uv[1:], color=color, s=100, alpha=0.7,
                   edgecolors="black", linewidths=1.0, marker="o")

        # Analytical prediction
        if pp["m_ref"] > 0:
            C_rate = result["C_rate"]
            a_p = (3 / pp["R_p"]) * (1 - pp["poros"]) * pp["P_L_c"]
            k0 = pp["m_ref"]
            J_P = k0 * a_p * 3600 / (pp["P_L_c"] * C_rate * (1 - pp["poros"]) * F * pp["c_s_max"])
            Da_p = k0 * a_p * pp["L"] ** 2 * (1 - pp["t_plus"]) / (pp["poros"] ** 1.5 * pp["c_e_ref"] * F * pp["D_eff"])
            sig_eff_el = pp["sigma_s_c"] * ((1 - pp["poros"]) ** 1.5)
            kap_eff = pp["kappa_eff"] * (pp["poros"] ** 1.5)
            Da_w_s = k0 * a_p * pp["L"] ** 2 / sig_eff_el / V_T
            Da_w_k = k0 * a_p * pp["L"] ** 2 / kap_eff / V_T
            Da_w = Da_w_s + Da_w_k
            Da_lim = Da_p / J_P

            X_pred = np.linspace(filling.min() + 0.02, min(1.0, filling.max() + 0.02), 100)
            Xp, Vp = predict_vq(NMC532_Colclasure20, X_pred, Da_w, Da_w_s, Da_w_k,
                                Da_p, Da_lim, J_P, k0, pp["R_film"])
            ax.plot(Xp, Vp, "-", color=color, linewidth=3, alpha=0.9)

    # MPET overlay
    if mpet_results:
        for result in mpet_results:
            lbl = result["label"]
            color = colors.get(lbl, "black")
            ff = result["ffrac_c"]
            V = result["voltage"]
            n_pts = 20
            idx = np.linspace(0, len(ff) - 1, min(n_pts, len(ff)), dtype=int)
            ax.scatter(
                ff[idx][1:],
                (NMC532_Colclasure20(ff[0]) - V * V_T)[idx][1:],
                color=color, s=120, alpha=0.7, edgecolors="black",
                linewidths=1.0, marker="^",
            )

    legend_handles = make_simulation_legend(include_mpet=bool(mpet_results))
    ax.legend(handles=legend_handles, fontsize=26, frameon=False, loc="upper right")
    add_colored_labels(ax, "C-rate:", crate_labels, [colors[l] for l in crate_labels],
                       x_start=0.6, y_header=0.675, fontsize=24)

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

    mpet_folder = "/home/shakulp/Desktop/mpet_scaling_test/mpet/store/NMC_500nm_MHC"
    mpet_results, _ = load_mpet_data(mpet_folder) if os.path.exists(mpet_folder) else ([], {})

    if pybamm_results or mpet_results:
        create_discharge_plot(pybamm_results, pybamm_params, mpet_results)


if __name__ == "__main__":
    main()

