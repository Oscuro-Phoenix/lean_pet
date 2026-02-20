import numpy as np
import pybamm
import matplotlib.pyplot as plt
import pandas as pd

import pybammeis

def NMC532_Colclasure20(x):
    """NMC532 OCP function"""
    OCV = (5.314735633000300E+00 +
           -3.640117692001490E+03*x**14.0 + 1.317657544484270E+04*x**13.0
           - 1.455742062291360E+04*x**12.0 - 1.571094264365090E+03*x**11.0
           + 1.265630978512400E+04*x**10.0 - 2.057808873526350E+03*x**9.0
           - 1.074374333186190E+04*x**8.0 + 8.698112755348720E+03*x**7.0
           - 8.297904604107030E+02*x**6.0 - 2.073765547574810E+03*x**5.0
           + 1.190223421193310E+03*x**4.0 - 2.724851668445780E+02*x**3.0
           + 2.723409218042130E+01*x**2.0 - 4.158276603609060E+00*x +
           -5.573191762723310E-04*np.exp(6.560240842659690E+00*x**4.148209275061330E+01)
           )
    return OCV

def electrolyte_conductivity(c_e, T):
    """
    Electrolyte conductivity as a function of concentration and temperature.
    Based on polynomial fit from experiments.
    c_e: electrolyte concentration [mol/m³]
    T: temperature [K]
    """
    # Polynomial coefficients
    (k00, k01, k02,
     k10, k11, k12,
     k20, k21) = (
        -8.2488, 0.053248, -0.000029871,
        0.26235, -0.0093063, 0.000008069,
        0.22002, -0.0001765)
    
    # Convert concentration from mol/m³ to mol/L (M)
    c = c_e / 1000.0
    
    # Calculate conductivity using polynomial model
    out = c * (k00 + k01*T + k02*T**2
               + k10*c + k11*c*T + k12*c*T**2
               + k20*c**2 + k21*c**2*T)**2  # mS/cm
    out *= 0.1  # Convert to S/m
    
    return out

# Setup PyBaMM model with options from plot_vq_combined.py
options = {
    "working electrode": "positive",
    "SEI": ("none", "none"),
    "SEI film resistance": "distributed",
    "surface form": "differential",
    "particle": "uniform profile",
    "intercalation kinetics": "linear",
}

model = pybamm.lithium_ion.DFN(options=options)

# Build parameter set from plot_vq_combined.py
sigma_s_c = 4e-1 # S/m
Dp = 2.2e-4  # m²/s (BOOSTED FROM 2.2e-10 to 2.2e-6)
Dm = 2.94e-4  # m²/s (BOOSTED FROM 2.94e-10 to 2.94e-6)
L_c = 100e-6  # m
poros_c = 0.5
mean_c = 500e-9  # m
c0 = 1000  # mol/m³
T_ref = 298.15  # K
L_s = 5e-6  # m
poros_s = 1.0
BruggExp_c = 1.5
BruggExp_s = 1.5

D_eff = 2*Dp*Dm/(Dp+Dm)
t_plus = 0.38 
F = 96485
R = 8.314

c_s_max_pos = 2.9869e28*1.6e-19/F  # mol/m³
D_s_pos = 1e-4 # m²/s
m_ref = 0.1/(c_s_max_pos*c0**0.5)
E_r = 39570  # J/mol

c_s_max_neg = 1000
L_n = 5e-6
poros_n = 0
L_cc_p = 1e-6
L_cc_n = 1e-6
height = 1e-2
width = 1e-2

def custom_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    arrhenius = pybamm.exp(E_r / R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

param = pybamm.ParameterValues({
    "Electrode height [m]": height,
    "Electrode width [m]": width,
    "Negative electrode thickness [m]": L_n,
    "Separator thickness [m]": L_s,
    "Positive electrode thickness [m]": L_c,
    "Negative current collector thickness [m]": L_cc_n,
    "Positive current collector thickness [m]": L_cc_p,
    "Negative electrode porosity": poros_n,
    "Negative electrode exchange-current density [A.m-2]": 1e6,
    "Negative electrode OCP [V]": 0.0,
    "Negative electrode conductivity [S.m-1]": 1e6,
    "Negative electrode double-layer capacity [F.m-2]": 1e-10,
    "Positive electrode porosity": poros_c,
    "Positive electrode active material volume fraction": 0.5,
    "Positive particle radius [m]": mean_c,
    "Positive electrode Bruggeman coefficient (electrolyte)": BruggExp_c,
    "Positive electrode Bruggeman coefficient (electrode)": BruggExp_c,
    "Positive electrode exchange-current density [A.m-2]": custom_exchange_current_density,
    "Positive electrode OCP [V]": NMC532_Colclasure20,
    "Positive electrode conductivity [S.m-1]": sigma_s_c,
    "Positive particle diffusivity [m2.s-1]": D_s_pos,
    "Maximum concentration in positive electrode [mol.m-3]": c_s_max_pos,
    "Positive electrode density [kg.m-3]": 3000,
    "Positive electrode OCP entropic change [V.K-1]": 0.0,
    "Positive electrode double-layer capacity [F.m-2]": 0.2,
    "Separator porosity": poros_s,
    "Separator Bruggeman coefficient (electrolyte)": BruggExp_s,
    "Separator density [kg.m-3]": 1000,
    "Separator specific heat capacity [J.kg-1.K-1]": 1000,
    "Separator thermal conductivity [W.m-1.K-1]": 1.0,
    "Initial concentration in electrolyte [mol.m-3]": c0,
    "Cation transference number": t_plus,
    "Thermodynamic factor": 1.0,
    "Electrolyte diffusivity [m2.s-1]": D_eff,
    "Electrolyte conductivity [S.m-1]": electrolyte_conductivity,
    "Exchange-current density for lithium metal electrode [A.m-2]": 1e6,
    "Reference temperature [K]": T_ref,
    "Ambient temperature [K]": T_ref,
    "Number of electrodes connected in parallel to make a cell": 1.0,
    "Number of cells connected in series to make a battery": 1.0,
    "Lower voltage cut-off [V]": 2.5,
    "Upper voltage cut-off [V]": 4.5,
    "Initial temperature [K]": T_ref,
    "Current function [A]": 1,
    "Nominal cell capacity [A.h]": height*width*L_c*(1-poros_c)*96485*c_s_max_pos/3600,
    "SEI partial molar volume [m3.mol-1]": 0,
    "Initial concentration in negative electrode [mol.m-3]": 0.99 * c_s_max_neg,
    "Initial concentration in positive electrode [mol.m-3]": 0.3 * c_s_max_pos,
})

# Set initial stoichiometry
initial_stoichiometry = 0.3
initial_concentration_pos = initial_stoichiometry * c_s_max_pos

param.update({
    "Initial concentration in positive electrode [mol.m-3]": initial_concentration_pos,
}, check_already_exists=False)

print("="*70)
print("PyBaMM-EIS Simulation with plot_vq_combined.py Parameters")
print("="*70)
print(f"\nKey Parameters:")
print(f"  Positive electrode thickness: {L_c*1e6:.1f} µm")
print(f"  Positive particle radius: {mean_c*1e9:.1f} nm")
print(f"  Positive electrode porosity: {poros_c}")
print(f"  Separator thickness: {L_s*1e6:.1f} µm")
print(f"  Electrolyte concentration: {c0} mol/m³")
print(f"  Cation transference number: {t_plus}")
print(f"  Initial stoichiometry: {initial_stoichiometry:.3f}")
print(f"  m_ref: {m_ref:.6e} A/m²/(mol/m³)^1.5")
print(f"  Expected k0: {m_ref * c_s_max_pos * (c0**0.5):.6e} A/m²")

# Create EIS simulation
eis_sim = pybammeis.EISSimulation(model, parameter_values=param)

# Choose frequencies and calculate impedance
frequencies = np.logspace(-3, 3, 1000)
print(f"\nCalculating impedance for {len(frequencies)} frequencies...")
sol = eis_sim.solve(frequencies)
print("✓ Impedance calculation complete")

# Save EIS data to CSV for later comparison
eis_data = {
    'freq': frequencies,
    'real': np.real(sol),
    'im': np.imag(sol)
}
df = pd.DataFrame(eis_data)
df.to_csv('eis_data_vq_params.csv', index=False)
print(f"✓ EIS data saved to 'eis_data_vq_params.csv'")

# Generate a Nyquist plot using the solution directly
fig, ax = plt.subplots(figsize=(8, 8))

pybammeis.nyquist_plot(
    sol,
    ax=ax,
    marker="o",
    markersize=6,
    color="blue",
    alpha=0.8,
    linewidth=2,
    label="DFN Model EIS (VQ Params)"
)

# Set axis labels with large font size
ax.set_xlabel(r"$Z_\mathrm{Re}$ [Ohm]", fontsize=24)
ax.set_ylabel(r"$-Z_\mathrm{Im}$ [Ohm]", fontsize=24)

# Set tick label size
ax.tick_params(axis='both', which='major', labelsize=18)

# Add legend with large font size
ax.legend(fontsize=18, loc="best", frameon=False)

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Set aspect ratio to equal for a true Nyquist plot
ax.set_aspect('equal', adjustable='box')

# Use tight layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig("nyquist_plot_vq_params.png", dpi=300, bbox_inches='tight')
print("✓ Nyquist plot saved to 'nyquist_plot_vq_params.png'")

plt.show()
