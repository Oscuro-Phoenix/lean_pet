import numpy as np
import pybamm
import matplotlib.pyplot as plt
import pandas as pd

import pybammeis

def MHC_kfunc(eta, lmbda):
    """
    MHC function adapted for PyBaMM.
    """
    # Use pybamm functions for symbolic operations
    a = 1.0 + pybamm.sqrt(lmbda)
        
    # term1 = sqrt(pi * lambda) / (1 + exp(-eta))
    term1 = 1 / (1.0 + pybamm.exp(-eta))
    
    # arg_erf = (lambda - sqrt(a + eta^2)) / (2 * sqrt(lambda))
    arg_erf = (lmbda - pybamm.sqrt(a + eta**2)) / (2.0 * pybamm.sqrt(lmbda))
    
    # term2 = 1 - erf(arg_erf)
    term2 = 1.0 - pybamm.erf(arg_erf)
    
    return term1 * term2

class CathodeKineticsCIET(pybamm.interface.kinetics.BaseKinetics):
    """
    Custom kinetics class implementing the CIET model (Marcus-Hush-Chidsey).
    """
    def __init__(self, param, domain, reaction, options=None, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def get_coupled_variables(self, variables):
        # Store variables temporarily so _get_kinetics can access them
        self.temp_variables = variables
        return super().get_coupled_variables(variables)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        # Retrieve variables
        domain = self.domain.capitalize() # "Positive"
        
        c_e_key = f"{domain} electrolyte concentration [mol.m-3]"
        c_e_m3 = self.temp_variables[c_e_key]
        # Use ONLY ONE source for solid concentration.
        c_s_key = f"{domain} particle concentration [mol.m-3]"
        
        if c_s_key in self.temp_variables:
            c_s_bulk_m3 = self.temp_variables[c_s_key]
        else:
            # Fallback
            c_s_key = f"{domain} primary particle concentration [mol.m-3]"
            if c_s_key in self.temp_variables:
                c_s_bulk_m3 = self.temp_variables[c_s_key]
            else:
                raise KeyError(
                    f"No particle concentration variable found (tried {domain} particle concentration)"
                )

        c_e0_m3 = 1000
        c_s_max_m3 = pybamm.Parameter(
            f"Maximum concentration in {domain.lower()} electrode [mol.m-3]"
        )

        # Put the solid concentration on the *electrode* domain
        if hasattr(c_s_bulk_m3, "domain") and any("particle" in d for d in c_s_bulk_m3.domain):
            c_s_bulk_m3 = pybamm.r_average(c_s_bulk_m3)

        # Dimensionless concentrations
        c_lyte = c_e_m3 / c_e0_m3
        c_lyte = (c_lyte*1.9*np.exp(-1))/(1+c_lyte*1.9*np.exp(-1))
        c_sld = c_s_bulk_m3 / c_s_max_m3
        
        # Numerical safety
        eps = 1e-12
        c_lyte = pybamm.maximum(c_lyte, eps)
        c_sld = pybamm.minimum(pybamm.maximum(c_sld, eps), 1 - eps)
        
        # Thermal voltage in Volts
        V_thermal = pybamm.constants.R * T / pybamm.constants.F
        
        # Normalize eta_r to thermal units (dimensionless)
        eta_dim = eta_r/V_thermal
      
        # Reorganization energy lambda [eV]
        lmbda_eV = pybamm.Parameter(f"{domain} electrode reorganization energy [eV]")
    
        lmbda_dim = lmbda_eV / V_thermal
        
        eta_f_dim = eta_dim + pybamm.log(c_lyte / c_sld)

        ecd_extras = (1.0 - c_sld) / 2
        
        # Using dimensionless inputs for MHC
        krd =  MHC_kfunc(-eta_f_dim, lmbda_dim)
        kox =  MHC_kfunc(eta_f_dim, lmbda_dim)
        
        # Calculate current density
        j = -j0*ecd_extras * (krd * c_lyte - kox * c_sld)
        
        return j

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

# Setup PyBaMM model with options from VQ_sim_example.py
# Note: Using ECIT (CathodeKineticsCIET) kinetics
options = {
    "working electrode": "positive",
    "SEI": ("none", "constant"),
    "SEI film resistance": "distributed",
    "surface form": "differential",
    "particle": "uniform profile",
    "intercalation kinetics": "linear",
}

# Create model with build=False so we can swap kinetics
model = pybamm.lithium_ion.DFN(options=options, build=False)

# Swap in the custom ECIT kinetics class
param_obj = model.param
new_kinetics = CathodeKineticsCIET(
    param_obj, 
    "positive", 
    "lithium-ion main", 
    model.options, 
    "primary"
)

# Replace the positive electrode kinetics submodel
target_key = "positive primary interface"
if target_key in model.submodels:
    model.submodels[target_key] = new_kinetics
else:
    # Fallback search
    for k in model.submodels.keys():
        if "positive" in k and "interface" in k:
            if k == "positive interface" or k == "positive primary interface":
                model.submodels[k] = new_kinetics
                break

# Now build the model with custom kinetics
model.build_model()

# Build parameter set from VQ_sim_example.py
sigma_s_c = 1e-1  # S/m
Dp = 2.2e-10  # m²/s (cation diffusivity - Li+)
Dm = 2.94e-10  # m²/s (anion diffusivity)
L_c = 100e-6  # m
poros_c = 0.5
P_L_c = 0.69
mean_c = 500e-9  # m
c0 = 1000  # mol/m³
T_ref = 298.15  # K
L_s = 5e-6  # m
poros_s = 1.0
BruggExp_c = 1.5
BruggExp_s = 1.5

# For Stefan-Maxwell, use individual species diffusivities
# Keep D_eff for backwards compatibility with analytical predictions
D_eff = 2*Dp*Dm/(Dp+Dm)
t_plus = 0.38 
F = 96485
R = 8.314

c_s_max_pos = 2.9869e28*1.6e-19/F  # mol/m³
D_s_pos = 1e-10 # m²/s

# For ECIT kinetics (CathodeKineticsCIET), use constant m_ref
m_ref = 5  # Constant prefactor for ECIT/MHC kinetics

E_r = 39570  # J/mol

c_s_max_neg = 10000
L_n = 50e-6
poros_n = 0
L_cc_p = 1e-6
L_cc_n = 1e-6
height = 1e-2
width = 1e-2

def custom_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    # For ECIT/MHC kinetics, return constant prefactor
    # The kinetics class handles concentration dependence
    return m_ref

param = pybamm.ParameterValues({
    "Electrode height [m]": height,
    "Electrode width [m]": width,
    "Negative electrode thickness [m]": L_n,
    "Separator thickness [m]": L_s,
    "Positive electrode thickness [m]": L_c,
    "Negative current collector thickness [m]": L_cc_n,
    "Positive current collector thickness [m]": L_cc_p,
    "Negative electrode porosity": poros_n,
    "Negative electrode exchange-current density [A.m-2]": 1e8,
    "Negative electrode OCP [V]": 0.0,
    "Negative electrode conductivity [S.m-1]": 1e8,
    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Positive electrode reorganization energy [eV]": 0.112,
    "Negative electrode reorganization energy [eV]": 0.01,
    "Positive electrode porosity": poros_c,
    "Positive electrode active material volume fraction": (1-poros_c)*P_L_c,
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
    "Electrolyte diffusivity [m2.s-1]": D_eff*10,
    "Cation diffusivity [m2.s-1]": Dp,  # Used by Stefan-Maxwell
    "Anion diffusivity [m2.s-1]": Dm,   # Used by Stefan-Maxwell
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
    "Nominal cell capacity [A.h]": height*width*L_c*(1-poros_c)*P_L_c*96485*c_s_max_pos/3600,
    "SEI partial molar volume [m3.mol-1]": 9.585e-04,
    "Ratio of lithium moles to SEI moles": 2.0,
    "Initial concentration in negative electrode [mol.m-3]": 0.99 * c_s_max_neg,
    "Initial concentration in positive electrode [mol.m-3]": 0.3 * c_s_max_pos,
})

# SEI parameters
R_film_cathode = 0.0
L_sei = 1e-3
R_sei = R_film_cathode / L_sei
param.update({
    "SEI resistivity [Ohm.m]": R_sei,
    "Initial SEI thickness [m]": L_sei,
}, check_already_exists=False)

# Set initial stoichiometry
initial_stoichiometry = 0.3
initial_concentration_pos = initial_stoichiometry * c_s_max_pos

param.update({
    "Initial concentration in positive electrode [mol.m-3]": initial_concentration_pos,
}, check_already_exists=False)

print("="*70)
print("PyBaMM-EIS Simulation with VQ_sim_example.py Parameters (ECIT)")
print("="*70)
print(f"\nKey Parameters:")
print(f"  Positive electrode thickness: {L_c*1e6:.1f} µm")
print(f"  Positive particle radius: {mean_c*1e9:.1f} nm")
print(f"  Positive electrode porosity: {poros_c}")
print(f"  Separator thickness: {L_s*1e6:.1f} µm")
print(f"  Electrolyte concentration: {c0} mol/m³")
print(f"  Cation transference number: {t_plus}")
print(f"  Initial stoichiometry: {initial_stoichiometry:.3f}")
print(f"  m_ref (constant for ECIT): {m_ref:.6e} A/m²")
print(f"  Exchange current density: {m_ref:.6e} A/m² (constant)")

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
