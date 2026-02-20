#!/usr/bin/env python3
"""
Combined plot: Voltage vs Cathode Filling Fraction
Allows specifying any kinetics for PyBaMM to compare against.
"""
import os
import scipy.io
import pybamm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import scipy.special as spl

# =============================================================================
# 1. Kinetics and OCV Definitions
# =============================================================================
def NMC532_Colclasure20(x):
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

def ecd_pybamm(c_sld, c_lyte=1, k0=5e-6, R_film=0.0):
    lmbda = 0.112/0.0257
    c_lyte = (c_lyte*1.9*np.exp(-1))/(1+c_lyte*1.9*np.exp(-1))
    eta = np.log(c_lyte/c_sld)
    a = 1.0 + np.sqrt(lmbda)
    term_erf = (1-spl.erf((lmbda - np.sqrt(a + eta**2)) / (2.0 * np.sqrt(lmbda))))
    f = (1-c_sld)*c_sld*c_lyte/(c_sld+c_lyte)*term_erf/2
    return f/(k0*f*R_film/0.0257+1)

def ecd_pybamm_df_dclyte(c_sld, c_lyte=1, k0=5e-6, R_film=0.0):
    g = (1/(c_sld+c_lyte) - c_lyte/(c_sld+c_lyte)**2)/(c_lyte/(c_sld+c_lyte))
    return g

def predict_VQ_pybamm(ocv_function, X, Da_w, Da_w_sigma, Da_w_kappa, Da_p, Da_lim, J_P, k0=5e-6, R_film=0.0):
    """
    Predicts the voltage and capacity with SEI film resistance (PyBaMM version).
    """
    X = np.asarray(X)
    ec = ecd_pybamm(X, c_lyte=1, k0=k0, R_film=R_film)
    alpha = ecd_pybamm_df_dclyte(X, c_lyte=1, k0=k0, R_film=R_film)
    Lambda = np.sqrt((Da_w*ec + alpha*Da_p/J_P))
    beta = Da_w_sigma/Da_w 
    fac2 = 1/(1+Da_w*ec/(alpha*Da_p/J_P))
    
    Xi = np.zeros_like(X)
    nonzero_mask = (ec > 1e-10)
    Lambda_nz = Lambda[nonzero_mask]
    Z = (Lambda_nz**2)*(2*beta*(1-beta)*(0.5+1/(Lambda_nz*np.sinh(Lambda_nz)))+(beta**2 + (1-beta)**2)*np.cosh(Lambda_nz)/(Lambda_nz*np.sinh(Lambda_nz)))
    # Xi[nonzero_mask] = ((1-fac2[nonzero_mask])*Z
    #                    + fac2[nonzero_mask])/(J_P*ec[nonzero_mask])
    Xi[nonzero_mask] = Z/(J_P*ec[nonzero_mask])
    
    V = ocv_function(X) - abs(Xi)*0.0257
    return (X, V)

# =============================================================================
# 3. Simulation Logic
# =============================================================================

def run_pybamm_simulations(kinetics_class=None):
    """
    Run PyBaMM simulations and return results.
    
    Args:
        kinetics_class: A custom kinetics class (e.g. CathodeKineticsCIET) to swap in.
                        If None, uses the default (linear) kinetics.
    """
    print("="*70)
    print("Running PyBaMM Simulations")
    if kinetics_class:
        print(f"Using Custom Kinetics: {kinetics_class.__name__}")
    else:
        print("Using Default Kinetics (linear)")
    print("="*70)
    
    # Setup PyBaMM model options
    options = {
        "working electrode": "positive",
        "SEI": ("none", "constant"),
        "SEI film resistance": "distributed",
        "surface form": "differential",
        "particle": "uniform profile",
        "intercalation kinetics": "linear", # Default, overridden if kinetics_class used
    }
    
    # Build parameter set
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
    
    def electrolyte_conductivity(c_e, T):
        (k00, k01, k02, k10, k11, k12, k20, k21) = (
            -8.2488, 0.053248, -0.000029871,
            0.26235, -0.0093063, 0.000008069,
            0.22002, -0.0001765)
        c = c_e / 1000.0
        out = c * (k00 + k01*T + k02*T**2
                   + k10*c + k11*c*T + k12*c*T**2
                   + k20*c**2 + k21*c**2*T)**2  # mS/cm
        out *= 0.1  # -> S/m
        return out
    
    # Calculate reference conductivity at c0 and T_ref
    kappa_ref = electrolyte_conductivity(c0, T_ref)
  
    c_s_max_pos = 2.9869e28*1.6e-19/F
    D_s_pos = 1e-10
    
    # Exchange current density parameters
    # Note: If using CIET/MHC, we might want a constant m_ref (prefactor) 
    # and let the kinetics class handle concentration dependence.
    if kinetics_class:
        # For custom kinetics (like CIET from sei_cath.py), use constant j0
        m_ref = 5  # Adjust as needed for the custom model
    else:
        # For standard linear/BV, use concentration dependent j0
        m_ref = 0.1/(c_s_max_pos*c0**0.5)

    E_r = 39570
    
    c_s_max_neg = 10000
    L_n = 50e-6
    poros_n = 0
    L_cc_p = 1e-6
    L_cc_n = 1e-6
    height = 1e-2
    width = 1e-2
    
    def nmc532_ocp(sto):
        return NMC532_Colclasure20(sto)
    
    def custom_exchange_current_density(c_e, c_s_surf, c_s_max, T):
        arrhenius = pybamm.exp(E_r / R * (1 / 298.15 - 1 / T))
        if kinetics_class:
             return m_ref # Constant prefactor for custom kinetics
        else:
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
        "Positive electrode OCP [V]": nmc532_ocp,
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
        # Provide both single and separate diffusivities for compatibility
        "Electrolyte diffusivity [m2.s-1]": D_eff,  # Used by default equations
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
    
    # C-rates to simulate
    crate_labels = ['0.5C', '1C', '1.5C', '2C']
    C_rates = [0.5, 1.0, 1.5, 2.0]
    cutoff_voltage = 3
    
    # Calculate absolute currents from C-rates
    nominal_capacity = height*width*L_c*(1-poros_c)*P_L_c*c_s_max_pos*96485/3600
    currents = [C_rate * nominal_capacity for C_rate in C_rates]
    
    results = []
    
    print(f"\nRunning {len(C_rates)} C-rate simulations...")
    for i, C_rate in enumerate(C_rates):
        current = currents[i]
        print(f"  {crate_labels[i]}: {current:.4f} A")
        
        # NOTE: Must set build=False if we plan to swap submodels
        model = pybamm.lithium_ion.DFN(options=options, build=False)
        
        # Swap kinetics if requested
        if kinetics_class:
            param_obj = model.param
            new_kinetics = kinetics_class(
                param_obj, 
                "positive", 
                "lithium-ion main", 
                model.options, 
                "primary"
            )
            target_key = "positive primary interface"
            if target_key in model.submodels:
                model.submodels[target_key] = new_kinetics
            else:
                # Fallback search
                for k in model.submodels.keys():
                    if "positive" in k and "interface" in k:
                        # Assuming the standard key is what we want to replace
                        if k == "positive interface" or k == "positive primary interface":
                             model.submodels[k] = new_kinetics
                             break
                # Force set if not found (standard DFN usually has "positive primary interface")
                model.submodels["positive primary interface"] = new_kinetics
        
        # Build the model now
        model.build_model()
        
        experiment = pybamm.Experiment([
            f"Discharge at {C_rate} C until {cutoff_voltage}V"
        ])
        
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        
        try:
            solution = sim.solve()
            
            voltage = solution["Voltage [V]"].entries
            time = solution["Time [s]"].entries
            cathode_filling_raw = solution["Positive particle stoichiometry"].entries
            
            if cathode_filling_raw.ndim > 1:
                cathode_filling = cathode_filling_raw.mean(axis=tuple(range(cathode_filling_raw.ndim - 1)))
            else:
                cathode_filling = cathode_filling_raw
            
            results.append({
                'C_rate': C_rate,
                'label': crate_labels[i],
                'voltage': voltage,
                'time': time,
                'cathode_filling': cathode_filling,
            })
            
            print(f"    ✓ Complete (Final V: {voltage[-1]:.3f} V)")
            
        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()

    
    # Return parameters for analytical prediction
    pybamm_params = {
        'L': param["Positive electrode thickness [m]"],
        'poros': param["Positive electrode porosity"],
        'R_p': param["Positive particle radius [m]"],
        'kappa_eff': kappa_ref,
        'D_eff': D_eff,  # Use calculated D_eff for analytical predictions
        'Dp': param["Cation diffusivity [m2.s-1]"],  # Li+ diffusivity
        'Dm': param["Anion diffusivity [m2.s-1]"],   # Anion diffusivity
        't_plus': param["Cation transference number"],
        'c_s_max': param["Maximum concentration in positive electrode [mol.m-3]"],
        'c_e_ref': param["Initial concentration in electrolyte [mol.m-3]"],
        'sigma_s_c': param["Positive electrode conductivity [S.m-1]"],
        'P_L_c': P_L_c,
        'm_ref': m_ref,
        'R_film': R_film_cathode,
    }
    
    return results, pybamm_params

def load_mpet_data(folder_path):
    """Load MPET simulation data from .mat files"""
    print("\n" + "="*70)
    print("Loading MPET Data")
    print("="*70)
    
    crate_labels = ['0.5C', '1C', '1.5C', '2C']
    default_L = 100.0 * 1e-6
    default_poros = 0.5
    
    print(f"Searching in: {folder_path}")
    
    mpet_results = []
    
    for idx, crate in enumerate(crate_labels):
        for foldername, subfolders, filenames in os.walk(folder_path):
            if crate in foldername:
                I = float(crate.replace('C', ''))
                for filename in filenames:
                    if filename.endswith('.mat'):
                        file_path = os.path.join(foldername, filename)
                        data = scipy.io.loadmat(file_path)
                        
                        t = np.squeeze(data['phi_applied_times'])
                        ffrac_c = np.squeeze(data['ffrac_c'])
                        V = np.squeeze(data['phi_applied'])
                        L = float(data['L_c']) if 'L_c' in data else default_L
                        poros = float(data['poros_c']) if 'poros_c' in data else default_poros
                        
                        mpet_results.append({
                            'C_rate': I,
                            'label': crate,
                            'time': t,
                            'ffrac_c': ffrac_c,
                            'voltage': V,
                            'L': L,
                            'poros': poros,
                        })
                        
                        print(f"  Loaded {crate}: {len(ffrac_c)} points")
                        break
    
    # MPET parameters
    R_p = 500e-9
    k0 = 10
    
    mpet_params = {
        'R_p': R_p,
        'k0': k0,
        'P_L_c': 0.69,
        't_plus': 0.38,
        'c_e_ref': 1000,
        'F': 96485,
        'D_eff': 2.5e-10,
    }
    
    print(f"Found {len(mpet_results)} MPET datasets")
    
    return mpet_results, mpet_params

# =============================================================================
# 4. Plotting Function
# =============================================================================

def create_combined_plot(pybamm_results, pybamm_params, mpet_results=None, mpet_params=None, title="Combined V-Q Plot"):
    print("\n" + "="*70)
    print("Creating Plot")
    print("="*70)
    
    plt.figure(figsize=(10, 7), dpi=300)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 24})
    
    ax = plt.gca()
    ax.set_xlabel('Cathode Filling Fraction', fontsize=28, fontweight='normal')
    ax.set_ylabel('Voltage (V)', fontsize=28, fontweight='normal')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylim(top=4.3, bottom=3.3)
    
    # Get all C-rates for color mapping
    crate_labels = ['0.5C', '1C', '1.5C', '2C']
    n_crates = len(crate_labels)
    cmap = plt.cm.coolwarm
    colors = {label: cmap(i / (n_crates-1)) for i, label in enumerate(crate_labels)}
    
    # Plot PyBaMM results
    print("\nPlotting PyBaMM results:")

    for result in pybamm_results:
        C_rate = result['C_rate']
        label = result['label']
        cathode_filling = result['cathode_filling']
        voltage = result['voltage']
        color = colors[label]
        
        # Plot PyBaMM simulation (scatter points)
        n_scatter_points = 15
        filling_min = np.min(cathode_filling)
        filling_max = np.max(cathode_filling)
        uniform_filling = np.linspace(filling_min, filling_max, n_scatter_points)
        
        interp_func = interp1d(cathode_filling, voltage, kind='linear', fill_value='extrapolate')
        uniform_voltage = interp_func(uniform_filling)
        
        ax.scatter(
            uniform_filling[1:], uniform_voltage[1:],
            color=color,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.0,
            marker='o'
        )
        
        # Calculate analytical prediction (PyBaMM version) for comparison
        a_p = (3/pybamm_params['R_p']) * (1-pybamm_params['poros'])*pybamm_params['P_L_c']
        
        k0 = pybamm_params['m_ref'] 
        
        # Only plot analytical if we have valid k0 (might be different for custom kinetics)
        if pybamm_params['m_ref'] > 0:
             J_P = k0*a_p*3600/(pybamm_params['P_L_c']*C_rate*(1-pybamm_params['poros'])*96485*pybamm_params['c_s_max'])
             Da_p = k0*a_p*pybamm_params['L']**2*(1-pybamm_params['t_plus'])/(pybamm_params['poros']**1.5*pybamm_params['c_e_ref']*96485*pybamm_params['D_eff'])
             sigma_s_c_eff = pybamm_params['sigma_s_c']*(((1-pybamm_params['poros']))**1.5)
             kappa_eff = pybamm_params['kappa_eff']*(pybamm_params['poros']**1.5)
             sigma_eff = (sigma_s_c_eff**-1 + kappa_eff**-1)**-1
             Da_w_sigma = k0*a_p*pybamm_params['L']**2/(sigma_s_c_eff)/0.0257
             Da_w_kappa = k0*a_p*pybamm_params['L']**2/(kappa_eff)/0.0257
             Da_w = Da_w_sigma + Da_w_kappa
             Da_lim = Da_p/J_P
             
             X_pred = np.linspace(min(cathode_filling)+0.02, min(1.0, max(cathode_filling+0.02)), 100)
             (predffrac_c, predicted_V) = predict_VQ_pybamm(
                 NMC532_Colclasure20, X_pred, Da_w, Da_w_sigma, Da_w_kappa, Da_p, Da_lim, J_P, k0, pybamm_params['R_film']
             )

             ax.plot(
                 predffrac_c, predicted_V,
                 '-',
                 color=color,
                 linewidth=3,
                 alpha=0.9
             )
        
        print(f"  PyBaMM {label}")
    
    # Plot MPET results
    if mpet_results:
        print("\nPlotting MPET results:")
        for result in mpet_results:
            C_rate = result['C_rate']
            label = result['label']
            ffrac_c = result['ffrac_c']
            V = result['voltage']
            L = result['L']
            poros = result['poros']
            
            # Map label to color if exists
            if label in colors:
                color = colors[label]
            else:
                color = 'black' # Fallback
            
            # Plot MPET simulation (triangle markers)
            n_scatter_points = 20
            if len(ffrac_c) > n_scatter_points:
                indices = np.linspace(0, len(ffrac_c) - 1, n_scatter_points, dtype=int)
            else:
                indices = np.arange(len(ffrac_c))
            
            ax.scatter(
                ffrac_c[indices][1:], (NMC532_Colclasure20(ffrac_c[0]) - V*0.0257)[indices][1:],
                color=color,
                s=120,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.0,
                marker='^'
            )
            
            print(f"  MPET {label}")

    # Custom legend
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=14, linestyle='None', label='PyBaMM'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=14, linestyle='None', label='MPET'),
        Line2D([0], [0], color='black', lw=3, label='Analytical')
    ]
    plt.legend(handles=custom_lines, fontsize=26, frameon=False, loc='upper right')
    
    # Add C-rate text labels
    x_text, y_text = 0.6, 0.675
    plt.text(x_text, y_text, 'C-rate:', color='black', fontsize=24,
             va='top', ha='left', transform=ax.transAxes, fontweight='normal')
    
    y_offset = y_text - 0.08
    x_offset = x_text
    for idx, crate in enumerate(crate_labels):
        text_str = crate + ("," if idx < len(crate_labels)-1 else "")
        t = ax.text(x_offset, y_offset, text_str, color=colors[crate], fontsize=26,
                   va='top', ha='left', transform=ax.transAxes, fontweight='normal')
        # Update x position for next label based on text length
        # Approximate character width in axes coordinates (adjust 0.016 for tighter/looser spacing)
        char_width = 0.02 * 26/24  # Scale by font size ratio
        x_offset += len(text_str) * char_width + 0.01  # text width + small padding
        
    #plt.title(title)
    plt.tight_layout()
    output_file = 'combined_vq_plot_custom.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_file}")
    
    plt.show()

# =============================================================================
# 5. Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # SPECIFY KINETICS HERE
    # Set to None for default Linear/BV
    # Set to CathodeKineticsCIET for Custom CIET
    KINETICS_TO_USE = CathodeKineticsCIET 
    
    sim_name = KINETICS_TO_USE.__name__ if KINETICS_TO_USE else "Standard Linear"
    print(f"Running simulations with: {sim_name}")
    
    pybamm_results, pybamm_params = run_pybamm_simulations(kinetics_class=KINETICS_TO_USE)
    
    # Load MPET data
    mpet_folder = '/home/shakulp/Desktop/mpet_scaling_test/mpet/store/NMC_500nm_MHC'
    if os.path.exists(mpet_folder):
        mpet_results, mpet_params = load_mpet_data(mpet_folder)
    else:
        print(f"\nWarning: MPET folder not found at {mpet_folder}")
        mpet_results = []
        mpet_params = {}
    
    if len(pybamm_results) > 0 or len(mpet_results) > 0:
        create_combined_plot(pybamm_results, pybamm_params, mpet_results, mpet_params, title=f"V-Q Plot ({sim_name})")
    else:
        print("No results to plot.")
