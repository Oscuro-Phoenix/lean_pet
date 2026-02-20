import numpy as np
import pybamm

import pybammeis

print("=" * 60)
print("HALF-CELL SIMULATION METHODS")
print("=" * 60)

# Method 1: Using working electrode option with any parameter set
print("\nMethod 1: Using 'working electrode' option")
print("-" * 40)

# This works with any parameter set - just specify which electrode to use
model_full = pybamm.lithium_ion.DFN(options={"working electrode": "positive", "surface form": "differential"})
param_mohtat = pybamm.ParameterValues("Mohtat2020")

print(f"Model options: {model_full.options}")
print(f"Parameter set: Mohtat2020 (NMC532)")
print("This simulates only the positive electrode (NMC532) against a reference electrode")

# Method 2: Using dedicated half-cell parameter sets
print("\nMethod 2: Using dedicated half-cell parameter sets")
print("-" * 40)

# Check available half-cell parameter sets
available_params = list(pybamm.parameter_sets.keys())
half_cell_params = [p for p in available_params if 'halfcell' in p.lower()]
print(f"Available half-cell parameter sets: {half_cell_params}")

if half_cell_params:
    model_half = pybamm.lithium_ion.DFN(options={"working electrode": "positive", "surface form": "differential"})
    param_half = pybamm.ParameterValues(half_cell_params[0])
    print(f"Using: {half_cell_params[0]}")
    print(f"Description: {pybamm.parameter_sets.get_docstring(half_cell_params[0])}")

# Method 3: Creating custom half-cell parameters
print("\nMethod 3: Creating custom half-cell parameters")
print("-" * 40)

# Start with a full cell parameter set and modify for half-cell
param_custom = pybamm.ParameterValues("Mohtat2020")

# You can modify specific parameters for half-cell simulation
# For example, set negative electrode parameters to reference electrode values
print("You can modify parameters to simulate specific half-cell configurations")
print("Example: Set negative electrode to lithium metal reference")

# Method 4: Different working electrodes
print("\nMethod 4: Different working electrode options")
print("-" * 40)

# Positive electrode as working electrode (cathode half-cell)
model_pos = pybamm.lithium_ion.DFN(options={"working electrode": "positive", "surface form": "differential"})
print("Positive electrode working electrode: Cathode half-cell")

# Negative electrode as working electrode (anode half-cell)  
model_neg = pybamm.lithium_ion.DFN(options={"working electrode": "negative", "surface form": "differential"})
print("Negative electrode working electrode: Anode half-cell")

print("\n" + "=" * 60)
print("RECOMMENDED APPROACH FOR NMC532 HALF-CELL")
print("=" * 60)

print("For NMC532 half-cell simulation, use:")
print("1. Model: DFN with options={'working electrode': 'positive', 'surface form': 'differential'}")
print("2. Parameters: Mohtat2020 (contains NMC532 parameters)")
print("3. This gives you NMC532 vs reference electrode simulation")

print("\nExample code:")
print("model = pybamm.lithium_ion.DFN(options={'working electrode': 'positive', 'surface form': 'differential'})")
print("param = pybamm.ParameterValues('Mohtat2020')")
print("eis_sim = pybammeis.EISSimulation(model, parameter_values=param)") 