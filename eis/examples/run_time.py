#
# Example showing how to load and solve the DFN for an NMC532 half-cell simulation
#

import pybamm

pybamm.set_logging_level("INFO")

# load model with half-cell options (positive electrode as working electrode)
model = pybamm.lithium_ion.DFN(options={"working electrode": "positive", "surface form": "differential"})
# create geometry
geometry = model.default_geometry

# load parameter values for NMC532 half-cell and process model and geometry
param = pybamm.ParameterValues("Mohtat2020")
param.process_geometry(geometry)
param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = [0, 3600]
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-3)
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(
    solution,
    [
        "Positive particle concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Current [A]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Voltage [V]",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()