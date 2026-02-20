"""
lean_pet — Lightweight Analytical Porous-Electrode Theory
=========================================================

Provides analytical models for discharge (VQ), pulsing (I-t), and EIS
protocols, together with tools for comparison against full-scale MPET /
PyBaMM simulations and parameter identifiability analysis.

Subpackages
-----------
core            Shared physics: OCV, kinetics, electrolyte, analytical models,
                parameters, and plotting utilities.
protocols       Protocol-specific runners: discharge, pulsing, EIS.
comparison      RMSE comparison against MPET simulation grids.
identifiability Chi-square landscape and MCMC sensitivity analysis.
"""
