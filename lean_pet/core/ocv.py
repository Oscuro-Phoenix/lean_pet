"""Open-circuit voltage (OCV) functions for NMC532 cathode material."""

import numpy as np


def _exp(x):
    """Dispatch ``exp`` to NumPy or PyBaMM depending on argument type."""
    try:
        import pybamm
        if isinstance(x, pybamm.Symbol):
            return pybamm.exp(x)
    except ImportError:
        pass
    return np.exp(x)


def NMC532_Colclasure20(x):
    """
    NMC532 OCV polynomial from Colclasure et al. 2020.

    Parameters
    ----------
    x : array_like or pybamm.Symbol
        Stoichiometry (lithium filling fraction), 0 < x < 1.
        Accepts NumPy arrays *and* PyBaMM symbolic objects.

    Returns
    -------
    OCV : ndarray or pybamm.Symbol
        Open-circuit voltage [V].
    """
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        pass  # PyBaMM symbolic — leave as-is

    OCV = (
        5.314735633000300e+00
        - 3.640117692001490e+03 * x ** 14.0
        + 1.317657544484270e+04 * x ** 13.0
        - 1.455742062291360e+04 * x ** 12.0
        - 1.571094264365090e+03 * x ** 11.0
        + 1.265630978512400e+04 * x ** 10.0
        - 2.057808873526350e+03 * x ** 9.0
        - 1.074374333186190e+04 * x ** 8.0
        + 8.698112755348720e+03 * x ** 7.0
        - 8.297904604107030e+02 * x ** 6.0
        - 2.073765547574810e+03 * x ** 5.0
        + 1.190223421193310e+03 * x ** 4.0
        - 2.724851668445780e+02 * x ** 3.0
        + 2.723409218042130e+01 * x ** 2.0
        - 4.158276603609060e+00 * x
        - 5.573191762723310e-04
        * _exp(6.560240842659690e+00 * x ** 4.148209275061330e+01)
    )
    return OCV


def NMC532_Colclasure20_deriv(x):
    """
    Analytical derivative dOCV/dx for the NMC532 Colclasure 2020 polynomial.

    Parameters
    ----------
    x : array_like or pybamm.Symbol
        Stoichiometry (lithium filling fraction).

    Returns
    -------
    dOCV_dx : ndarray or pybamm.Symbol
        Derivative of OCV with respect to stoichiometry [V].
    """
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        pass  # PyBaMM symbolic — leave as-is

    dOCV_dx = (
        - 3.640117692001490e+03 * 14.0 * x ** 13.0
        + 1.317657544484270e+04 * 13.0 * x ** 12.0
        - 1.455742062291360e+04 * 12.0 * x ** 11.0
        - 1.571094264365090e+03 * 11.0 * x ** 10.0
        + 1.265630978512400e+04 * 10.0 * x ** 9.0
        - 2.057808873526350e+03 * 9.0  * x ** 8.0
        - 1.074374333186190e+04 * 8.0  * x ** 7.0
        + 8.698112755348720e+03 * 7.0  * x ** 6.0
        - 8.297904604107030e+02 * 6.0  * x ** 5.0
        - 2.073765547574810e+03 * 5.0  * x ** 4.0
        + 1.190223421193310e+03 * 4.0  * x ** 3.0
        - 2.724851668445780e+02 * 3.0  * x ** 2.0
        + 2.723409218042130e+01 * 2.0  * x ** 1.0
        - 4.158276603609060e+00
        - 5.573191762723310e-04
        * _exp(6.560240842659690e+00 * x ** 4.148209275061330e+01)
        * (6.560240842659690e+00 * 4.148209275061330e+01
           * x ** (4.148209275061330e+01 - 1.0))
    )
    return dOCV_dx
