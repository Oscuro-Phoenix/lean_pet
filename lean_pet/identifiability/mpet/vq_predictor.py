import numpy as np
from typing import Callable, Tuple


def ecd(c_sld: np.ndarray, c_lyte: float = 1.0) -> np.ndarray:
    """
    Effective charge diffusion term: sqrt((1 - c_sld) * c_sld * c_lyte).

    Parameters:
    - c_sld: Solid fraction (state of charge), array-like in [0, 1]
    - c_lyte: Electrolyte concentration scaling (dimensionless)

    Returns:
    - Array of the same shape as c_sld
    """
    c_sld = np.asarray(c_sld)
    return np.sqrt(np.clip((1.0 - c_sld) * c_sld * c_lyte, a_min=0.0, a_max=None))


def predict_vq(
    ocv_function: Callable[[np.ndarray], np.ndarray],
    ffrac_c: np.ndarray,
    Da_w: float,
    Da_p: float,
    Da_lim: float,
    J_P: float,
    R_series_drop: float = 0.0,
    *,
    x_ref: float = 0.3,
    temperature_factor: float = 0.0257,
    apply_separator_correction: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict V–Q curve for a given OCV function and operating parameters.

    Parameters:
    - ocv_function: Maps fractional cathode filling (ffrac_c in [0, 1]) to OCV (V, dimensionless here)
    - ffrac_c: Array of fractional cathode filling values (capacity axis)
    - Da_w: Dimensionless number capturing reaction/transport interplay (wall-related)
    - Da_p: Dimensionless number for porosity/transport
    - Da_lim: Dimensionless limiting-current parameter (currently unused in this closure, included for API completeness)
    - J_P: Dimensionless current-density-like parameter
    - x_ref: Reference offset for empirical fraction correction (default 0.3)
    - temperature_factor: Thermal voltage scaling (default 0.0257 ~ RT/F at room temp)
    - apply_separator_correction: Whether to apply the empirical fraction correction

    Returns:
    - (ffrac_c_out, V_pred): Tuple of corrected capacity axis and predicted voltage (same shape)

    Notes:
    - This implements the same predictor used in analysis scripts, extracted for reuse.
    - Inputs are assumed dimensionless and consistent with upstream preprocessing.
    """
    X = np.asarray(ffrac_c)

    # Core terms
    ec = ecd(X)

    #Da_w = Da_w *L_frac**2 
    Lambda = np.sqrt(np.maximum(Da_w * ec + Da_p / max(J_P, 1e-30), 0.0))

    # Avoid division-by-zero in fac2 and J_P terms by using masks
    with np.errstate(divide='ignore', invalid='ignore'):
        fac2 = 1.0 / (1.0 + (Da_w * J_P * ec) / np.maximum(Da_p, 1e-30))

    # Compute Xi using a safe mask where ec > 0 to avoid 0/0
    Xi = np.zeros_like(X, dtype=float)
    nonzero_mask = ec > 1e-12
    
    if np.any(nonzero_mask):
        Lnz = Lambda[nonzero_mask]
        fac2_nz = fac2[nonzero_mask]
        # term = ((1 - fac2) * Lambda / tanh(Lambda) + fac2) / (J_P * ec)
        tanh_Lnz = np.tanh(Lnz)
        # Prevent division by exactly zero if tanh(Lambda) == 0 for tiny Lambda
        safe_tanh = np.where(np.abs(tanh_Lnz) < 1e-12, 1e-12, tanh_Lnz)
        numerator = (1.0 - fac2_nz) * (Lnz / safe_tanh) + fac2_nz
        denominator = np.maximum(J_P * ec[nonzero_mask], 1e-30)
        Xi[nonzero_mask] = numerator / denominator

    # Predicted voltage
    V = ocv_function(X) - np.abs(Xi) * temperature_factor

    # Optional empirical correction to capacity axis via separator concentration
    if apply_separator_correction:
        with np.errstate(divide='ignore', invalid='ignore'):
            fac = 1.0 / (1.0 + (Da_w * J_P * ec) / np.maximum(Da_p, 1e-30))
        # c_sep = 1 - fac * (1 - Lambda / tanh(Lambda))
        tanh_L = np.tanh(Lambda)
        safe_tanh_L = np.where(np.abs(tanh_L) < 1e-12, 1e-12, tanh_L)
        c_sep = 1.0 - fac * (1.0 - Lambda / safe_tanh_L)

        # frac_correction = (2*c_sep*J_P/Da_p) * tanh(1/(2*c_sep*J_P/Da_p))
        safe_c = np.maximum(2.0 * c_sep * np.maximum(J_P, 1e-30) / np.maximum(Da_p, 1e-30), 1e-30)
        frac_correction = safe_c * np.tanh(1.0 / safe_c)

        X_corrected = np.clip((X - x_ref) * frac_correction + x_ref, 0.0, 1.0)
    else:
        X_corrected = X
 
    return X_corrected, (V - R_series_drop)


def predict_vq_from_range(
    ocv_function: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    num_points: int,
    Da_w: float,
    Da_p: float,
    Da_lim: float,
    J_P: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to generate an evenly spaced ffrac_c range and predict V–Q.

    Parameters are the same as in `predict_vq`, with the addition of:
    - x_min, x_max: Range for ffrac_c (inclusive bounds will be clipped to [0, 1])
    - num_points: Number of points in the range
    - kwargs: Forwarded to `predict_vq` (e.g., x_ref, temperature_factor, apply_separator_correction)
    """
    x_min = float(x_min)
    x_max = float(x_max)
    num_points = int(num_points)
    X = np.linspace(max(0.0, x_min), min(1.0, x_max), max(num_points, 2))
    return predict_vq(ocv_function, X, Da_w, Da_p, Da_lim, J_P, **kwargs)


