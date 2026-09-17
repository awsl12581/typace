"""Elliptic Kepler equation and analytic two-body propagation."""

from math import pi, sin

from scipy.optimize import root_scalar

TAU = 2.0 * pi
KEPLER_ABSOLUTE_TOLERANCE_RAD = 1.0e-12
KEPLER_MAX_ITERATIONS = 64


def normalize_angle(angle_rad: float) -> float:
    return angle_rad % TAU


def solve_eccentric_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    """Solve M = E - e sin(E) through SciPy's sole production root path."""

    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    normalized_mean = normalize_angle(mean_anomaly_rad)
    if eccentricity == 0.0:
        return normalized_mean

    def residual(eccentric_anomaly_rad: float) -> float:
        return (
            eccentric_anomaly_rad
            - eccentricity * sin(eccentric_anomaly_rad)
            - normalized_mean
        )

    result = root_scalar(
        residual,
        bracket=(0.0, TAU),
        method="brentq",
        xtol=KEPLER_ABSOLUTE_TOLERANCE_RAD,
        maxiter=KEPLER_MAX_ITERATIONS,
    )
    if not result.converged:
        raise ValueError("Kepler equation did not converge")
    return float(result.root)
