"""Single Lambert production adapter backed by lamberthub.izzo2015."""

from dataclasses import dataclass
from typing import cast

from lamberthub import izzo2015
import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float64]

LAMBERT_MAX_ITERATIONS = 35
LAMBERT_ABSOLUTE_TOLERANCE = 1.0e-5
LAMBERT_RELATIVE_TOLERANCE = 1.0e-7


@dataclass(frozen=True, slots=True)
class LambertSolution:
    departure_velocity_m_s: Vector
    arrival_velocity_m_s: Vector


class LambertFailure(ValueError):
    """The requested zero-revolution transfer has no accepted solution."""


def solve_lambert(
    mu_m3_s2: float,
    departure_position_m: Vector,
    arrival_position_m: Vector,
    flight_time_s: float,
    *,
    prograde: bool,
    low_path: bool,
) -> LambertSolution:
    """Solve one M=0 transfer without algorithm selection or fallback."""

    if mu_m3_s2 <= 0.0 or flight_time_s <= 0.0:
        raise LambertFailure("gravity and flight time must be positive")
    departure = np.asarray(departure_position_m, dtype=np.float64)
    arrival = np.asarray(arrival_position_m, dtype=np.float64)
    if departure.shape != (3,) or arrival.shape != (3,):
        raise LambertFailure("positions must contain three components")
    if not np.isfinite(departure).all() or not np.isfinite(arrival).all():
        raise LambertFailure("positions must be finite")
    try:
        raw_solution = izzo2015(
            mu_m3_s2,
            departure,
            arrival,
            flight_time_s,
            M=0,
            prograde=prograde,
            low_path=low_path,
            maxiter=LAMBERT_MAX_ITERATIONS,
            atol=LAMBERT_ABSOLUTE_TOLERANCE,
            rtol=LAMBERT_RELATIVE_TOLERANCE,
        )
        departure_velocity, arrival_velocity = cast(tuple[Vector, Vector], raw_solution)
    except (ValueError, ZeroDivisionError, RuntimeError) as error:
        raise LambertFailure("Lambert transfer is unreachable") from error
    if (
        not np.isfinite(departure_velocity).all()
        or not np.isfinite(arrival_velocity).all()
    ):
        raise LambertFailure("Lambert transfer produced a non-finite state")
    return LambertSolution(departure_velocity, arrival_velocity)
