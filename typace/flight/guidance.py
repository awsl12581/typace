"""Convert flight-plan burns into attitude and throttle targets."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from typace.satellites.definition import SatelliteDefinition
from typace.flight.models import BurnStep, FlightPlan


@dataclass(frozen=True, slots=True)
class GuidanceRequest:
    burn: BurnStep
    target_quaternion_wxyz: np.ndarray
    throttle: float


def guidance_for_plan(
    plan: FlightPlan,
    elapsed_s: float,
    definition: SatelliteDefinition,
) -> GuidanceRequest | None:
    for step in plan.steps:
        if not isinstance(step, BurnStep):
            continue
        burn_end_s = step.start_after_s + step.duration_s
        if elapsed_s >= burn_end_s:
            continue
        target = target_attitude_for_direction(
            np.asarray(definition.propulsion.main_body_axis),
            np.asarray(step.delta_v_m_s),
        )
        throttle = step.throttle if elapsed_s >= step.start_after_s else 0.0
        return GuidanceRequest(step, target, throttle)
    return None


def target_attitude_for_direction(
    body_axis: np.ndarray, inertial_direction: np.ndarray
) -> np.ndarray:
    body_length = float(np.linalg.norm(body_axis))
    direction_length = float(np.linalg.norm(inertial_direction))
    if body_length == 0.0 or direction_length == 0.0:
        raise ValueError("attitude alignment vectors must be non-zero")
    alignment = Rotation.align_vectors(
        np.asarray((inertial_direction / direction_length,)),
        np.asarray((body_axis / body_length,)),
    )
    rotation = alignment[0]
    x, y, z, w = rotation.as_quat()
    return np.asarray((w, x, y, z))
