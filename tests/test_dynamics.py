from math import cos, pi, sqrt
import unittest

import numpy as np

from typace.config.physics import (
    EARTH_ATMOSPHERE_LAYER_ALTITUDES_M,
    EARTH_EQUATORIAL_RADIUS_M,
    EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
    EARTH_J2,
    EARTH_MEAN_RADIUS_M,
    EARTH_ROTATION_RATE_RAD_S,
    MOON_GRAVITATIONAL_PARAMETER_M3_S2,
    MOON_MEAN_RADIUS_M,
)
from typace.physics.elements import (
    CartesianState,
    ClassicalElements,
    elements_to_state,
    state_to_elements,
)
from typace.physics.environment import (
    OccludingBody,
    atmospheric_density_kg_m3,
    dynamic_pressure_pa,
    is_fully_eclipsed,
    stagnation_heat_flux_w_m2,
)
from typace.physics.forces import (
    ForceInputs,
    ForceModel,
    central_gravity_acceleration_m_s2,
    total_acceleration_m_s2,
)
from typace.physics.propagation import (
    PropagationContext,
    TranslationalState,
    locate_event_time_s,
    propagate,
)
from typace.physics.bodies import force_model_for
from typace.solar_system import load_solar_system

TWO_BODY_RELATIVE_ENERGY_TOLERANCE = 2.0e-6
TWO_BODY_RELATIVE_MOMENTUM_TOLERANCE = 1.0e-6
J2_RELATIVE_RATE_TOLERANCE = 0.03


def _specific_energy(state: TranslationalState, mu_m3_s2: float) -> float:
    return float(np.dot(state.velocity_m_s, state.velocity_m_s)) / 2.0 - (
        mu_m3_s2 / float(np.linalg.norm(state.position_m))
    )


def _specific_momentum(state: TranslationalState) -> float:
    return float(np.linalg.norm(np.cross(state.position_m, state.velocity_m_s)))


class EnvironmentTests(unittest.TestCase):
    def test_atmosphere_is_continuous_and_non_increasing(self) -> None:
        boundary_offset_m = 1.0e-3
        for altitude_m in EARTH_ATMOSPHERE_LAYER_ALTITUDES_M[1:]:
            below = atmospheric_density_kg_m3(altitude_m - boundary_offset_m)
            above = atmospheric_density_kg_m3(altitude_m + boundary_offset_m)
            self.assertAlmostEqual(below, above, delta=below * 1.0e-6)
        sampled = tuple(
            atmospheric_density_kg_m3(float(altitude_m))
            for altitude_m in range(0, 1_100_000, 10_000)
        )
        self.assertTrue(
            all(first >= second for first, second in zip(sampled, sampled[1:]))
        )

    def test_eclipse_dynamic_pressure_and_heat_flux(self) -> None:
        satellite = np.asarray((7_000_000.0, 0.0, 0.0))
        sun = np.asarray((-150_000_000_000.0, 0.0, 0.0))
        earth = OccludingBody(np.zeros(3), EARTH_MEAN_RADIUS_M)
        self.assertTrue(is_fully_eclipsed(satellite, sun, (earth,)))
        relative_velocity = np.asarray((0.0, 7_500.0, 0.0))
        self.assertGreater(dynamic_pressure_pa(0.01, relative_velocity), 0.0)
        self.assertGreater(stagnation_heat_flux_w_m2(0.01, 1.0, relative_velocity), 0.0)


class PropagationTests(unittest.TestCase):
    def test_every_catalog_body_has_a_positive_force_model(self) -> None:
        for body in load_solar_system().bodies:
            with self.subTest(body=body.id):
                model = force_model_for(body.id)
                self.assertGreater(model.gravitational_parameter_m3_s2, 0.0)
                self.assertGreater(model.body_radius_m, 0.0)

    def test_two_body_rk4_conserves_energy_and_momentum_for_100_periods(self) -> None:
        radius_m = 7_000_000.0
        speed_m_s = sqrt(EARTH_GRAVITATIONAL_PARAMETER_M3_S2 / radius_m)
        period_s = 2.0 * pi * sqrt(radius_m**3 / EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
        initial = TranslationalState(
            np.asarray((radius_m, 0.0, 0.0)),
            np.asarray((0.0, speed_m_s, 0.0)),
            1000.0,
        )
        context = PropagationContext(
            ForceModel(
                "earth", EARTH_GRAVITATIONAL_PARAMETER_M3_S2, EARTH_MEAN_RADIUS_M
            ),
            0.0,
            0.0,
            np.zeros(3),
        )

        final = propagate(
            initial,
            context,
            100.0 * period_s,
            period_s / 180.0,
            allow_analytic=False,
        )

        energy_error = abs(
            (
                _specific_energy(final, EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
                - _specific_energy(initial, EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
            )
            / _specific_energy(initial, EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
        )
        momentum_error = abs(
            (_specific_momentum(final) - _specific_momentum(initial))
            / _specific_momentum(initial)
        )
        self.assertLess(energy_error, TWO_BODY_RELATIVE_ENERGY_TOLERANCE)
        self.assertLess(momentum_error, TWO_BODY_RELATIVE_MOMENTUM_TOLERANCE)

    def test_j2_nodal_precession_matches_first_order_rate(self) -> None:
        elements = ClassicalElements(7_000_000.0, 0.01, 0.9, 1.0, 0.4, 0.2)
        cartesian = elements_to_state(elements, EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
        initial = TranslationalState(
            cartesian.position_m, cartesian.velocity_m_s, 800.0
        )
        model = ForceModel(
            "earth",
            EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
            EARTH_MEAN_RADIUS_M,
            EARTH_J2,
            EARTH_EQUATORIAL_RADIUS_M,
            EARTH_ROTATION_RATE_RAD_S,
        )
        context = PropagationContext(model, 0.0, 0.0, np.zeros(3))
        mean_motion_rad_s = sqrt(
            EARTH_GRAVITATIONAL_PARAMETER_M3_S2 / elements.semi_major_axis_m**3
        )
        period_s = 2.0 * pi / mean_motion_rad_s
        duration_s = 20.0 * period_s

        final = propagate(
            initial, context, duration_s, period_s / 240.0, allow_analytic=False
        )
        final_elements = state_to_elements(
            CartesianState(final.position_m, final.velocity_m_s),
            EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
        )
        measured_change = (
            final_elements.ascending_node_rad - elements.ascending_node_rad + pi
        ) % (2.0 * pi) - pi
        measured_rate = measured_change / duration_s
        parameter_m = elements.semi_major_axis_m * (1.0 - elements.eccentricity**2)
        expected_rate = (
            -1.5
            * EARTH_J2
            * mean_motion_rad_s
            * (EARTH_EQUATORIAL_RADIUS_M / parameter_m) ** 2
            * cos(elements.inclination_rad)
        )
        self.assertLess(
            abs((measured_rate - expected_rate) / expected_rate),
            J2_RELATIVE_RATE_TOLERANCE,
        )

    def test_drag_reduces_orbital_energy_and_moon_excludes_earth_terms(self) -> None:
        radius_m = EARTH_MEAN_RADIUS_M + 250_000.0
        velocity_m_s = sqrt(EARTH_GRAVITATIONAL_PARAMETER_M3_S2 / radius_m)
        initial = TranslationalState(
            np.asarray((radius_m, 0.0, 0.0)),
            np.asarray((0.0, velocity_m_s, 0.0)),
            120.0,
        )
        earth_model = ForceModel(
            "earth",
            EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
            EARTH_MEAN_RADIUS_M,
            0.0,
            EARTH_EQUATORIAL_RADIUS_M,
            EARTH_ROTATION_RATE_RAD_S,
        )
        with_drag = PropagationContext(earth_model, 30.0, 2.2, np.zeros(3))
        without_drag = PropagationContext(earth_model, 0.0, 0.0, np.zeros(3))
        period_s = 2.0 * pi * sqrt(radius_m**3 / EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
        dragged = propagate(initial, with_drag, period_s, 10.0, allow_analytic=False)
        control = propagate(initial, without_drag, period_s, 10.0, allow_analytic=False)
        self.assertLess(
            _specific_energy(dragged, EARTH_GRAVITATIONAL_PARAMETER_M3_S2),
            _specific_energy(control, EARTH_GRAVITATIONAL_PARAMETER_M3_S2),
        )

        moon_position = np.asarray((MOON_MEAN_RADIUS_M + 100_000.0, 0.0, 0.0))
        moon_velocity = np.asarray((0.0, 1_600.0, 0.0))
        moon_model = ForceModel(
            "moon",
            MOON_GRAVITATIONAL_PARAMETER_M3_S2,
            MOON_MEAN_RADIUS_M,
            EARTH_J2,
            EARTH_EQUATORIAL_RADIUS_M,
            EARTH_ROTATION_RATE_RAD_S,
        )
        inputs = ForceInputs(100.0, 100.0, 3.0, np.zeros(3))
        np.testing.assert_allclose(
            total_acceleration_m_s2(moon_model, moon_position, moon_velocity, inputs),
            central_gravity_acceleration_m_s2(
                moon_position, MOON_GRAVITATIONAL_PARAMETER_M3_S2
            ),
        )

    def test_finite_thrust_updates_mass_and_event_time_is_refined(self) -> None:
        state = TranslationalState(np.asarray((1.0, 0.0, 0.0)), np.zeros(3), 100.0)
        context = PropagationContext(
            ForceModel("test", 0.0, 1.0),
            0.0,
            0.0,
            np.asarray((10.0, 0.0, 0.0)),
            mass_flow_kg_s=0.1,
        )
        final = propagate(state, context, 10.0, 0.25, allow_analytic=False)
        self.assertAlmostEqual(final.mass_kg, 99.0)
        self.assertGreater(final.velocity_m_s[0], 1.0)

        event_time_s = locate_event_time_s(
            10.0,
            lambda elapsed_s: TranslationalState(
                np.asarray((elapsed_s, 0.0, 0.0)), np.zeros(3), 1.0
            ),
            lambda current: float(current.position_m[0]) - 7.25,
        )
        self.assertAlmostEqual(event_time_s, 7.25, delta=1.0e-3)


if __name__ == "__main__":
    unittest.main()
