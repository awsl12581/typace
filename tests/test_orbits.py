from math import pi, sqrt
import unittest

import numpy as np

from typace.config.physics import EARTH_GRAVITATIONAL_PARAMETER_M3_S2
from typace.physics.elements import (
    CartesianState,
    ClassicalElements,
    analytic_kepler_step,
    elements_to_state,
    state_to_elements,
)
from typace.physics.frames import radial_transverse_normal_basis
from typace.physics.kepler import solve_eccentric_anomaly
from typace.physics.lambert import LambertFailure, solve_lambert
from typace.physics.soi import OriginState, change_primary, sphere_of_influence_radius_m


class OrbitalElementsTests(unittest.TestCase):
    def test_elements_round_trip_for_regular_and_singular_orbits(self) -> None:
        cases = (
            ClassicalElements(7_000_000.0, 0.0, 0.0, 0.0, 0.0, 1.2),
            ClassicalElements(8_000_000.0, 0.2, 0.0, 0.0, 0.7, 2.1),
            ClassicalElements(9_000_000.0, 0.01, 0.6, 1.1, 0.8, 0.3),
            ClassicalElements(7_500_000.0, 0.03, pi / 2.0, 2.2, 1.4, 5.0),
        )
        for elements in cases:
            with self.subTest(elements=elements):
                initial = elements_to_state(
                    elements, EARTH_GRAVITATIONAL_PARAMETER_M3_S2
                )
                recovered = state_to_elements(
                    initial, EARTH_GRAVITATIONAL_PARAMETER_M3_S2
                )
                final = elements_to_state(
                    recovered, EARTH_GRAVITATIONAL_PARAMETER_M3_S2
                )
                np.testing.assert_allclose(
                    final.position_m, initial.position_m, rtol=0.0, atol=1.0e-5
                )
                np.testing.assert_allclose(
                    final.velocity_m_s, initial.velocity_m_s, rtol=0.0, atol=1.0e-8
                )

    def test_analytic_step_closes_after_one_hundred_periods(self) -> None:
        elements = ClassicalElements(7_000_000.0, 0.05, 0.7, 0.4, 1.2, 0.8)
        initial = elements_to_state(elements, EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
        period_s = (
            2.0
            * pi
            * sqrt(elements.semi_major_axis_m**3 / EARTH_GRAVITATIONAL_PARAMETER_M3_S2)
        )

        final = analytic_kepler_step(
            initial, EARTH_GRAVITATIONAL_PARAMETER_M3_S2, 100.0 * period_s
        )

        np.testing.assert_allclose(
            final.position_m, initial.position_m, rtol=0.0, atol=1.0e-4
        )
        np.testing.assert_allclose(
            final.velocity_m_s, initial.velocity_m_s, rtol=0.0, atol=1.0e-7
        )

    def test_kepler_solver_and_rtn_basis_reject_invalid_states(self) -> None:
        eccentric_anomaly = solve_eccentric_anomaly(2.1, 0.72)
        self.assertAlmostEqual(
            eccentric_anomaly - 0.72 * np.sin(eccentric_anomaly), 2.1
        )
        with self.assertRaises(ValueError):
            solve_eccentric_anomaly(0.0, 1.0)
        with self.assertRaises(ValueError):
            radial_transverse_normal_basis(np.zeros(3), np.zeros(3))


class LambertAndSphereOfInfluenceTests(unittest.TestCase):
    def test_izzo2015_matches_quarter_circular_transfer(self) -> None:
        radius_m = 7_000_000.0
        speed_m_s = sqrt(EARTH_GRAVITATIONAL_PARAMETER_M3_S2 / radius_m)
        period_s = 2.0 * pi * sqrt(radius_m**3 / EARTH_GRAVITATIONAL_PARAMETER_M3_S2)

        solution = solve_lambert(
            EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
            np.asarray((radius_m, 0.0, 0.0)),
            np.asarray((0.0, radius_m, 0.0)),
            period_s / 4.0,
            prograde=True,
            low_path=True,
        )

        np.testing.assert_allclose(
            solution.departure_velocity_m_s,
            np.asarray((0.0, speed_m_s, 0.0)),
            rtol=1.0e-7,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            solution.arrival_velocity_m_s,
            np.asarray((-speed_m_s, 0.0, 0.0)),
            rtol=1.0e-7,
            atol=1.0e-6,
        )
        with self.assertRaises(LambertFailure):
            solve_lambert(
                EARTH_GRAVITATIONAL_PARAMETER_M3_S2,
                np.zeros(3),
                np.ones(3),
                0.0,
                prograde=True,
                low_path=True,
            )

    def test_soi_change_preserves_inertial_state(self) -> None:
        old_origin = OriginState(
            np.asarray((10.0, 20.0, 30.0)), np.asarray((1.0, 2.0, 3.0))
        )
        new_origin = OriginState(
            np.asarray((-5.0, 4.0, 8.0)), np.asarray((-1.0, 0.5, 1.0))
        )
        old_position = np.asarray((2.0, 3.0, 4.0))
        old_velocity = np.asarray((0.2, 0.3, 0.4))

        new_position, new_velocity = change_primary(
            old_position, old_velocity, old_origin, new_origin
        )

        np.testing.assert_allclose(
            new_position + new_origin.position_m,
            old_position + old_origin.position_m,
        )
        np.testing.assert_allclose(
            new_velocity + new_origin.velocity_m_s,
            old_velocity + old_origin.velocity_m_s,
        )
        self.assertGreater(
            sphere_of_influence_radius_m(384_400_000.0, 7.342e22, 5.9722e24),
            60_000_000.0,
        )


if __name__ == "__main__":
    unittest.main()
