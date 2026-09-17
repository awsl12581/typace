import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers

from typace.privacy import sanitize_text, sanitize_url
from typace.satellites import CatalogError, load_catalog
from typace.satellites.definition import KeplerianDefinition, ProvenanceKind

_BUNDLED_CATALOG = (
    Path(__file__).parents[1] / "typace" / "satellites" / "data" / "catalog.json"
)


class SatelliteCatalogTests(unittest.TestCase):
    def test_bundled_catalog_loads_six_data_driven_satellites_offline(self) -> None:
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("network")
        ):
            catalog = load_catalog()

        self.assertEqual(
            tuple(satellite.id for satellite in catalog.satellites),
            ("iss", "gps-biir-2", "goes-16", "lro", "danuri", "queqiao-2"),
        )
        self.assertFalse(iers.conf.auto_download)
        self.assertEqual(iers.conf.iers_degraded_accuracy, "error")
        self.assertEqual(solar_system_ephemeris.get(), "builtin")

        iss = catalog.satellite("iss")
        lro = catalog.satellite("lro")
        self.assertIsInstance(iss.initial_orbit, KeplerianDefinition)
        self.assertIsInstance(lro.initial_orbit, KeplerianDefinition)
        assert isinstance(iss.initial_orbit, KeplerianDefinition)
        assert isinstance(lro.initial_orbit, KeplerianDefinition)
        self.assertEqual(iss.initial_orbit.frame, "earth_j2000")
        self.assertEqual(lro.initial_orbit.frame, "moon_j2000")
        self.assertEqual(iss.initial_orbit.semi_major_axis_m, 6_786_000.0)
        self.assertEqual(lro.initial_orbit.semi_major_axis_m, 1_787_400.0)
        self.assertTrue(
            all(
                source.kind in ProvenanceKind
                for satellite in catalog.satellites
                for source in satellite.sources
            )
        )

    def test_external_directory_is_sorted_and_duplicate_ids_are_rejected(self) -> None:
        document = json.loads(_BUNDLED_CATALOG.read_text(encoding="utf-8"))
        duplicate = {
            "schema_version": 1,
            "scenario_epoch": document["scenario_epoch"],
            "satellites": [document["satellites"][0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.json").write_text(json.dumps(duplicate), encoding="utf-8")
            (root / "a.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenario_epoch": document["scenario_epoch"],
                        "satellites": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CatalogError) as raised:
                load_catalog((root,))

        message = str(raised.exception)
        self.assertIn("catalog-1:b.json", message)
        self.assertIn("duplicate satellite ID", message)
        self.assertNotIn(directory, message)

    def test_uri_and_mutually_present_initial_states_are_rejected(self) -> None:
        with self.assertRaisesRegex(CatalogError, "URI inputs are not allowed"):
            load_catalog((Path("https://invalid.example/catalog.json"),))

        document = json.loads(_BUNDLED_CATALOG.read_text(encoding="utf-8"))
        satellite = document["satellites"][0]
        satellite["state_vector"] = {
            "epoch": document["scenario_epoch"],
            "frame": "earth_j2000",
            "position_m": [6_786_000.0, 0.0, 0.0],
            "velocity_m_s": [0.0, 7_660.0, 0.0],
        }
        external = {
            "schema_version": 1,
            "scenario_epoch": document["scenario_epoch"],
            "satellites": [satellite],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(external), encoding="utf-8")
            with self.assertRaisesRegex(
                CatalogError, "exactly one of state_vector or keplerian"
            ):
                load_catalog((path,))

    def test_diagnostics_and_urls_remove_sensitive_parts(self) -> None:
        text = sanitize_text(
            "failed at /private/example/catalog.json for user@example.test "
            "token=secret-value"
        )
        self.assertNotIn("/private/example", text)
        self.assertNotIn("user@example.test", text)
        self.assertNotIn("secret-value", text)
        self.assertEqual(
            sanitize_url("https://user:pass@example.test/source?q=secret#person"),
            "https://example.test/source",
        )

    def test_empty_directory_and_unknown_document_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(CatalogError, "contains no JSON files"):
                load_catalog((root,))

            (root / "invalid.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenario_epoch": "2026-09-17T00:00:00Z",
                        "satellites": [],
                        "unexpected": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CatalogError, "unknown fields: unexpected"):
                load_catalog((root,))


if __name__ == "__main__":
    unittest.main()
