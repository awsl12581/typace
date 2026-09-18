"""Strict, offline-only loading for satellite JSON catalogs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from astropy.time import Time

from typace.config.astronomy import configure_offline_astronomy
from typace.config.simulation import (
    MAX_CATALOG_BYTES,
    MAX_CATALOG_OBJECTS,
    MAX_CATALOG_STRING_LENGTH,
)
from typace.privacy import safe_catalog_location, sanitize_text, sanitize_url
from typace.solar_system import load_solar_system
from typace.satellites.definition import (
    AutonomyDefinition,
    KeplerianDefinition,
    PowerDefinition,
    PropulsionDefinition,
    ProvenanceKind,
    SatelliteCatalog,
    SatelliteDefinition,
    SourceRecord,
    StateVectorDefinition,
    Vector3,
)

_BUNDLED_CATALOG = Path(__file__).with_name("data") / "catalog.json"
_ALLOWED_PRIMARY_BODIES = frozenset(
    body.id for body in load_solar_system().bodies if body.mass_kg > 0.0
)
_ALLOWED_COMMANDS = frozenset(
    (
        "maintain_orbit",
        "set_apsides",
        "set_inclination",
        "transfer_primary",
        "return_stable_orbit",
        "deorbit",
        "avoid_collision",
    )
)
_REQUIRED_PROVENANCE_FIELDS = frozenset(
    (
        "initial_orbit",
        "dry_mass_kg",
        "propellant",
        "inertia",
        "aerodynamics",
        "propulsion",
        "reaction_wheels",
        "power",
        "limits",
        "autonomy",
    )
)
_SATELLITE_KEYS = frozenset(
    (
        "id",
        "display_name",
        "operator",
        "primary_body_id",
        "state_vector",
        "keplerian",
        "dry_mass_kg",
        "main_propellant_mass_kg",
        "rcs_propellant_mass_kg",
        "inertia_diagonal_kg_m2",
        "collision_radius_m",
        "drag_area_m2",
        "drag_coefficient",
        "nose_radius_m",
        "propulsion",
        "reaction_wheels",
        "power",
        "maximum_dynamic_pressure_pa",
        "maximum_heat_flux_w_m2",
        "autonomy",
        "sources",
        "field_provenance",
    )
)
_DOCUMENT_KEYS = frozenset(("schema_version", "scenario_epoch", "satellites"))


@dataclass(frozen=True, slots=True)
class CatalogIssue:
    location: str
    object_id: str
    field_path: str
    message: str

    def render(self) -> str:
        return sanitize_text(
            f"{self.location} [{self.object_id}] {self.field_path}: {self.message}"
        )


class CatalogError(ValueError):
    """One or more safe, structured catalog diagnostics."""

    def __init__(self, issues: Sequence[CatalogIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.render() for issue in self.issues))


class _InvalidField(ValueError):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(message)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise _InvalidField(path, "must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise _InvalidField(path, "must be an array")
    return cast(Sequence[object], value)


def _text(data: Mapping[str, object], key: str, path: str) -> str:
    value = data.get(key)
    field_path = f"{path}.{key}"
    if not isinstance(value, str) or not value:
        raise _InvalidField(field_path, "must be non-empty text")
    if len(value) > MAX_CATALOG_STRING_LENGTH:
        raise _InvalidField(field_path, "is too long")
    return value


def _number(
    data: Mapping[str, object],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = data.get(key)
    field_path = f"{path}.{key}"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _InvalidField(field_path, "must be numeric")
    result = float(value)
    if not isfinite(result):
        raise _InvalidField(field_path, "must be finite")
    if minimum is not None and result < minimum:
        raise _InvalidField(field_path, f"must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise _InvalidField(field_path, f"must be at most {maximum}")
    return result


def _vector3(
    data: Mapping[str, object], key: str, path: str, *, minimum: float | None = None
) -> Vector3:
    field_path = f"{path}.{key}"
    values = _sequence(data.get(key), field_path)
    if len(values) != 3:
        raise _InvalidField(field_path, "must contain three values")
    parsed: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _InvalidField(f"{field_path}[{index}]", "must be numeric")
        number = float(value)
        if not isfinite(number):
            raise _InvalidField(f"{field_path}[{index}]", "must be finite")
        if minimum is not None and number < minimum:
            raise _InvalidField(f"{field_path}[{index}]", "is below the minimum")
        parsed.append(number)
    return parsed[0], parsed[1], parsed[2]


def _time(data: Mapping[str, object], key: str, path: str) -> Time:
    value = _text(data, key, path)
    if not value.endswith("Z"):
        raise _InvalidField(f"{path}.{key}", "must be RFC 3339 UTC ending in Z")
    try:
        return Time(value, scale="utc")
    except ValueError as error:
        raise _InvalidField(f"{path}.{key}", "must be RFC 3339 UTC") from error


def _parse_source(value: object, path: str) -> SourceRecord:
    data = _mapping(value, path)
    expected = frozenset(
        ("id", "kind", "organization", "document_title", "published_date", "url")
    )
    unknown = frozenset(data) - expected
    if unknown:
        raise _InvalidField(path, f"unknown fields: {', '.join(sorted(unknown))}")
    try:
        kind = ProvenanceKind(_text(data, "kind", path))
    except ValueError as error:
        raise _InvalidField(
            f"{path}.kind", "must be observed, derived, or assumed"
        ) from error
    raw_url = data.get("url")
    if not isinstance(raw_url, str):
        raise _InvalidField(f"{path}.url", "must be text")
    url: str | None = None
    if raw_url:
        parts = urlsplit(raw_url)
        has_private_url_parts = parts.username is not None or bool(
            parts.query or parts.fragment
        )
        if parts.scheme != "https" or not parts.hostname or has_private_url_parts:
            raise _InvalidField(
                f"{path}.url", "must be a public HTTPS URL without private parts"
            )
        url = sanitize_url(raw_url)
    elif kind is not ProvenanceKind.ASSUMED:
        raise _InvalidField(f"{path}.url", "is required for observed and derived data")
    return SourceRecord(
        _text(data, "id", path),
        kind,
        _text(data, "organization", path),
        _text(data, "document_title", path),
        _text(data, "published_date", path),
        url,
    )


def _parse_initial_orbit(
    data: Mapping[str, object], path: str
) -> StateVectorDefinition | KeplerianDefinition:
    has_vector = data.get("state_vector") is not None
    has_keplerian = data.get("keplerian") is not None
    if has_vector == has_keplerian:
        raise _InvalidField(
            path, "must define exactly one of state_vector or keplerian"
        )
    key = "state_vector" if has_vector else "keplerian"
    orbit_path = f"{path}.{key}"
    orbit = _mapping(data.get(key), orbit_path)
    epoch = _time(orbit, "epoch", orbit_path)
    frame = _text(orbit, "frame", orbit_path)
    supported_frames = frozenset(
        f"{body_id}_j2000" for body_id in _ALLOWED_PRIMARY_BODIES
    )
    if frame not in supported_frames:
        raise _InvalidField(f"{orbit_path}.frame", "is not a supported inertial frame")
    if has_vector:
        return StateVectorDefinition(
            epoch,
            frame,
            _vector3(orbit, "position_m", orbit_path),
            _vector3(orbit, "velocity_m_s", orbit_path),
        )
    return KeplerianDefinition(
        epoch,
        frame,
        _number(orbit, "semi_major_axis_m", orbit_path, minimum=1.0),
        _number(orbit, "eccentricity", orbit_path, minimum=0.0, maximum=0.999999),
        _number(orbit, "inclination_deg", orbit_path, minimum=0.0, maximum=180.0),
        _number(orbit, "ascending_node_deg", orbit_path, minimum=0.0, maximum=360.0),
        _number(
            orbit, "periapsis_argument_deg", orbit_path, minimum=0.0, maximum=360.0
        ),
        _number(orbit, "mean_anomaly_deg", orbit_path, minimum=0.0, maximum=360.0),
    )


def _parse_satellite(value: object, index: int) -> SatelliteDefinition:
    path = f"satellites[{index}]"
    data = _mapping(value, path)
    unknown = frozenset(data) - _SATELLITE_KEYS
    if unknown:
        raise _InvalidField(path, f"unknown fields: {', '.join(sorted(unknown))}")
    primary_body_id = _text(data, "primary_body_id", path)
    if primary_body_id not in _ALLOWED_PRIMARY_BODIES:
        raise _InvalidField(
            f"{path}.primary_body_id", "is not in the solar-system catalog"
        )

    propulsion_data = _mapping(data.get("propulsion"), f"{path}.propulsion")
    propulsion = PropulsionDefinition(
        _number(propulsion_data, "main_thrust_n", f"{path}.propulsion", minimum=0.0),
        _number(
            propulsion_data,
            "main_specific_impulse_s",
            f"{path}.propulsion",
            minimum=1.0,
        ),
        _number(
            propulsion_data,
            "main_minimum_throttle",
            f"{path}.propulsion",
            minimum=0.0,
            maximum=1.0,
        ),
        _vector3(propulsion_data, "main_body_axis", f"{path}.propulsion"),
        _number(propulsion_data, "rcs_thrust_n", f"{path}.propulsion", minimum=0.0),
        _number(
            propulsion_data, "rcs_specific_impulse_s", f"{path}.propulsion", minimum=1.0
        ),
        _number(
            propulsion_data, "rcs_maximum_torque_n_m", f"{path}.propulsion", minimum=0.0
        ),
    )

    wheel_data = _mapping(data.get("reaction_wheels"), f"{path}.reaction_wheels")
    power_data = _mapping(data.get("power"), f"{path}.power")
    power = PowerDefinition(
        _number(power_data, "panel_area_m2", f"{path}.power", minimum=0.0),
        _number(
            power_data, "panel_efficiency", f"{path}.power", minimum=0.0, maximum=1.0
        ),
        _vector3(power_data, "panel_body_normal", f"{path}.power"),
        _number(power_data, "battery_capacity_j", f"{path}.power", minimum=1.0),
        _number(power_data, "initial_battery_energy_j", f"{path}.power", minimum=0.0),
        _number(power_data, "base_power_w", f"{path}.power", minimum=0.0),
        _number(power_data, "computer_power_w", f"{path}.power", minimum=0.0),
        _number(power_data, "sensor_power_w", f"{path}.power", minimum=0.0),
        _number(power_data, "actuator_power_w", f"{path}.power", minimum=0.0),
    )
    if power.initial_battery_energy_j > power.battery_capacity_j:
        raise _InvalidField(
            f"{path}.power.initial_battery_energy_j", "exceeds capacity"
        )

    autonomy_data = _mapping(data.get("autonomy"), f"{path}.autonomy")
    command_values = _sequence(
        autonomy_data.get("allowed_commands"), f"{path}.autonomy.allowed_commands"
    )
    commands = tuple(
        value
        for value in command_values
        if isinstance(value, str) and value in _ALLOWED_COMMANDS
    )
    if len(commands) != len(command_values):
        raise _InvalidField(
            f"{path}.autonomy.allowed_commands", "contains an unknown command"
        )
    autonomy = AutonomyDefinition(
        _number(
            autonomy_data,
            "target_periapsis_altitude_m",
            f"{path}.autonomy",
            minimum=0.0,
        ),
        _number(
            autonomy_data, "target_apoapsis_altitude_m", f"{path}.autonomy", minimum=0.0
        ),
        _number(
            autonomy_data,
            "target_inclination_deg",
            f"{path}.autonomy",
            minimum=0.0,
            maximum=180.0,
        ),
        _number(autonomy_data, "altitude_tolerance_m", f"{path}.autonomy", minimum=0.0),
        _number(
            autonomy_data,
            "minimum_main_propellant_reserve_kg",
            f"{path}.autonomy",
            minimum=0.0,
        ),
        _number(autonomy_data, "avoidance_distance_m", f"{path}.autonomy", minimum=0.0),
        commands,
    )

    sources = tuple(
        _parse_source(item, f"{path}.sources[{source_index}]")
        for source_index, item in enumerate(
            _sequence(data.get("sources"), f"{path}.sources")
        )
    )
    source_ids = {source.id for source in sources}
    if len(source_ids) != len(sources):
        raise _InvalidField(f"{path}.sources", "source IDs must be unique")
    provenance_data = _mapping(data.get("field_provenance"), f"{path}.field_provenance")
    missing_provenance = _REQUIRED_PROVENANCE_FIELDS - frozenset(provenance_data)
    if missing_provenance:
        raise _InvalidField(
            f"{path}.field_provenance",
            f"missing fields: {', '.join(sorted(missing_provenance))}",
        )
    field_provenance: list[tuple[str, str]] = []
    for field_name, source_id in sorted(provenance_data.items()):
        if not isinstance(source_id, str) or source_id not in source_ids:
            raise _InvalidField(
                f"{path}.field_provenance.{field_name}", "references an unknown source"
            )
        field_provenance.append((field_name, source_id))

    return SatelliteDefinition(
        id=_text(data, "id", path),
        display_name=_text(data, "display_name", path),
        operator=_text(data, "operator", path),
        primary_body_id=primary_body_id,
        initial_orbit=_parse_initial_orbit(data, path),
        dry_mass_kg=_number(data, "dry_mass_kg", path, minimum=1.0),
        main_propellant_mass_kg=_number(
            data, "main_propellant_mass_kg", path, minimum=0.0
        ),
        rcs_propellant_mass_kg=_number(
            data, "rcs_propellant_mass_kg", path, minimum=0.0
        ),
        inertia_diagonal_kg_m2=_vector3(
            data, "inertia_diagonal_kg_m2", path, minimum=1.0
        ),
        collision_radius_m=_number(data, "collision_radius_m", path, minimum=0.01),
        drag_area_m2=_number(data, "drag_area_m2", path, minimum=0.0),
        drag_coefficient=_number(data, "drag_coefficient", path, minimum=0.0),
        nose_radius_m=_number(data, "nose_radius_m", path, minimum=0.01),
        propulsion=propulsion,
        reaction_wheel_maximum_torque_n_m=_vector3(
            wheel_data, "maximum_torque_n_m", f"{path}.reaction_wheels", minimum=0.0
        ),
        reaction_wheel_maximum_momentum_n_m_s=_vector3(
            wheel_data, "maximum_momentum_n_m_s", f"{path}.reaction_wheels", minimum=0.0
        ),
        power=power,
        maximum_dynamic_pressure_pa=_number(
            data, "maximum_dynamic_pressure_pa", path, minimum=0.0
        ),
        maximum_heat_flux_w_m2=_number(
            data, "maximum_heat_flux_w_m2", path, minimum=0.0
        ),
        autonomy=autonomy,
        sources=sources,
        field_provenance=tuple(field_provenance),
    )


def _catalog_files(root: Path, alias: str) -> tuple[tuple[Path, str], ...]:
    raw = str(root)
    if raw.startswith(("http:", "https:", "data:", "ftp:")):
        raise CatalogError(
            (
                CatalogIssue(
                    f"{alias}:<root>", "catalog", "$", "URI inputs are not allowed"
                ),
            )
        )
    if root.is_file():
        return ((root, safe_catalog_location(alias, Path(root.name))),)
    if root.is_dir():
        files = tuple(sorted(root.glob("*.json"), key=lambda path: path.name))
        if not files:
            raise CatalogError(
                (
                    CatalogIssue(
                        f"{alias}:<root>",
                        "catalog",
                        "$",
                        "catalog directory contains no JSON files",
                    ),
                )
            )
        return tuple(
            (path, safe_catalog_location(alias, path.relative_to(root)))
            for path in files
        )
    raise CatalogError(
        (
            CatalogIssue(
                f"{alias}:<root>", "catalog", "$", "catalog path does not exist"
            ),
        )
    )


def _read_document(path: Path, location: str) -> Mapping[str, object]:
    if path.stat().st_size > MAX_CATALOG_BYTES:
        raise CatalogError(
            (CatalogIssue(location, "catalog", "$", "catalog file is too large"),)
        )
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        return _mapping(raw, "$")
    except (OSError, json.JSONDecodeError, UnicodeError, _InvalidField) as error:
        message = (
            error.message if isinstance(error, _InvalidField) else "cannot parse JSON"
        )
        raise CatalogError((CatalogIssue(location, "catalog", "$", message),)) from None


def load_catalog(paths: Sequence[Path] = ()) -> SatelliteCatalog:
    """Load the bundled catalog and optional local JSON files or directories."""

    configure_offline_astronomy()
    roots = ((_BUNDLED_CATALOG, "bundled"),) + tuple(
        (path, f"catalog-{index}") for index, path in enumerate(paths, start=1)
    )
    documents = tuple(
        (document_path, location)
        for root, alias in roots
        for document_path, location in _catalog_files(root, alias)
    )
    issues: list[CatalogIssue] = []
    scenario_epoch: Time | None = None
    satellites: list[SatelliteDefinition] = []
    seen_ids: set[str] = set()
    for document_path, location in documents:
        try:
            document = _read_document(document_path, location)
            unknown_document_fields = frozenset(document) - _DOCUMENT_KEYS
            if unknown_document_fields:
                raise _InvalidField(
                    "$",
                    "unknown fields: " + ", ".join(sorted(unknown_document_fields)),
                )
            if document.get("schema_version") != 1:
                raise _InvalidField("$.schema_version", "must equal 1")
            document_epoch = _time(document, "scenario_epoch", "$")
            if scenario_epoch is None:
                scenario_epoch = document_epoch
            elif document_epoch != scenario_epoch:
                raise _InvalidField(
                    "$.scenario_epoch", "must match the bundled scenario epoch"
                )
            values = _sequence(document.get("satellites"), "$.satellites")
            if len(values) > MAX_CATALOG_OBJECTS:
                raise _InvalidField("$.satellites", "contains too many objects")
            for index, value in enumerate(values):
                object_id = f"satellites[{index}]"
                try:
                    satellite = _parse_satellite(value, index)
                    object_id = satellite.id
                    if satellite.id in seen_ids:
                        raise _InvalidField(
                            f"satellites[{index}].id", "duplicate satellite ID"
                        )
                    expected_frame = f"{satellite.primary_body_id}_j2000"
                    if satellite.initial_orbit.frame != expected_frame:
                        raise _InvalidField(
                            f"satellites[{index}].initial_orbit.frame",
                            f"must be {expected_frame}",
                        )
                    seen_ids.add(satellite.id)
                    satellites.append(satellite)
                    if len(satellites) > MAX_CATALOG_OBJECTS:
                        raise _InvalidField(
                            "$.satellites", "contains too many aggregate objects"
                        )
                except _InvalidField as error:
                    issues.append(
                        CatalogIssue(location, object_id, error.path, error.message)
                    )
        except CatalogError as error:
            issues.extend(error.issues)
        except _InvalidField as error:
            issues.append(CatalogIssue(location, "catalog", error.path, error.message))
    if issues:
        raise CatalogError(issues)
    if scenario_epoch is None:
        raise CatalogError(
            (
                CatalogIssue(
                    "bundled:<root>", "catalog", "$.scenario_epoch", "is required"
                ),
            )
        )
    return SatelliteCatalog(scenario_epoch, tuple(satellites))
