"""Typed celestial-system catalogs loaded from JSON data."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

type RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, factor: float) -> "Vec3":
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return self.dot(self) ** 0.5

    def unit(self) -> "Vec3":
        length = self.norm()
        return self * (1.0 / length) if length else Vec3()


@dataclass(frozen=True, slots=True)
class TextureBand:
    latitude_min_deg: float
    latitude_max_deg: float
    color: RGB


@dataclass(frozen=True, slots=True)
class TextureEllipse:
    latitude_deg: float
    longitude_deg: float
    latitude_radius_deg: float
    longitude_radius_deg: float
    color: RGB | None
    rim: RGB | None


@dataclass(frozen=True, slots=True)
class TextureRegion:
    kind: str
    vertices: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class StarTexture:
    core: RGB
    surface: RGB
    limb: RGB
    spot: RGB
    granulation: float
    seed: int


@dataclass(frozen=True, slots=True)
class Texture:
    base: RGB
    bands: tuple[TextureBand, ...]
    continents: tuple[TextureEllipse, ...]
    craters: tuple[TextureEllipse, ...]
    spots: tuple[TextureEllipse, ...]
    regions: tuple[TextureRegion, ...]
    biomes: dict[str, RGB]
    limb_tint: RGB | None
    star: StarTexture | None


@dataclass(frozen=True, slots=True)
class RingBand:
    inner_radius_m: float
    outer_radius_m: float
    color: RGB


@dataclass(frozen=True, slots=True)
class CelestialBody:
    id: str
    name: str
    body_type: str
    parent_id: str | None
    radius_m: float
    semimajor_axis_m: float
    eccentricity: float
    inclination_deg: float
    ascending_node_deg: float
    periapsis_argument_deg: float
    orbit_period_days: float
    rotation_period_hours: float
    axial_tilt_deg: float
    mean_anomaly_at_epoch_deg: float | None
    tidally_locked: bool
    color: RGB
    texture: Texture | None
    rings: tuple[RingBand, ...]


@dataclass(frozen=True, slots=True)
class CelestialSystem:
    name: str
    bodies: tuple[CelestialBody, ...]

    def body(self, body_id: str) -> CelestialBody:
        for body in self.bodies:
            if body.id == body_id:
                return body
        raise KeyError(body_id)

    def parent_of(self, body: CelestialBody) -> CelestialBody | None:
        if body.parent_id is None:
            return self.bodies[0] if body is not self.bodies[0] else None
        return self.body(body.parent_id)

    def children_of(self, parent: CelestialBody) -> tuple[CelestialBody, ...]:
        return tuple(body for body in self.bodies if body.parent_id == parent.id)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _items(value: object, context: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return cast(list[object], value)


def _number(data: dict[str, object], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _text(data: dict[str, object], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value


def parse_color(value: str, fallback: RGB = (180, 180, 180)) -> RGB:
    if not value:
        return fallback
    if len(value) not in (4, 7) or not value.startswith("#"):
        raise ValueError(f"invalid color: {value!r}")
    digits = value[1:]
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    try:
        return (
            int(digits[0:2], 16),
            int(digits[2:4], 16),
            int(digits[4:6], 16),
        )
    except ValueError as error:
        raise ValueError(f"invalid color: {value!r}") from error


def _ellipse(value: object, fallback: RGB) -> TextureEllipse:
    data = _mapping(value, "texture ellipse")
    color = _text(data, "color")
    rim = _text(data, "rim")
    latitude_radius = _number(data, "latR")
    longitude_radius = _number(data, "lonR")
    if latitude_radius <= 0 or longitude_radius <= 0:
        raise ValueError("texture ellipse radii must be positive")
    return TextureEllipse(
        latitude_deg=_number(data, "lat"),
        longitude_deg=_number(data, "lon"),
        latitude_radius_deg=latitude_radius,
        longitude_radius_deg=longitude_radius,
        color=parse_color(color, fallback) if color else None,
        rim=parse_color(rim, fallback) if rim else None,
    )


def _texture(value: object, fallback: RGB) -> Texture | None:
    if value is None:
        return None
    data = _mapping(value, "texture")
    base = parse_color(_text(data, "base"), fallback)
    bands = tuple(
        TextureBand(
            _number(item, "latMin"),
            _number(item, "latMax"),
            parse_color(_text(item, "color"), base),
        )
        for raw in _items(data.get("bands"), "texture bands")
        for item in (_mapping(raw, "texture band"),)
    )
    continents = tuple(
        _ellipse(raw, base)
        for raw in _items(data.get("continents"), "texture continents")
    )
    craters = tuple(
        _ellipse(raw, (239, 234, 224))
        for raw in _items(data.get("craters"), "texture craters")
    )
    spots = tuple(
        _ellipse(raw, base) for raw in _items(data.get("spots"), "texture spots")
    )

    regions: tuple[TextureRegion, ...] = ()
    biomes: dict[str, RGB] = {}
    mask_value = data.get("mask")
    if mask_value is not None:
        mask = _mapping(mask_value, "texture mask")
        biome_data = _mapping(mask.get("biomes", {}), "texture biomes")
        biomes = {
            key: parse_color(cast(str, value), base)
            for key, value in biome_data.items()
            if isinstance(value, str)
        }
        parsed_regions: list[TextureRegion] = []
        for raw_region in _items(mask.get("polys"), "texture polygons"):
            region = _mapping(raw_region, "texture polygon")
            vertices = tuple(
                (_number(point, "lat"), _number(point, "lon"))
                for raw_point in _items(region.get("vertices"), "polygon vertices")
                for point in (_mapping(raw_point, "polygon vertex"),)
            )
            if len(vertices) < 3 or _text(region, "kind") not in biomes:
                raise ValueError(
                    "texture polygon requires a known biome and 3 vertices"
                )
            parsed_regions.append(TextureRegion(_text(region, "kind"), vertices))
        regions = tuple(parsed_regions)

    limb_text = _text(data, "limbTint")
    limb_tint = parse_color(limb_text, base) if limb_text else None
    star_value = data.get("star")
    star = None
    if star_value is not None:
        item = _mapping(star_value, "star texture")
        surface = parse_color(_text(item, "surface"), base)
        granulation = _number(item, "granulation")
        if not 0 <= granulation <= 1:
            raise ValueError("star granulation must be between 0 and 1")
        star = StarTexture(
            core=parse_color(_text(item, "core"), surface),
            surface=surface,
            limb=parse_color(_text(item, "limb"), surface),
            spot=parse_color(_text(item, "spot"), surface),
            granulation=granulation,
            seed=int(_number(item, "seed")),
        )
    return Texture(
        base,
        bands,
        continents,
        craters,
        spots,
        regions,
        biomes,
        limb_tint,
        star,
    )


def _body(value: object) -> CelestialBody:
    data = _mapping(value, "body")
    body_id = _text(data, "id")
    color = parse_color(_text(data, "color"))
    parent_id = _text(data, "parentId") or None
    mean_anomaly = data.get("meanAnomalyAtEpoch")
    if mean_anomaly is not None and not isinstance(mean_anomaly, int | float):
        raise ValueError(f"{body_id}: meanAnomalyAtEpoch must be numeric")
    rings = tuple(
        RingBand(
            _number(item, "innerRadiusKm") * 1000.0,
            _number(item, "outerRadiusKm") * 1000.0,
            parse_color(_text(item, "color"), color),
        )
        for raw in _items(data.get("rings"), "rings")
        for item in (_mapping(raw, "ring"),)
    )
    return CelestialBody(
        id=body_id,
        name=_text(data, "englishName", _text(data, "name", body_id)),
        body_type=_text(data, "bodyType"),
        parent_id=parent_id,
        radius_m=_number(data, "meanRadius") * 1000.0,
        semimajor_axis_m=_number(data, "semimajorAxis") * 1000.0,
        eccentricity=_number(data, "eccentricity"),
        inclination_deg=_number(data, "inclination"),
        ascending_node_deg=_number(data, "longitudeOfAscendingNode"),
        periapsis_argument_deg=_number(data, "argumentOfPeriapsis"),
        orbit_period_days=_number(data, "sideralOrbit"),
        rotation_period_hours=_number(data, "sideralRotation"),
        axial_tilt_deg=_number(data, "axialTilt"),
        mean_anomaly_at_epoch_deg=(
            float(mean_anomaly) if isinstance(mean_anomaly, int | float) else None
        ),
        tidally_locked=data.get("tidallyLocked") is True,
        color=color,
        texture=_texture(data.get("texture"), color),
        rings=rings,
    )


def load_system(path: Path) -> CelestialSystem:
    """Load and validate a celestial-system catalog."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    data = _mapping(raw, "celestial system")
    bodies = tuple(_body(item) for item in _items(data.get("bodies"), "bodies"))
    if not bodies or bodies[0].body_type != "Star":
        raise ValueError("the first body must be the system star")
    ids = {body.id for body in bodies}
    if len(ids) != len(bodies):
        raise ValueError("body IDs must be unique")
    for body in bodies:
        if body.parent_id is not None and body.parent_id not in ids:
            raise ValueError(f"{body.id}: unknown parent {body.parent_id}")
        if body.eccentricity < 0 or body.eccentricity >= 1:
            raise ValueError(f"{body.id}: only elliptical orbits are supported")
        seen: set[str] = set()
        current = body
        while current.parent_id is not None:
            if current.id in seen:
                raise ValueError(f"{body.id}: parent cycle")
            seen.add(current.id)
            current = next(
                candidate for candidate in bodies if candidate.id == current.parent_id
            )
    return CelestialSystem(_text(data, "systemName", path.stem), bodies)
