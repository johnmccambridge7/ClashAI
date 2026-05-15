from __future__ import annotations

from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .entities import BUILDING_SPECS, GRID_SIZE, Building


LayoutArchetype = Literal[
    "core",
    "offset_core",
    "exposed_townhall",
    "resource_bait",
    "split",
]
BuildingCounts = tuple[tuple[str, int], ...]
Point = tuple[float, float]
Cell = tuple[int, int]
Orient = Callable[[float, float], Point]


@dataclass(frozen=True)
class LayoutProfile:
    """Compact description of a generated base family."""

    name: str
    building_counts: BuildingCounts = (
        ("townhall", 1),
        ("cannon", 3),
        ("wizard_tower", 1),
        ("mortar", 1),
        ("storage", 3),
        ("bomb", 2),
    )
    wall_count: int = 28
    archetype: LayoutArchetype = "core"
    max_retries: int = 200

    def count(self, kind: str) -> int:
        return sum(count for k, count in self.building_counts if k == kind)

    @property
    def max_buildings(self) -> int:
        return sum(count for _, count in self.building_counts) + self.wall_count


PRESET_LAYOUT_PROFILES: dict[str, LayoutProfile] = {
    "easy": LayoutProfile(
        name="easy",
        building_counts=(
            ("townhall", 1),
            ("cannon", 2),
            ("wizard_tower", 1),
            ("mortar", 1),
            ("storage", 3),
            ("bomb", 2),
        ),
        wall_count=96,
        archetype="exposed_townhall",
    ),
    "medium": LayoutProfile(name="medium", wall_count=176),
    "hard": LayoutProfile(
        name="hard",
        building_counts=(
            ("townhall", 1),
            ("cannon", 4),
            ("wizard_tower", 2),
            ("mortar", 2),
            ("storage", 4),
            ("bomb", 4),
        ),
        wall_count=240,
        archetype="offset_core",
    ),
    "resource_bait": LayoutProfile(
        name="resource_bait",
        building_counts=(
            ("townhall", 1),
            ("cannon", 3),
            ("wizard_tower", 1),
            ("mortar", 1),
            ("storage", 4),
            ("bomb", 3),
        ),
        wall_count=192,
        archetype="resource_bait",
    ),
    "split": LayoutProfile(
        name="split",
        building_counts=(
            ("townhall", 1),
            ("cannon", 4),
            ("wizard_tower", 2),
            ("mortar", 2),
            ("storage", 4),
            ("bomb", 3),
        ),
        wall_count=240,
        archetype="split",
    ),
}


@dataclass(frozen=True)
class _Placement:
    kind: str
    x: int
    y: int


@dataclass(frozen=True)
class _Anchors:
    townhall: list[Point]
    storages: list[Point]
    cannons: list[Point]
    wizard_towers: list[Point]
    mortars: list[Point]
    traps: list[Point]
    other: list[Point]


def preset_layout_profile(name: str) -> LayoutProfile:
    try:
        return PRESET_LAYOUT_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESET_LAYOUT_PROFILES))
        raise ValueError(f"unknown layout profile {name!r}; expected one of: {known}") from exc


def generate_layout(profile: LayoutProfile, rng: np.random.Generator) -> list[Building]:
    """Generate a structured, deterministic-by-seed base layout."""

    _validate_profile(profile)
    last_errors: list[str] = []
    for _ in range(profile.max_retries):
        placements = _generate_placements(profile, rng)
        layout = _to_buildings(placements)
        last_errors = validate_layout(layout, profile) + layout_quality_errors(layout, profile)
        if not last_errors:
            return layout
    joined = "; ".join(last_errors) if last_errors else "no candidate produced"
    raise RuntimeError(f"could not generate valid layout for {profile.name!r}: {joined}")


def validate_layout(layout: list[Building], profile: LayoutProfile | None = None) -> list[str]:
    errors: list[str] = []
    counts = Counter(b.spec.kind for b in layout)

    if counts["townhall"] != 1:
        errors.append(f"expected exactly one townhall, found {counts['townhall']}")

    if profile is not None:
        expected = Counter(dict(profile.building_counts))
        expected["wall"] = profile.wall_count
        for kind, count in expected.items():
            if counts[kind] != count:
                errors.append(f"expected {count} {kind}, found {counts[kind]}")
        for kind in counts:
            if kind not in expected:
                errors.append(f"unexpected building kind {kind!r}")

    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for b in layout:
        if b.spec.kind not in BUILDING_SPECS:
            errors.append(f"unknown building kind {b.spec.kind!r}")
        if b.x < 0 or b.y < 0 or b.x + b.spec.size > GRID_SIZE or b.y + b.spec.size > GRID_SIZE:
            errors.append(f"{b.spec.kind} {b.id} is out of bounds")
            continue
        cells = occupied[b.y : b.y + b.spec.size, b.x : b.x + b.spec.size]
        if cells.any():
            errors.append(f"{b.spec.kind} {b.id} overlaps another object")
        cells[:] = True

    if layout and not _has_legal_deploy_cell(layout):
        errors.append("layout leaves no legal deploy cell")

    return errors


def layout_quality_errors(layout: list[Building], profile: LayoutProfile) -> list[str]:
    """Cheap structural checks that keep generated bases from becoming clutter."""

    errors: list[str] = []
    wall_cells = _wall_cells(layout)
    isolated = [
        cell for cell in wall_cells
        if sum(neighbor in wall_cells for neighbor in _cardinal_neighbors(cell)) == 0
    ]
    if isolated:
        errors.append(f"layout has {len(isolated)} isolated wall cells")

    if profile.wall_count >= 28 and profile.archetype in {"core", "offset_core", "split"}:
        townhall = next((b for b in layout if b.spec.kind == "townhall"), None)
        if townhall is not None:
            if not _has_townhall_enclosure(townhall, wall_cells):
                errors.append("townhall is not enclosed by a wall compartment")
            outside_townhall = {
                cell for cell in wall_cells
                if _distance_to_building(cell, townhall) > 7.0
            }
            if len(outside_townhall) < max(24, profile.wall_count // 3):
                errors.append("too many walls are concentrated around the townhall")

    return errors


def _validate_profile(profile: LayoutProfile) -> None:
    if profile.max_retries <= 0:
        raise ValueError("max_retries must be positive")
    if profile.wall_count < 0:
        raise ValueError("wall_count must be non-negative")
    if profile.count("townhall") != 1:
        raise ValueError("a layout profile must contain exactly one townhall")
    for kind, count in profile.building_counts:
        if count < 0:
            raise ValueError(f"{kind} count must be non-negative")
        if kind == "wall":
            raise ValueError("wall count belongs in wall_count")
        if kind not in BUILDING_SPECS:
            raise ValueError(f"unknown building kind {kind!r}")


def _generate_placements(profile: LayoutProfile, rng: np.random.Generator) -> list[_Placement]:
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placements: list[_Placement] = []
    anchors = _anchor_plan(profile, rng)

    if not _place_many(placements, occupied, "townhall", 1, anchors.townhall, rng):
        return placements
    if not _place_many(placements, occupied, "storage", profile.count("storage"), anchors.storages, rng):
        return placements

    for kind, count in profile.building_counts:
        if kind in {"townhall", "storage"}:
            continue
        if not _place_many(placements, occupied, kind, count, _anchors_for_kind(kind, anchors), rng):
            return placements

    _place_walls(placements, occupied, profile, rng)
    return placements


def _anchor_plan(profile: LayoutProfile, rng: np.random.Generator) -> _Anchors:
    jitter = 2 if profile.archetype in {"core", "exposed_townhall", "split"} else 5
    origin = (
        GRID_SIZE / 2.0 + float(rng.integers(-jitter, jitter + 1)),
        GRID_SIZE / 2.0 + float(rng.integers(-jitter, jitter + 1)),
    )
    orient = _orientation(rng)

    if profile.archetype == "exposed_townhall":
        return _anchors_from_offsets(
            origin,
            orient,
            townhall=[(0.0, -16.0)],
            storages=[(-6.0, 0.0), (6.0, 0.0), (0.0, 6.0), (0.0, -6.0)],
            cannons=[(-12.0, -5.0), (12.0, -5.0), (-10.0, 10.0), (10.0, 10.0)],
            wizard_towers=[(-5.0, 1.0), (5.0, 1.0), (0.0, 5.0), (0.0, -5.0)],
            mortars=[(0.0, 8.0), (0.0, 2.0), (-4.0, 6.0), (4.0, 6.0)],
            traps=[(-4.0, -9.0), (4.0, -9.0), (-9.0, 0.0), (9.0, 0.0), (0.0, 10.0)],
            other=[(-12.0, 0.0), (12.0, 0.0), (0.0, 12.0)],
        )
    if profile.archetype == "resource_bait":
        return _anchors_from_offsets(
            origin,
            orient,
            townhall=[(0.0, 0.0)],
            storages=[(-13.0, 0.0), (13.0, 0.0), (0.0, -13.0), (0.0, 13.0)],
            cannons=[(-11.0, -11.0), (11.0, -11.0), (-11.0, 11.0), (11.0, 11.0), (0.0, -14.0)],
            wizard_towers=[(-8.0, 0.0), (8.0, 0.0), (0.0, -8.0), (0.0, 8.0)],
            mortars=[(0.0, -4.0), (0.0, 4.0), (-4.0, 0.0), (4.0, 0.0)],
            traps=[(-6.0, -6.0), (6.0, -6.0), (-6.0, 6.0), (6.0, 6.0), (0.0, -15.0), (0.0, 15.0)],
            other=[(-15.0, -5.0), (15.0, 5.0), (0.0, 15.0)],
        )
    if profile.archetype == "split":
        return _anchors_from_offsets(
            origin,
            orient,
            townhall=[(8.0, 0.0)],
            storages=[(-9.0, 0.0), (-9.0, -6.0), (-9.0, 6.0), (-15.0, 0.0)],
            cannons=[(1.0, -10.0), (1.0, 10.0), (15.0, -7.0), (15.0, 7.0), (-16.0, 0.0)],
            wizard_towers=[(-5.0, -5.0), (-5.0, 5.0), (10.0, -4.0), (10.0, 4.0)],
            mortars=[(4.0, 0.0), (10.0, 0.0), (-8.0, -2.0), (-8.0, 2.0)],
            traps=[(0.0, 0.0), (4.0, -5.0), (4.0, 5.0), (-13.0, -3.0), (-13.0, 3.0), (14.0, 0.0)],
            other=[(-15.0, -9.0), (-15.0, 9.0), (16.0, 0.0)],
        )

    offset = 4.0 if profile.archetype == "offset_core" else 0.0
    return _anchors_from_offsets(
        origin,
        orient,
        townhall=[(offset, 0.0)],
        storages=[(offset - 9.0, 0.0), (offset + 9.0, 0.0), (offset, -9.0), (offset, 9.0)],
        cannons=[
            (offset - 10.0, -10.0),
            (offset + 10.0, -10.0),
            (offset - 10.0, 10.0),
            (offset + 10.0, 10.0),
            (offset, -14.0),
            (offset, 14.0),
        ],
        wizard_towers=[
            (offset - 5.0, -5.0),
            (offset + 5.0, -5.0),
            (offset - 5.0, 5.0),
            (offset + 5.0, 5.0),
        ],
        mortars=[
            (offset, -4.0),
            (offset, 4.0),
            (offset - 4.0, 0.0),
            (offset + 4.0, 0.0),
        ],
        traps=[
            (offset - 7.0, 0.0),
            (offset + 7.0, 0.0),
            (offset, -7.0),
            (offset, 7.0),
            (offset - 6.0, -6.0),
            (offset + 6.0, 6.0),
        ],
        other=[(offset - 14.0, 0.0), (offset + 14.0, 0.0), (offset, 14.0)],
    )


def _anchors_from_offsets(
    origin: Point,
    orient: Orient,
    *,
    townhall: list[Point],
    storages: list[Point],
    cannons: list[Point],
    wizard_towers: list[Point],
    mortars: list[Point],
    traps: list[Point],
    other: list[Point],
) -> _Anchors:
    return _Anchors(
        townhall=_points(origin, orient, townhall),
        storages=_points(origin, orient, storages),
        cannons=_points(origin, orient, cannons),
        wizard_towers=_points(origin, orient, wizard_towers),
        mortars=_points(origin, orient, mortars),
        traps=_points(origin, orient, traps),
        other=_points(origin, orient, other),
    )


def _anchors_for_kind(kind: str, anchors: _Anchors) -> list[Point]:
    if kind == "cannon":
        return anchors.cannons
    if kind == "wizard_tower":
        return anchors.wizard_towers
    if kind == "mortar":
        return anchors.mortars
    if BUILDING_SPECS[kind].is_trap:
        return anchors.traps
    if BUILDING_SPECS[kind].is_defense:
        return anchors.cannons + anchors.wizard_towers + anchors.mortars
    return anchors.other


def _orientation(rng: np.random.Generator) -> Orient:
    turns = int(rng.integers(0, 4))
    mirror = bool(rng.integers(0, 2))

    def orient(dx: float, dy: float) -> Point:
        if mirror:
            dx = -dx
        for _ in range(turns):
            dx, dy = -dy, dx
        return dx, dy

    return orient


def _points(origin: Point, orient: Orient, offsets: list[Point]) -> list[Point]:
    ox, oy = origin
    return [(ox + dx, oy + dy) for dx, dy in (orient(x, y) for x, y in offsets)]


def _place_many(
    placements: list[_Placement],
    occupied: np.ndarray,
    kind: str,
    count: int,
    anchors: list[Point],
    rng: np.random.Generator,
) -> bool:
    if count == 0:
        return True
    if not anchors:
        return False

    start = int(rng.integers(0, len(anchors)))
    ordered = anchors[start:] + anchors[:start]
    for i in range(count):
        center = ordered[i % len(ordered)]
        if not _place_near(placements, occupied, kind, center, rng):
            return False
    return True


def _place_near(
    placements: list[_Placement],
    occupied: np.ndarray,
    kind: str,
    center: Point,
    rng: np.random.Generator,
) -> bool:
    size = BUILDING_SPECS[kind].size
    base_x = int(round(center[0] - size / 2.0))
    base_y = int(round(center[1] - size / 2.0))

    for radius in range(7):
        offsets = _offset_ring(radius)
        order = rng.permutation(len(offsets))
        for index in order:
            dx, dy = offsets[int(index)]
            x, y = _clamp_top_left(base_x + dx, base_y + dy, size)
            if _place(placements, occupied, kind, x, y):
                return True
    return False


def _place_walls(
    placements: list[_Placement],
    occupied: np.ndarray,
    profile: LayoutProfile,
    rng: np.random.Generator,
) -> None:
    remaining = profile.wall_count
    for cells in _wall_blueprint(profile, placements):
        available = [(x, y) for x, y in cells if _fits(occupied, "wall", x, y)]
        if not available or remaining < len(available):
            continue
        for x, y in available:
            _place(placements, occupied, "wall", x, y)
        remaining -= len(available)
        if remaining == 0:
            return

    _fill_connected_walls(placements, occupied, remaining, rng)


def _wall_blueprint(profile: LayoutProfile, placements: list[_Placement]) -> list[list[Cell]]:
    townhall = next(p for p in placements if p.kind == "townhall")
    storages = [p for p in placements if p.kind == "storage"]
    defenses = [p for p in placements if BUILDING_SPECS[p.kind].is_defense]
    scored = [townhall] + storages

    if profile.archetype == "exposed_townhall":
        return [
            *[_diamond_for_building(p, 3) for p in _by_distance_from_center(storages)],
            _diamond_for_group(storages + defenses, 4),
        ]
    if profile.archetype == "resource_bait":
        return [
            _diamond_for_building(townhall, 5),
            *[_diamond_for_building(p, 4) for p in _by_distance_from_center(storages)],
            _diamond_for_group(scored + defenses, 4),
        ]
    if profile.archetype == "split":
        townhall_group, resource_group = _split_compartments(townhall, storages, defenses)
        return [
            _diamond_for_building(townhall, 5),
            _diamond_for_group(resource_group, 4),
            _diamond_for_group(townhall_group, 4),
            _diamond_for_group(scored + defenses, 5),
        ]
    if profile.archetype == "offset_core":
        return [
            _diamond_for_building(townhall, 5),
            *[_diamond_for_building(p, 3) for p in _by_distance_from_center(storages)],
            _diamond_for_group(scored, 4),
            _diamond_for_group(scored + defenses, 5),
        ]
    return [
        _diamond_for_building(townhall, 5),
        *[_diamond_for_building(p, 3) for p in _by_distance_from_center(storages)],
        _diamond_for_group(scored, 4),
        _diamond_for_group(scored + defenses, 5),
    ]


def _fill_connected_walls(
    placements: list[_Placement],
    occupied: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> None:
    if count <= 0:
        return

    wall_cells = {cell for p in placements if p.kind == "wall" for cell in _placement_cells(p)}
    while count > 0 and wall_cells:
        progress = False
        components = sorted(_wall_components(wall_cells), key=len, reverse=True)
        start = int(rng.integers(0, len(components)))
        for component in components[start:] + components[:start]:
            for x, y in _reinforcement_cells(component):
                if count <= 0:
                    return
                if not _touches_wall((x, y), wall_cells):
                    continue
                if _place(placements, occupied, "wall", x, y):
                    wall_cells.add((x, y))
                    count -= 1
                    progress = True
        if not progress:
            return


def _ring_for_building(placement: _Placement, pad: int) -> list[Cell]:
    size = BUILDING_SPECS[placement.kind].size
    x0 = placement.x - pad
    y0 = placement.y - pad
    x1 = placement.x + size + pad - 1
    y1 = placement.y + size + pad - 1
    return _ring_cells(x0, y0, x1, y1)


def _ring_for_group(placements: list[_Placement], pad: int) -> list[Cell]:
    if not placements:
        return []
    x0 = min(p.x for p in placements) - pad
    y0 = min(p.y for p in placements) - pad
    x1 = max(p.x + BUILDING_SPECS[p.kind].size - 1 for p in placements) + pad
    y1 = max(p.y + BUILDING_SPECS[p.kind].size - 1 for p in placements) + pad
    return _ring_cells(x0, y0, x1, y1)


def _diamond_for_building(placement: _Placement, pad: int) -> list[Cell]:
    cx, cy = _rounded_center([placement])
    radius = max(4, math.ceil(BUILDING_SPECS[placement.kind].size / 2.0) + pad)
    return _stair_diamond_cells(cx, cy, radius)


def _diamond_for_group(placements: list[_Placement], pad: int) -> list[Cell]:
    if not placements:
        return []
    cx, cy = _rounded_center(placements)
    radius = max(
        5,
        max(
            math.ceil(abs(px - cx) + abs(py - cy)) + math.ceil(BUILDING_SPECS[p.kind].size / 2.0)
            for p in placements
            for px, py in [_center(p)]
        ) + pad,
    )
    radius = min(radius, cx - 1, cy - 1, GRID_SIZE - cx - 2, GRID_SIZE - cy - 2)
    if radius < 5:
        return []
    return _stair_diamond_cells(cx, cy, radius)


def _rounded_center(placements: list[_Placement]) -> tuple[int, int]:
    x = sum(_center(p)[0] for p in placements) / len(placements)
    y = sum(_center(p)[1] for p in placements) / len(placements)
    return int(round(x)), int(round(y))


def _stair_diamond_cells(cx: int, cy: int, radius: int) -> list[Cell]:
    cells: list[Cell] = []
    corners = [
        (cx, cy - radius),
        (cx + radius, cy),
        (cx, cy + radius),
        (cx - radius, cy),
    ]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        cells.extend(_stair_path(start, end))
    return _dedupe_cells(cells)


def _stair_path(start: Cell, end: Cell) -> list[Cell]:
    x, y = start
    ex, ey = end
    dx = 1 if ex > x else -1
    dy = 1 if ey > y else -1
    cells = [(x, y)]
    while x != ex or y != ey:
        if x != ex:
            x += dx
            cells.append((x, y))
        if y != ey:
            y += dy
            cells.append((x, y))
    return cells


def _ring_cells(x0: int, y0: int, x1: int, y1: int) -> list[Cell]:
    if x0 < 0 or y0 < 0 or x1 >= GRID_SIZE or y1 >= GRID_SIZE:
        return []
    top = [(x, y0) for x in range(x0, x1 + 1)]
    right = [(x1, y) for y in range(y0 + 1, y1 + 1)]
    bottom = [(x, y1) for x in range(x1 - 1, x0 - 1, -1)]
    left = [(x0, y) for y in range(y1 - 1, y0, -1)]
    return top + right + bottom + left


def _has_townhall_enclosure(building: Building, wall_cells: set[Cell]) -> bool:
    cx, cy = building.center
    north = any(y < building.y and abs(x + 0.5 - cx) <= 6.0 for x, y in wall_cells)
    south = any(y > building.y + building.spec.size - 1 and abs(x + 0.5 - cx) <= 6.0 for x, y in wall_cells)
    west = any(x < building.x and abs(y + 0.5 - cy) <= 6.0 for x, y in wall_cells)
    east = any(x > building.x + building.spec.size - 1 and abs(y + 0.5 - cy) <= 6.0 for x, y in wall_cells)
    return north and south and west and east


def _distance_to_building(cell: Cell, building: Building) -> float:
    x, y = cell
    return building.distance_to(x + 0.5, y + 0.5)


def _from_building(building: Building) -> _Placement:
    return _Placement(kind=building.spec.kind, x=building.x, y=building.y)


def _split_compartments(
    townhall: _Placement,
    storages: list[_Placement],
    defenses: list[_Placement],
) -> tuple[list[_Placement], list[_Placement]]:
    tx, ty = _center(townhall)
    sx = sum(_center(p)[0] for p in storages) / max(1, len(storages))
    sy = sum(_center(p)[1] for p in storages) / max(1, len(storages))
    vx = sx - tx
    vy = sy - ty
    if abs(vx) + abs(vy) <= 1e-9:
        vx = 1.0

    townhall_group: list[_Placement] = [townhall]
    resource_group: list[_Placement] = list(storages)
    for p in defenses:
        px, py = _center(p)
        if (px - tx) * vx + (py - ty) * vy >= 0:
            resource_group.append(p)
        else:
            townhall_group.append(p)
    return townhall_group, resource_group


def _by_distance_from_center(placements: list[_Placement]) -> list[_Placement]:
    mid = GRID_SIZE / 2.0
    return sorted(placements, key=lambda p: math.hypot(_center(p)[0] - mid, _center(p)[1] - mid))


def _cells_available(occupied: np.ndarray, cells: list[Cell]) -> bool:
    return bool(cells) and all(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and not occupied[y, x] for x, y in cells)


def _place(
    placements: list[_Placement],
    occupied: np.ndarray,
    kind: str,
    x: int,
    y: int,
) -> bool:
    if not _fits(occupied, kind, x, y):
        return False
    size = BUILDING_SPECS[kind].size
    occupied[y : y + size, x : x + size] = True
    placements.append(_Placement(kind=kind, x=x, y=y))
    return True


def _fits(occupied: np.ndarray, kind: str, x: int, y: int) -> bool:
    size = BUILDING_SPECS[kind].size
    if x < 0 or y < 0 or x + size > GRID_SIZE or y + size > GRID_SIZE:
        return False
    return not occupied[y : y + size, x : x + size].any()


def _center(placement: _Placement) -> Point:
    size = BUILDING_SPECS[placement.kind].size
    return placement.x + size / 2.0, placement.y + size / 2.0


def _clamp_top_left(x: int, y: int, size: int) -> tuple[int, int]:
    return max(1, min(x, GRID_SIZE - size - 1)), max(1, min(y, GRID_SIZE - size - 1))


def _offset_ring(radius: int) -> list[tuple[int, int]]:
    if radius == 0:
        return [(0, 0)]
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if max(abs(dx), abs(dy)) == radius
    ]


def _to_buildings(placements: list[_Placement]) -> list[Building]:
    order = {
        "townhall": 0,
        "cannon": 10,
        "wizard_tower": 11,
        "mortar": 12,
        "storage": 20,
        "bomb": 30,
        "wall": 100,
    }
    sorted_placements = sorted(
        placements,
        key=lambda p: (order.get(p.kind, 50), p.kind, p.y, p.x),
    )
    return [
        Building(id=i, spec=BUILDING_SPECS[p.kind], x=p.x, y=p.y)
        for i, p in enumerate(sorted_placements)
    ]


def _dedupe_cells(cells: list[Cell]) -> list[Cell]:
    seen: set[Cell] = set()
    out: list[Cell] = []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        out.append(cell)
    return out


def _wall_components(wall_cells: set[Cell]) -> list[set[Cell]]:
    remaining = set(wall_cells)
    components: list[set[Cell]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            for neighbor in _cardinal_neighbors(cell):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    return components


def _reinforcement_cells(component: set[Cell]) -> list[Cell]:
    x0 = min(x for x, _ in component)
    x1 = max(x for x, _ in component)
    y0 = min(y for _, y in component)
    y1 = max(y for _, y in component)
    top = [(x, y0 - 1) for x in _centered_range(x0, x1)]
    bottom = [(x, y1 + 1) for x in _centered_range(x0, x1)]
    left = [(x0 - 1, y) for y in _centered_range(y0, y1)]
    right = [(x1 + 1, y) for y in _centered_range(y0, y1)]
    return [
        cell for cell in top + bottom + left + right
        if 0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE
    ]


def _centered_range(lo: int, hi: int) -> list[int]:
    center = (lo + hi) // 2
    values = [center]
    for delta in range(1, hi - lo + 1):
        left = center - delta
        right = center + delta
        if lo <= right <= hi:
            values.append(right)
        if lo <= left <= hi:
            values.append(left)
    return values


def _touches_wall(cell: Cell, wall_cells: set[Cell]) -> bool:
    return any(neighbor in wall_cells for neighbor in _cardinal_neighbors(cell))


def _placement_cells(placement: _Placement) -> list[Cell]:
    size = BUILDING_SPECS[placement.kind].size
    return [
        (x, y)
        for y in range(placement.y, placement.y + size)
        for x in range(placement.x, placement.x + size)
    ]


def _wall_cells(layout: list[Building]) -> set[Cell]:
    cells: set[Cell] = set()
    for b in layout:
        if b.spec.kind == "wall":
            cells.add((b.x, b.y))
    return cells


def _cardinal_neighbors(cell: Cell) -> list[Cell]:
    x, y = cell
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _has_legal_deploy_cell(layout: list[Building]) -> bool:
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            px = x + 0.5
            py = y + 0.5
            if any(b.distance_to(px, py) <= 1e-9 for b in layout):
                continue
            if any(
                b.spec.is_defense and math.hypot(px - b.center[0], py - b.center[1]) <= b.spec.attack_range
                for b in layout
            ):
                continue
            return True
    return False
