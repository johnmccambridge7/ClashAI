from __future__ import annotations

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


@dataclass(frozen=True)
class LayoutProfile:
    """Compact description of a generated base family."""

    name: str
    building_counts: BuildingCounts = (
        ("townhall", 1),
        ("cannon", 3),
        ("storage", 3),
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
        building_counts=(("townhall", 1), ("cannon", 2), ("storage", 3)),
        wall_count=16,
        archetype="exposed_townhall",
    ),
    "medium": LayoutProfile(name="medium"),
    "hard": LayoutProfile(
        name="hard",
        building_counts=(("townhall", 1), ("cannon", 4), ("storage", 4)),
        wall_count=52,
        archetype="offset_core",
    ),
    "resource_bait": LayoutProfile(
        name="resource_bait",
        building_counts=(("townhall", 1), ("cannon", 3), ("storage", 4)),
        wall_count=32,
        archetype="resource_bait",
    ),
    "split": LayoutProfile(
        name="split",
        building_counts=(("townhall", 1), ("cannon", 4), ("storage", 4)),
        wall_count=44,
        archetype="split",
    ),
}


@dataclass(frozen=True)
class _Placement:
    kind: str
    x: int
    y: int


def preset_layout_profile(name: str) -> LayoutProfile:
    try:
        return PRESET_LAYOUT_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESET_LAYOUT_PROFILES))
        raise ValueError(f"unknown layout profile {name!r}; expected one of: {known}") from exc


def generate_layout(profile: LayoutProfile, rng: np.random.Generator) -> list[Building]:
    """Generate a valid, deterministic-by-seed layout for the given profile."""

    _validate_profile(profile)
    last_errors: list[str] = []
    for _ in range(profile.max_retries):
        placements = _generate_placements(profile, rng)
        layout = _to_buildings(placements)
        last_errors = validate_layout(layout, profile)
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

    tx, ty = _townhall_position(profile, rng)
    if not _place(placements, occupied, "townhall", tx, ty):
        return placements

    townhall_center = _center(placements[0])
    storage_centers = _storage_centers(profile, townhall_center)
    _place_many(placements, occupied, "storage", profile.count("storage"), storage_centers, rng)

    protected = [townhall_center] + [_center(p) for p in placements if p.kind == "storage"]
    for kind, count in profile.building_counts:
        if kind in {"townhall", "storage"}:
            continue
        if BUILDING_SPECS[kind].is_defense:
            _place_many(placements, occupied, kind, count, protected, rng, min_radius=5.0, max_radius=16.0)
        else:
            _place_many(placements, occupied, kind, count, protected, rng)

    _place_walls(placements, occupied, profile, rng)
    return placements


def _townhall_position(profile: LayoutProfile, rng: np.random.Generator) -> tuple[int, int]:
    size = BUILDING_SPECS["townhall"].size
    core = GRID_SIZE // 2 - size // 2

    if profile.archetype == "exposed_townhall":
        side = int(rng.integers(0, 4))
        along = int(rng.integers(8, GRID_SIZE - size - 8))
        edge = 4
        if side == 0:
            return along, edge
        if side == 1:
            return GRID_SIZE - size - edge, along
        if side == 2:
            return along, GRID_SIZE - size - edge
        return edge, along

    jitter = 3 if profile.archetype == "core" else 7
    x = core + int(rng.integers(-jitter, jitter + 1))
    y = core + int(rng.integers(-jitter, jitter + 1))
    return _clamp_top_left(x, y, size)


def _storage_centers(
    profile: LayoutProfile,
    townhall_center: tuple[float, float],
) -> list[tuple[float, float]]:
    cx, cy = townhall_center
    mid = GRID_SIZE / 2.0
    if profile.archetype == "exposed_townhall":
        return [(mid, mid)]
    if profile.archetype == "resource_bait":
        return [(cx - 12.0, cy), (cx + 12.0, cy), (cx, cy - 12.0), (cx, cy + 12.0)]
    if profile.archetype == "split":
        side = -1.0 if cx > mid else 1.0
        return [(cx + side * 12.0, cy), (cx + side * 12.0, cy - 6.0), (cx + side * 12.0, cy + 6.0)]
    return [(cx - 9.0, cy), (cx + 9.0, cy), (cx, cy - 9.0), (cx, cy + 9.0)]


def _place_many(
    placements: list[_Placement],
    occupied: np.ndarray,
    kind: str,
    count: int,
    centers: list[tuple[float, float]],
    rng: np.random.Generator,
    *,
    min_radius: float = 0.0,
    max_radius: float = 18.0,
) -> None:
    for _ in range(count):
        candidate = _best_position(placements, occupied, kind, centers, rng, min_radius, max_radius)
        if candidate is None:
            return
        _place(placements, occupied, kind, candidate[0], candidate[1])


def _best_position(
    placements: list[_Placement],
    occupied: np.ndarray,
    kind: str,
    centers: list[tuple[float, float]],
    rng: np.random.Generator,
    min_radius: float,
    max_radius: float,
) -> tuple[int, int] | None:
    spec = BUILDING_SPECS[kind]
    desired = (min_radius + max_radius) / 2.0
    same_kind = [_center(p) for p in placements if p.kind == kind]
    ranked: list[tuple[float, int, int]] = []

    for y in range(1, GRID_SIZE - spec.size):
        for x in range(1, GRID_SIZE - spec.size):
            if not _fits(occupied, kind, x, y):
                continue
            cx = x + spec.size / 2.0
            cy = y + spec.size / 2.0
            d = min(math.hypot(cx - ox, cy - oy) for ox, oy in centers)
            if not (min_radius <= d <= max_radius):
                continue
            spread = min((math.hypot(cx - ox, cy - oy) for ox, oy in same_kind), default=12.0)
            score = abs(d - desired) - 0.18 * min(spread, 12.0) + float(rng.random()) * 0.35
            ranked.append((score, x, y))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    top = ranked[: min(8, len(ranked))]
    _, x, y = top[int(rng.integers(0, len(top)))]
    return x, y


def _place_walls(
    placements: list[_Placement],
    occupied: np.ndarray,
    profile: LayoutProfile,
    rng: np.random.Generator,
) -> None:
    remaining = profile.wall_count
    if remaining == 0:
        return

    protected = _protected_rects(placements, profile)
    wall_cells: list[tuple[int, int]] = []
    for rect in protected:
        for pad in (2, 5, 8):
            wall_cells.extend(_ring_cells(rect, pad))

    wall_cells = _dedupe_cells(wall_cells)
    if wall_cells:
        start = int(rng.integers(0, len(wall_cells)))
        wall_cells = wall_cells[start:] + wall_cells[:start]

    for x, y in wall_cells:
        if remaining <= 0:
            return
        if _place(placements, occupied, "wall", x, y):
            remaining -= 1

    for y in range(2, GRID_SIZE - 2):
        for x in range(2, GRID_SIZE - 2):
            if remaining <= 0:
                return
            if _near_existing_building(placements, x, y) and _place(placements, occupied, "wall", x, y):
                remaining -= 1


def _protected_rects(
    placements: list[_Placement],
    profile: LayoutProfile,
) -> list[tuple[int, int, int, int]]:
    townhall = next(p for p in placements if p.kind == "townhall")
    storages = [p for p in placements if p.kind == "storage"]
    if profile.archetype == "exposed_townhall" and storages:
        return [_bounding_rect(storages)]
    if profile.archetype == "split" and storages:
        return [_placement_rect(townhall), _bounding_rect(storages)]
    return [_placement_rect(townhall)]


def _ring_cells(rect: tuple[int, int, int, int], pad: int) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = rect
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    if x0 < 0 or y0 < 0 or x1 >= GRID_SIZE or y1 >= GRID_SIZE:
        return []
    top = [(x, y0) for x in range(x0, x1 + 1)]
    right = [(x1, y) for y in range(y0 + 1, y1 + 1)]
    bottom = [(x, y1) for x in range(x1 - 1, x0 - 1, -1)]
    left = [(x0, y) for y in range(y1 - 1, y0, -1)]
    return top + right + bottom + left


def _bounding_rect(placements: list[_Placement]) -> tuple[int, int, int, int]:
    x0 = min(p.x for p in placements)
    y0 = min(p.y for p in placements)
    x1 = max(p.x + BUILDING_SPECS[p.kind].size - 1 for p in placements)
    y1 = max(p.y + BUILDING_SPECS[p.kind].size - 1 for p in placements)
    return x0, y0, x1, y1


def _placement_rect(placement: _Placement) -> tuple[int, int, int, int]:
    size = BUILDING_SPECS[placement.kind].size
    return placement.x, placement.y, placement.x + size - 1, placement.y + size - 1


def _dedupe_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        out.append(cell)
    return out


def _near_existing_building(placements: list[_Placement], x: int, y: int) -> bool:
    return any(math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= 10.0 for cx, cy in map(_center, placements))


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


def _center(placement: _Placement) -> tuple[float, float]:
    size = BUILDING_SPECS[placement.kind].size
    return placement.x + size / 2.0, placement.y + size / 2.0


def _clamp_top_left(x: int, y: int, size: int) -> tuple[int, int]:
    return max(1, min(x, GRID_SIZE - size - 1)), max(1, min(y, GRID_SIZE - size - 1))


def _to_buildings(placements: list[_Placement]) -> list[Building]:
    order = {"townhall": 0, "cannon": 1, "storage": 2, "wall": 100}
    sorted_placements = sorted(
        placements,
        key=lambda p: (order.get(p.kind, 50), p.kind, p.y, p.x),
    )
    return [
        Building(id=i, spec=BUILDING_SPECS[p.kind], x=p.x, y=p.y)
        for i, p in enumerate(sorted_placements)
    ]


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
