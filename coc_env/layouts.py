from __future__ import annotations

from .entities import Building, BUILDING_SPECS


Placement = tuple[str, int, int]


def _wall_ring(x0: int, y0: int, x1: int, y1: int) -> list[tuple[str, int, int]]:
    top_bottom = [("wall", x, y) for x in range(x0, x1 + 1) for y in (y0, y1)]
    sides = [("wall", x, y) for x in (x0, x1) for y in range(y0 + 1, y1)]
    return top_bottom + sides


def _build(placements: list[Placement]) -> list[Building]:
    return [
        Building(id=i, spec=BUILDING_SPECS[kind], x=x, y=y)
        for i, (kind, x, y) in enumerate(placements)
    ]


def default_layout(seed: int | None = None) -> list[Building]:
    placements: list[Placement] = [
        ("townhall", 20, 20),
        ("cannon",   11, 11),
        ("cannon",   28, 11),
        ("cannon",   19, 29),
        ("wizard_tower", 11, 29),
        ("mortar",   28, 29),
        ("storage",  20, 11),
        ("storage",  11, 22),
        ("storage",  28, 22),
        ("bomb",     19, 19),
        ("bomb",     24, 24),
    ] + _wall_ring(18, 18, 25, 25)
    return _build(placements)
