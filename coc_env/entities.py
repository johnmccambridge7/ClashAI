from __future__ import annotations
from dataclasses import dataclass, field

GRID_SIZE: int = 44
TICK_SECONDS: float = 0.25
MAX_TICKS: int = 720
DECISION_INTERVAL: int = 10


@dataclass(frozen=True)
class BuildingSpec:
    kind: str
    hp: int
    size: int
    dps: float = 0.0
    attack_range: float = 0.0
    is_defense: bool = False
    blocks_movement: bool = False
    counts_for_score: bool = True


BUILDING_SPECS: dict[str, BuildingSpec] = {
    "townhall": BuildingSpec("townhall", hp=1500, size=4),
    "cannon":   BuildingSpec("cannon",   hp=400,  size=3, dps=11.0, attack_range=9.0, is_defense=True),
    "storage":  BuildingSpec("storage",  hp=900,  size=3),
    "wall":     BuildingSpec("wall",     hp=300,  size=1, blocks_movement=True, counts_for_score=False),
}


@dataclass(frozen=True)
class TroopSpec:
    """A troop's physical stats.

    speed is in *tiles per second* (consistent with dps being damage-per-second).
    Movement code multiplies by TICK_SECONDS to get tiles-per-tick.
    """
    kind: str
    hp: int
    dps: float
    attack_range: float
    speed: float


TROOP_SPECS: dict[str, TroopSpec] = {
    "barbarian": TroopSpec("barbarian", hp=65, dps=14.0, attack_range=0.0, speed=2.0),
}


@dataclass
class Building:
    id: int
    spec: BuildingSpec
    x: int
    y: int
    hp: float = field(init=False)

    def __post_init__(self) -> None:
        self.hp = float(self.spec.hp)

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.spec.size / 2.0, self.y + self.spec.size / 2.0)

    def distance_to(self, px: float, py: float) -> float:
        dx = max(self.x - px, 0.0, px - (self.x + self.spec.size))
        dy = max(self.y - py, 0.0, py - (self.y + self.spec.size))
        return (dx * dx + dy * dy) ** 0.5

    def nearest_point(self, px: float, py: float) -> tuple[float, float]:
        nx = min(max(px, float(self.x)), float(self.x + self.spec.size))
        ny = min(max(py, float(self.y)), float(self.y + self.spec.size))
        return nx, ny


@dataclass
class Troop:
    id: int
    spec: TroopSpec
    x: float
    y: float
    hp: float = field(init=False)
    target_id: int | None = None

    def __post_init__(self) -> None:
        self.hp = float(self.spec.hp)

    @property
    def alive(self) -> bool:
        return self.hp > 0
