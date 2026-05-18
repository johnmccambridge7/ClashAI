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
    damage: float = 0.0
    attack_range: float = 0.0
    min_attack_range: float = 0.0
    attack_cooldown: float = TICK_SECONDS
    splash_radius: float = 0.0
    projectile_delay: float = 0.0
    is_defense: bool = False
    blocks_movement: bool = True
    counts_for_score: bool = True
    is_trap: bool = False
    trigger_radius: float = 0.0
    trigger_delay: float = 0.0
    hidden: bool = False
    one_shot: bool = False


BUILDING_SPECS: dict[str, BuildingSpec] = {
    "townhall": BuildingSpec("townhall", hp=1500, size=4),
    "cannon": BuildingSpec(
        "cannon", hp=400, size=3, dps=11.0, attack_range=9.0,
        attack_cooldown=0.8, is_defense=True,
    ),
    "wizard_tower": BuildingSpec(
        "wizard_tower", hp=620, size=3, dps=11.0, attack_range=7.0,
        attack_cooldown=1.3, splash_radius=1.0, is_defense=True,
    ),
    "mortar": BuildingSpec(
        "mortar", hp=400, size=3, dps=5.0, min_attack_range=4.0,
        attack_range=11.0, attack_cooldown=5.0, splash_radius=1.5,
        projectile_delay=1.0, is_defense=True,
    ),
    "bomb": BuildingSpec(
        "bomb", hp=1, size=1, damage=80.0, splash_radius=3.0,
        blocks_movement=False, counts_for_score=False, is_trap=True,
        trigger_radius=1.5, trigger_delay=1.5, hidden=True, one_shot=True,
    ),
    "storage": BuildingSpec("storage", hp=900, size=3),
    "wall": BuildingSpec("wall", hp=300, size=1, counts_for_score=False),
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
    target_priority: str = "scored"
    explodes_on_wall: bool = False
    wall_damage_fraction: float = 0.0
    wall_damage_count: int = 0
    explosion_trigger_range: float = 0.7


@dataclass(frozen=True)
class SpellSpec:
    kind: str
    radius: float
    duration: float
    damage_multiplier: float = 1.0
    freezes_defenses: bool = False


TROOP_SPECS: dict[str, TroopSpec] = {
    "barbarian": TroopSpec("barbarian", hp=65, dps=14.0, attack_range=0.0, speed=2.0),
    "wall_breaker": TroopSpec(
        "wall_breaker",
        hp=20,
        dps=0.0,
        attack_range=0.0,
        speed=2.4,
        target_priority="wall",
        explodes_on_wall=True,
        wall_damage_fraction=0.5,
        wall_damage_count=5,
        explosion_trigger_range=0.7,
    ),
}


SPELL_SPECS: dict[str, SpellSpec] = {
    "rage": SpellSpec("rage", radius=5.0, duration=18.0, damage_multiplier=1.3),
    "freeze": SpellSpec("freeze", radius=5.0, duration=6.0, freezes_defenses=True),
}


@dataclass
class Building:
    id: int
    spec: BuildingSpec
    x: int
    y: int
    hp: float = field(init=False)
    cooldown_remaining: float = 0.0
    revealed: bool = field(init=False)
    triggered: bool = False

    def __post_init__(self) -> None:
        self.hp = float(self.spec.hp)
        self.revealed = not self.spec.hidden

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
