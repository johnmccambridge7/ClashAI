from __future__ import annotations
from dataclasses import dataclass
import heapq
import math

from .entities import (
    Building,
    SpellSpec,
    Troop,
    SPELL_SPECS,
    TROOP_SPECS,
    GRID_SIZE,
    TICK_SECONDS,
    MAX_TICKS,
)


Engagement = dict[str, float | str | int]
FlowField = list[list[float]]
WALL_STEP_COST: float = 20.0


@dataclass
class PendingImpact:
    from_x: float
    from_y: float
    x: float
    y: float
    damage: float
    radius: float
    delay: float
    source_id: int
    source_kind: str
    target_id: int
    impact_tick: int


@dataclass
class ActiveSpell:
    kind: str
    spec: SpellSpec
    x: float
    y: float
    remaining: float


class Simulator:
    """Deterministic battle simulator.

    Slot stability invariant: once a building or troop is created it stays in
    `self.buildings` / `self.troops` for the lifetime of the episode. Death is
    indicated by `.alive == False` (i.e. `hp <= 0`). Targeting and rendering
    code filters by `.alive`; observation slots map to list indices.

    Building slot i is the i-th building in the layout (stable).
    Troop slot i is the troop with `id == i` (ids are assigned monotonically
    starting at 0, so slot index == troop id).
    """

    def __init__(
        self,
        layout: list[Building],
        army_size: int,
        seed: int = 0,
        army_composition: dict[str, int] | None = None,
        spell_composition: dict[str, int] | None = None,
        max_ticks: int = MAX_TICKS,
    ):
        if army_size < 0:
            raise ValueError("army_size must be non-negative")
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        composition = (
            {"barbarian": int(army_size)}
            if army_composition is None
            else {kind: int(count) for kind, count in army_composition.items()}
        )
        for kind, count in composition.items():
            if kind not in TROOP_SPECS:
                raise ValueError(f"unknown troop kind {kind!r}")
            if count < 0:
                raise ValueError(f"{kind} count must be non-negative")
        spells = {} if spell_composition is None else {kind: int(count) for kind, count in spell_composition.items()}
        for kind, count in spells.items():
            if kind not in SPELL_SPECS:
                raise ValueError(f"unknown spell kind {kind!r}")
            if count < 0:
                raise ValueError(f"{kind} count must be non-negative")
        self.buildings: list[Building] = [
            Building(id=b.id, spec=b.spec, x=b.x, y=b.y) for b in layout
        ]
        self.army_remaining_by_kind: dict[str, int] = composition
        self.spells_remaining_by_kind: dict[str, int] = spells
        self.army_size: int = sum(composition.values())
        self.max_ticks: int = int(max_ticks)
        self.troops: list[Troop] = []
        self.active_spells: list[ActiveSpell] = []
        self.deployment_cells: list[tuple[int, int, str]] = []
        self.troop_deploy_origins: dict[int, tuple[int, int, int]] = {}
        self.tick_count: int = 0
        self.next_troop_id: int = 0
        self.pending_impacts: list[PendingImpact] = []
        self.visual_events: list[Engagement] = []
        self.terrain_version: int = 0
        self._flow_cache: dict[tuple[int, int], FlowField] = {}
        self._cell_index: dict[tuple[int, int], Building] = {}
        self._cell_index_version: int = -1
        self.original_total_hp: float = float(sum(
            b.spec.hp for b in self.buildings if b.spec.counts_for_score
        ))
        self.original_defense_hp: float = float(sum(
            b.spec.hp for b in self.buildings if b.spec.is_defense
        ))
        self.original_townhall_hp: float = float(sum(
            b.spec.hp for b in self.buildings if b.spec.kind == "townhall"
        ))
        self.original_army_hp: float = float(sum(
            TROOP_SPECS[kind].hp * count for kind, count in composition.items()
        ))
        self.defense_damage: float = 0.0
        self.townhall_damage: float = 0.0
        self.defenses_destroyed: int = 0
        self.rage_bonus_scored_damage: float = 0.0
        self.rage_bonus_defense_damage: float = 0.0
        self.rage_bonus_townhall_damage: float = 0.0
        self.frozen_defense_ticks: int = 0
        self.frozen_threat_ticks: int = 0
        self.frozen_threat_damage_prevented: float = 0.0
        self.troop_damage_taken: float = 0.0
        self.splash_damage_taken: float = 0.0
        self.multi_hit_splash_damage_taken: float = 0.0
        self.splash_cluster_risk: float = 0.0
        self.multi_hit_splash_events: int = 0
        self.scored_damage_by_origin_quadrant: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.wall_breaker_explosions: int = 0
        self.wall_breaker_damaged_wall_segments: int = 0
        self.wall_breaker_destroyed_wall_segments: int = 0
        self.wall_breaker_exploded_ids: set[int] = set()
        self.spell_casts_by_kind: dict[str, int] = {kind: 0 for kind in SPELL_SPECS}
        self.useful_spell_casts_by_kind: dict[str, int] = {kind: 0 for kind in SPELL_SPECS}
        self.wasted_spell_casts_by_kind: dict[str, int] = {kind: 0 for kind in SPELL_SPECS}

    @property
    def army_remaining(self) -> int:
        return sum(self.army_remaining_by_kind.values())

    @property
    def spells_remaining(self) -> int:
        return sum(self.spells_remaining_by_kind.values())

    @property
    def spell_casts(self) -> int:
        return sum(self.spell_casts_by_kind.values())

    @property
    def useful_spell_casts(self) -> int:
        return sum(self.useful_spell_casts_by_kind.values())

    @property
    def wasted_spell_casts(self) -> int:
        return sum(self.wasted_spell_casts_by_kind.values())

    @property
    def wall_breakers_dead_without_explosion(self) -> int:
        return sum(
            1
            for troop in self.troops
            if troop.spec.kind == "wall_breaker"
            and not troop.alive
            and troop.id not in self.wall_breaker_exploded_ids
        )

    # ── view helpers ─────────────────────────────────────────────────────
    @property
    def alive_buildings(self) -> list[Building]:
        return [b for b in self.buildings if b.alive]

    @property
    def visible_buildings(self) -> list[Building]:
        return [b for b in self.alive_buildings if b.revealed]

    @property
    def active_troops(self) -> list[Troop]:
        return [t for t in self.troops if t.alive]

    @property
    def scored_buildings(self) -> list[Building]:
        return [b for b in self.buildings if b.spec.counts_for_score]

    # ── deploy ───────────────────────────────────────────────────────────
    def deploy(self, x: float, y: float, troop_kind: str = "barbarian") -> bool:
        if troop_kind not in TROOP_SPECS:
            raise ValueError(f"unknown troop kind {troop_kind!r}")
        if self.army_remaining_by_kind.get(troop_kind, 0) <= 0:
            return False
        if not (0.0 <= x < GRID_SIZE and 0.0 <= y < GRID_SIZE):
            return False
        if self._blocking_building(x, y) is not None:
            return False
        self.army_remaining_by_kind[troop_kind] -= 1
        troop_id = self.next_troop_id
        cell_x, cell_y = int(x), int(y)
        quadrant = int(cell_x >= GRID_SIZE / 2) + 2 * int(cell_y >= GRID_SIZE / 2)
        self.deployment_cells.append((cell_x, cell_y, troop_kind))
        self.troop_deploy_origins[troop_id] = (cell_x, cell_y, quadrant)
        self.troops.append(
            Troop(id=troop_id, spec=TROOP_SPECS[troop_kind],
                  x=float(x), y=float(y))
        )
        self.next_troop_id += 1
        return True

    def cast_spell(self, x: float, y: float, spell_kind: str) -> bool:
        if spell_kind not in SPELL_SPECS:
            raise ValueError(f"unknown spell kind {spell_kind!r}")
        if self.spells_remaining_by_kind.get(spell_kind, 0) <= 0:
            return False
        if not (0.0 <= x < GRID_SIZE and 0.0 <= y < GRID_SIZE):
            return False
        spec = SPELL_SPECS[spell_kind]
        useful = self._spell_cast_has_target(x, y, spell_kind)
        self.spell_casts_by_kind[spell_kind] = self.spell_casts_by_kind.get(spell_kind, 0) + 1
        if useful:
            self.useful_spell_casts_by_kind[spell_kind] = self.useful_spell_casts_by_kind.get(spell_kind, 0) + 1
        else:
            self.wasted_spell_casts_by_kind[spell_kind] = self.wasted_spell_casts_by_kind.get(spell_kind, 0) + 1
        self.spells_remaining_by_kind[spell_kind] -= 1
        self.active_spells.append(ActiveSpell(
            kind=spell_kind,
            spec=spec,
            x=float(x),
            y=float(y),
            remaining=spec.duration,
        ))
        self._emit_visual_event({
            "kind": "spell_cast",
            "source_kind": spell_kind,
            "attacker_id": -1,
            "target_id": -1,
            "from_x": x,
            "from_y": y,
            "to_x": x,
            "to_y": y,
            "radius": spec.radius,
            "duration": spec.duration,
            "useful": int(useful),
        })
        return True

    # ── single tick ──────────────────────────────────────────────────────
    def tick(self) -> None:
        if self.is_done:
            return
        self.visual_events = []
        self.tick_count += 1
        self._drop_expired_spells()

        self._advance_impacts()
        self._update_traps()

        for t in self.troops:
            if not t.alive:
                continue
            if self._target_for(t) is None:
                t.target_id = self._pick_target_id(t)
                if t.target_id is None and t.spec.target_priority == "wall":
                    t.hp = 0.0

        for t in self.troops:
            if not t.alive:
                continue
            target = self._target_for(t)
            if target is None:
                continue
            d = target.distance_to(t.x, t.y)
            if self._should_wall_break(t, target, d):
                self._explode_wall_breaker(t, target)
                continue
            if d <= t.spec.attack_range + 0.7:
                base_damage = t.spec.dps * TICK_SECONDS
                multiplier = self._troop_damage_multiplier(t)
                self._damage_building(
                    target,
                    base_damage * multiplier,
                    rage_bonus=max(0.0, base_damage * (multiplier - 1.0)),
                    attacker=t,
                )
            else:
                self._move_toward(t, target)

        self.splash_cluster_risk += self._current_splash_cluster_risk()
        self._update_defenses()
        self._advance_spell_durations()

    # ── targeting / motion ───────────────────────────────────────────────
    def _pick_target_id(self, troop: Troop) -> int | None:
        if troop.spec.target_priority == "wall":
            return self._pick_wall_target_id(troop)

        candidates: list[tuple[float, Building]] = []
        for b in self.buildings:
            if b.alive and b.spec.counts_for_score:
                candidates.append((b.distance_to(troop.x, troop.y), b))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p[0])
        top3 = [b for _, b in candidates[:3]]
        sx = max(0, min(GRID_SIZE - 1, int(troop.x)))
        sy = max(0, min(GRID_SIZE - 1, int(troop.y)))
        best_id: int | None = None
        best_cost = math.inf
        for b in top3:
            cost = self._flow_to_target(b)[sy][sx]
            if cost < best_cost:
                best_cost = cost
                best_id = b.id
        return best_id

    def _pick_wall_target_id(self, troop: Troop) -> int | None:
        best_id: int | None = None
        best_d = math.inf
        for b in self.buildings:
            if not b.alive or b.spec.kind != "wall":
                continue
            d = b.distance_to(troop.x, troop.y)
            if d < best_d:
                best_d = d
                best_id = b.id
        return best_id

    def _target_for(self, troop: Troop) -> Building | None:
        if troop.target_id is None:
            return None
        return self._building_by_id(troop.target_id)

    def _building_by_id(self, building_id: int) -> Building | None:
        for b in self.buildings:
            if b.id == building_id and b.alive:
                return b
        return None

    def predicted_target_for(self, troop: Troop) -> Building | None:
        target = self._target_for(troop)
        if target is not None:
            return target
        target_id = self._pick_target_id(troop)
        if target_id is None:
            return None
        return self._building_by_id(target_id)

    def predicted_path_for(self, troop: Troop, max_steps: int = 72) -> list[tuple[float, float]]:
        if not troop.alive:
            return []
        target = self.predicted_target_for(troop)
        if target is None:
            return []

        x, y = troop.x, troop.y
        path: list[tuple[float, float]] = [(x, y)]
        seen: set[tuple[float, float]] = set()
        for _ in range(max(0, max_steps)):
            if target.distance_to(x, y) <= troop.spec.attack_range + 0.7:
                tx, ty = target.nearest_point(x, y)
                if math.hypot(tx - x, ty - y) > 1e-9:
                    path.append((tx, ty))
                break

            target_cell = self._next_path_cell_from(x, y, target)
            if target_cell is None:
                tx, ty = target.nearest_point(x, y)
            else:
                tx, ty = self._path_cell_entry_point_from(x, y, target_cell)
            if math.hypot(tx - x, ty - y) <= 1e-9:
                break

            x, y = tx, ty
            point_key = (round(x, 3), round(y, 3))
            if point_key in seen:
                break
            seen.add(point_key)
            path.append((x, y))

        return path

    def troop_has_attack_opportunity(self, troop: Troop) -> bool:
        if not troop.alive:
            return False
        target = self.predicted_target_for(troop)
        if target is None:
            return False
        distance = target.distance_to(troop.x, troop.y)
        if troop.spec.explodes_on_wall and target.spec.kind == "wall":
            return distance <= troop.spec.explosion_trigger_range
        return distance <= troop.spec.attack_range + 0.7

    def _move_toward(self, troop: Troop, target: Building) -> None:
        target_cell = self._next_path_cell(troop, target)
        if target_cell is None:
            tx, ty = target.nearest_point(troop.x, troop.y)
        else:
            cell_blocker = self._blocking_building_at_cell(target_cell[0], target_cell[1], target.id)
            if cell_blocker is not None and cell_blocker.spec.kind == "wall":
                if troop.spec.explodes_on_wall:
                    self._explode_wall_breaker(troop, cell_blocker)
                    return
                troop.target_id = cell_blocker.id
                base_damage = troop.spec.dps * TICK_SECONDS
                multiplier = self._troop_damage_multiplier(troop)
                self._damage_building(
                    cell_blocker,
                    base_damage * multiplier,
                    rage_bonus=max(0.0, base_damage * (multiplier - 1.0)),
                    attacker=troop,
                )
                return
            if cell_blocker is not None:
                return
            tx, ty = self._path_cell_entry_point(troop, target_cell)
        dx = tx - troop.x
        dy = ty - troop.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            return
        # speed is tiles/sec; convert to tiles/tick
        step = min(troop.spec.speed * TICK_SECONDS, dist)
        nx = troop.x + step * dx / dist
        ny = troop.y + step * dy / dist
        blocker = self._blocking_building(nx, ny, target.id)
        if blocker is not None and blocker.spec.kind == "wall":
            if troop.spec.explodes_on_wall:
                self._explode_wall_breaker(troop, blocker)
                return
            troop.target_id = blocker.id
            base_damage = troop.spec.dps * TICK_SECONDS
            multiplier = self._troop_damage_multiplier(troop)
            self._damage_building(
                blocker,
                base_damage * multiplier,
                rage_bonus=max(0.0, base_damage * (multiplier - 1.0)),
                attacker=troop,
            )
            return
        if blocker is not None:
            return
        troop.x = nx
        troop.y = ny

    def _path_cell_entry_point(self, troop: Troop, cell: tuple[int, int]) -> tuple[float, float]:
        return self._path_cell_entry_point_from(troop.x, troop.y, cell)

    def _path_cell_entry_point_from(self, x: float, y: float, cell: tuple[int, int]) -> tuple[float, float]:
        sx = max(0, min(GRID_SIZE - 1, int(x)))
        sy = max(0, min(GRID_SIZE - 1, int(y)))
        cx, cy = cell
        if cx > sx and cy > sy:
            # SE: flanking cell (sx, cy) — clip risk if wall there
            if self._blocking_building_at_cell(sx, cy) is not None:
                point = (float(cx), y)
            else:
                point = (cx + 0.5, cy + 0.5)
        elif cx > sx and cy < sy:
            # NE: flanking cell (sx, cy) — clip risk if wall there
            if self._blocking_building_at_cell(sx, cy) is not None:
                point = (float(cx), y)
            else:
                point = (cx + 0.5, cy + 0.5)
        elif cx < sx and cy > sy:
            # SW: flanking cell (cx, sy) — clip risk if wall there
            if self._blocking_building_at_cell(cx, sy) is not None:
                point = (float(cx + 1), y)
            else:
                point = (cx + 0.5, cy + 0.5)
        elif cx < sx and cy < sy:
            # NW: flanking cell (cx, sy) — clip risk if wall there
            if self._blocking_building_at_cell(cx, sy) is not None:
                point = (float(cx + 1), y)
            else:
                point = (cx + 0.5, cy + 0.5)
        elif cx > sx:
            point = (float(cx), y)
        elif cx < sx:
            point = (float(cx + 1), y)
        elif cy > sy:
            point = (x, float(cy))
        elif cy < sy:
            point = (x, float(cy + 1))
        else:
            point = (cx + 0.5, cy + 0.5)
        if math.hypot(point[0] - x, point[1] - y) <= 1e-9:
            return cx + 0.5, cy + 0.5
        return point

    def _building_at(self, px: float, py: float) -> Building | None:
        for b in self.buildings:
            if b.alive and b.distance_to(px, py) <= 1e-9:
                return b
        return None

    def _ensure_cell_index(self) -> None:
        if self._cell_index_version == self.terrain_version:
            return
        index: dict[tuple[int, int], Building] = {}
        for b in self.buildings:
            if not b.alive or not b.spec.blocks_movement:
                continue
            for cx in range(b.x, b.x + b.spec.size):
                for cy in range(b.y, b.y + b.spec.size):
                    index[(cx, cy)] = b
        self._cell_index = index
        self._cell_index_version = self.terrain_version

    def _blocking_building(self, px: float, py: float, target_id: int | None = None) -> Building | None:
        self._ensure_cell_index()
        b = self._cell_index.get((int(px), int(py)))
        if b is None or b.id == target_id:
            return None
        if not self._point_inside_building(px, py, b):
            return None
        return b

    def _blocking_building_at_cell(self, x: int, y: int, target_id: int | None = None) -> Building | None:
        self._ensure_cell_index()
        b = self._cell_index.get((x, y))
        if b is None or b.id == target_id:
            return None
        return b

    def _point_inside_building(self, px: float, py: float, building: Building) -> bool:
        eps = 1e-6
        return (
            building.x + eps < px < building.x + building.spec.size - eps
            and building.y + eps < py < building.y + building.spec.size - eps
        )

    def _damage_building(
        self,
        building: Building,
        amount: float,
        *,
        rage_bonus: float = 0.0,
        attacker: Troop | None = None,
    ) -> None:
        was_alive = building.alive
        hp_before = max(0.0, building.hp)
        base_amount = max(0.0, amount - max(0.0, rage_bonus))
        building.hp = max(0.0, building.hp - amount)
        actual_damage = hp_before - max(0.0, building.hp)
        if building.spec.counts_for_score and rage_bonus > 0.0 and actual_damage > 0.0:
            base_actual = min(hp_before, base_amount)
            actual_bonus = max(0.0, actual_damage - base_actual)
            self.rage_bonus_scored_damage += actual_bonus
            if building.spec.is_defense:
                self.rage_bonus_defense_damage += actual_bonus
            if building.spec.kind == "townhall":
                self.rage_bonus_townhall_damage += actual_bonus
        if actual_damage > 0.0:
            if building.spec.is_defense:
                self.defense_damage += actual_damage
            if building.spec.kind == "townhall":
                self.townhall_damage += actual_damage
            if building.spec.counts_for_score and attacker is not None:
                origin = self.troop_deploy_origins.get(attacker.id)
                if origin is not None:
                    self.scored_damage_by_origin_quadrant[origin[2]] += actual_damage
        if was_alive and not building.alive:
            if building.spec.is_defense:
                self.defenses_destroyed += 1
            self.terrain_version += 1
            self._flow_cache.clear()

    def _emit_visual_event(self, event: Engagement) -> None:
        event["tick"] = self.tick_count
        self.visual_events.append(event)

    def _should_wall_break(self, troop: Troop, target: Building, distance: float) -> bool:
        return (
            troop.spec.explodes_on_wall
            and target.spec.kind == "wall"
            and distance <= troop.spec.explosion_trigger_range
        )

    def _explode_wall_breaker(self, troop: Troop, target: Building) -> None:
        if not troop.alive or target.spec.kind != "wall":
            return
        self.wall_breaker_explosions += 1
        self.wall_breaker_exploded_ids.add(troop.id)
        wall_count = max(0, troop.spec.wall_damage_count)
        if wall_count > 0 and troop.spec.wall_damage_fraction > 0.0:
            multiplier = self._troop_damage_multiplier(troop)
            for wall in self._connected_walls(target, wall_count):
                was_alive = wall.alive
                hp_before = max(0.0, wall.hp)
                self._damage_building(
                    wall,
                    wall.spec.hp * troop.spec.wall_damage_fraction * multiplier,
                    attacker=troop,
                )
                if hp_before > max(0.0, wall.hp):
                    self.wall_breaker_damaged_wall_segments += 1
                if was_alive and not wall.alive:
                    self.wall_breaker_destroyed_wall_segments += 1
        self._emit_visual_event({
            "kind": "wall_breaker_explosion",
            "source_kind": troop.spec.kind,
            "attacker_id": troop.id,
            "target_id": target.id,
            "from_x": troop.x,
            "from_y": troop.y,
            "to_x": troop.x,
            "to_y": troop.y,
            "radius": 1.6,
        })
        troop.hp = 0.0

    def _connected_walls(self, start: Building, limit: int) -> list[Building]:
        walls_by_cell: dict[tuple[int, int], Building] = {
            (b.x, b.y): b
            for b in self.buildings
            if b.alive and b.spec.kind == "wall"
        }
        start_cell = (start.x, start.y)
        if start_cell not in walls_by_cell:
            return []

        out: list[Building] = []
        seen = {start_cell}
        queue = [start_cell]
        while queue and len(out) < limit:
            cell = queue.pop(0)
            wall = walls_by_cell[cell]
            out.append(wall)
            x, y = cell
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in seen or neighbor not in walls_by_cell:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        return out

    def _next_path_cell(self, troop: Troop, target: Building) -> tuple[int, int] | None:
        return self._next_path_cell_from(troop.x, troop.y, target)

    def _next_path_cell_from(self, x: float, y: float, target: Building) -> tuple[int, int] | None:
        sx = max(0, min(GRID_SIZE - 1, int(x)))
        sy = max(0, min(GRID_SIZE - 1, int(y)))
        flow = self._flow_to_target(target)
        if math.isinf(flow[sy][sx]):
            return None
        best = (flow[sy][sx], sx, sy)
        for nx, ny in self._neighbors(sx, sy):
            cost = flow[ny][nx]
            if cost < best[0]:
                best = (cost, nx, ny)
        if best[1] == sx and best[2] == sy:
            return None
        return best[1], best[2]

    def _flow_to_target(self, target: Building) -> FlowField:
        key = (target.id, self.terrain_version)
        cached = self._flow_cache.get(key)
        if cached is not None:
            return cached

        flow = [[math.inf for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        blocked = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        wall_cells: set[tuple[int, int]] = set()

        for b in self.buildings:
            if not b.alive or not b.spec.blocks_movement or b.id == target.id:
                continue
            for y in range(b.y, b.y + b.spec.size):
                for x in range(b.x, b.x + b.spec.size):
                    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
                        continue
                    if b.spec.kind == "wall":
                        wall_cells.add((x, y))
                    else:
                        blocked[y][x] = True

        frontier: list[tuple[float, int, int]] = []
        for x, y in self._attack_cells(target, blocked, wall_cells):
            flow[y][x] = 0.0
            heapq.heappush(frontier, (0.0, x, y))

        while frontier:
            cost, x, y = heapq.heappop(frontier)
            if cost != flow[y][x]:
                continue
            for nx, ny in self._neighbors(x, y):
                if blocked[ny][nx]:
                    continue
                step_cost = WALL_STEP_COST if (nx, ny) in wall_cells else 1.0
                new_cost = cost + step_cost
                if new_cost < flow[ny][nx]:
                    flow[ny][nx] = new_cost
                    heapq.heappush(frontier, (new_cost, nx, ny))

        self._flow_cache[key] = flow
        return flow

    def _attack_cells(
        self,
        target: Building,
        blocked: list[list[bool]],
        wall_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        x0, y0 = target.x, target.y
        x1 = target.x + target.spec.size - 1
        y1 = target.y + target.spec.size - 1
        for x in range(x0, x1 + 1):
            for y in (y0 - 1, y1 + 1):
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and not blocked[y][x] and (x, y) not in wall_cells:
                    cells.append((x, y))
        for y in range(y0, y1 + 1):
            for x in (x0 - 1, x1 + 1):
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and not blocked[y][x] and (x, y) not in wall_cells:
                    cells.append((x, y))
        return cells

    def _neighbors(self, x: int, y: int) -> tuple[tuple[int, int], ...]:
        out: list[tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    out.append((nx, ny))
        return tuple(out)

    def _nearest_troop_in_range(self, defense: Building) -> Troop | None:
        best: Troop | None = None
        best_d = math.inf
        cx, cy = defense.center
        for t in self.troops:
            if not t.alive:
                continue
            d = math.hypot(t.x - cx, t.y - cy)
            if defense.spec.min_attack_range <= d <= defense.spec.attack_range and d < best_d:
                best_d = d
                best = t
        return best

    def _nearest_troop_to_point(self, x: float, y: float, radius: float) -> Troop | None:
        best: Troop | None = None
        best_d = math.inf
        for t in self.troops:
            if not t.alive:
                continue
            d = math.hypot(t.x - x, t.y - y)
            if d <= radius and d < best_d:
                best_d = d
                best = t
        return best

    def _attack_damage(self, building: Building) -> float:
        if building.spec.damage > 0.0:
            return building.spec.damage
        cooldown = max(TICK_SECONDS, building.spec.attack_cooldown)
        return building.spec.dps * cooldown

    def _active_spell_at(self, spell_kind: str, x: float, y: float) -> ActiveSpell | None:
        best: ActiveSpell | None = None
        best_remaining = -math.inf
        for spell in self.active_spells:
            if spell.kind != spell_kind:
                continue
            if math.hypot(spell.x - x, spell.y - y) > spell.spec.radius:
                continue
            if spell.remaining > best_remaining:
                best = spell
                best_remaining = spell.remaining
        return best

    def _spell_cast_has_target(self, x: float, y: float, spell_kind: str) -> bool:
        spec = SPELL_SPECS[spell_kind]
        if spell_kind == "rage":
            return any(
                math.hypot(t.x - x, t.y - y) <= spec.radius
                and self.troop_has_attack_opportunity(t)
                for t in self.active_troops
            )
        if spell_kind == "freeze":
            return any(
                b.alive and b.spec.is_defense and math.hypot(b.center[0] - x, b.center[1] - y) <= spec.radius
                and self._nearest_troop_in_range(b) is not None
                for b in self.buildings
            )
        return True

    def _troop_damage_multiplier(self, troop: Troop) -> float:
        spell = self._active_spell_at("rage", troop.x, troop.y)
        return 1.0 if spell is None else max(1.0, spell.spec.damage_multiplier)

    def _defense_frozen(self, defense: Building) -> bool:
        cx, cy = defense.center
        return self._active_spell_at("freeze", cx, cy) is not None

    def _drop_expired_spells(self) -> None:
        self.active_spells = [spell for spell in self.active_spells if spell.remaining > 1e-9]

    def _advance_spell_durations(self) -> None:
        for spell in self.active_spells:
            spell.remaining -= TICK_SECONDS
        self._drop_expired_spells()

    def _update_defenses(self) -> None:
        for b in self.buildings:
            if not b.alive or not b.spec.is_defense:
                continue
            if self._defense_frozen(b):
                self.frozen_defense_ticks += 1
                if self._nearest_troop_in_range(b) is not None:
                    self.frozen_threat_ticks += 1
                    self.frozen_threat_damage_prevented += self._frozen_threat_damage_per_tick(b)
                continue
            b.cooldown_remaining -= TICK_SECONDS
            if b.cooldown_remaining > 1e-9:
                continue
            victim = self._nearest_troop_in_range(b)
            if victim is None:
                b.cooldown_remaining = 0.0
                continue
            self._fire_defense(b, victim)
            b.cooldown_remaining += max(TICK_SECONDS, b.spec.attack_cooldown)

    def _frozen_threat_damage_per_tick(self, defense: Building) -> float:
        cooldown = max(TICK_SECONDS, defense.spec.attack_cooldown)
        damage_per_attack = defense.spec.damage if defense.spec.damage > 0.0 else defense.spec.dps * cooldown
        splash_multiplier = 1.0 + min(1.0, max(0.0, defense.spec.splash_radius) / 2.0)
        return damage_per_attack / cooldown * TICK_SECONDS * splash_multiplier

    def _fire_defense(self, defense: Building, victim: Troop) -> None:
        damage = self._attack_damage(defense)
        radius = defense.spec.splash_radius
        cx, cy = defense.center
        impact_tick = self.tick_count + max(
            0,
            math.ceil(defense.spec.projectile_delay / TICK_SECONDS),
        )
        self._emit_visual_event({
            "kind": "defense_fire",
            "source_kind": defense.spec.kind,
            "attacker_id": defense.id,
            "target_id": victim.id,
            "from_x": cx,
            "from_y": cy,
            "to_x": victim.x,
            "to_y": victim.y,
            "attack_cooldown": defense.spec.attack_cooldown,
            "projectile_delay": defense.spec.projectile_delay,
            "splash_radius": radius,
            "impact_tick": impact_tick,
        })
        if defense.spec.projectile_delay > 0.0:
            self.pending_impacts.append(PendingImpact(
                from_x=cx,
                from_y=cy,
                x=victim.x,
                y=victim.y,
                damage=damage,
                radius=radius,
                delay=defense.spec.projectile_delay,
                source_id=defense.id,
                source_kind=defense.spec.kind,
                target_id=victim.id,
                impact_tick=impact_tick,
            ))
            return
        self._damage_troops_at(
            victim.x,
            victim.y,
            radius,
            damage,
            fallback=victim,
            source_kind=defense.spec.kind,
        )
        self._emit_visual_event({
            "kind": "impact",
            "source_kind": defense.spec.kind,
            "attacker_id": defense.id,
            "target_id": victim.id,
            "from_x": cx,
            "from_y": cy,
            "to_x": victim.x,
            "to_y": victim.y,
            "radius": radius,
        })

    def _update_traps(self) -> None:
        for b in self.buildings:
            if not b.alive or not b.spec.is_trap:
                continue
            if b.triggered:
                b.cooldown_remaining -= TICK_SECONDS
                if b.cooldown_remaining > 1e-9:
                    continue
                self._damage_troops_at(
                    *b.center,
                    b.spec.splash_radius,
                    self._attack_damage(b),
                    source_kind=b.spec.kind,
                )
                if b.spec.one_shot:
                    self._damage_building(b, b.hp)
                self._emit_visual_event({
                    "kind": "impact",
                    "source_kind": b.spec.kind,
                    "attacker_id": b.id,
                    "target_id": -1,
                    "from_x": b.center[0],
                    "from_y": b.center[1],
                    "to_x": b.center[0],
                    "to_y": b.center[1],
                    "radius": b.spec.splash_radius,
                })
                b.triggered = False
                continue

            cx, cy = b.center
            if self._nearest_troop_to_point(cx, cy, b.spec.trigger_radius) is None:
                continue
            b.revealed = True
            b.triggered = True
            b.cooldown_remaining = b.spec.trigger_delay

    def _advance_impacts(self) -> None:
        remaining: list[PendingImpact] = []
        for impact in self.pending_impacts:
            impact.delay -= TICK_SECONDS
            if impact.delay > 1e-9:
                remaining.append(impact)
                continue
            self._damage_troops_at(
                impact.x,
                impact.y,
                impact.radius,
                impact.damage,
                source_kind=impact.source_kind,
            )
            self._emit_visual_event({
                "kind": "impact",
                "source_kind": impact.source_kind,
                "attacker_id": impact.source_id,
                "target_id": impact.target_id,
                "from_x": impact.from_x,
                "from_y": impact.from_y,
                "to_x": impact.x,
                "to_y": impact.y,
                "radius": impact.radius,
                "impact_tick": impact.impact_tick,
            })
        self.pending_impacts = remaining

    def _damage_troops_at(
        self,
        x: float,
        y: float,
        radius: float,
        damage: float,
        *,
        fallback: Troop | None = None,
        source_kind: str | None = None,
    ) -> None:
        if radius <= 0.0:
            if fallback is not None and fallback.alive:
                before = max(0.0, fallback.hp)
                fallback.hp = max(0.0, fallback.hp - damage)
                self.troop_damage_taken += before - max(0.0, fallback.hp)
            return
        hits = 0
        damage_done = 0.0
        for t in self.troops:
            if t.alive and math.hypot(t.x - x, t.y - y) <= radius:
                before = max(0.0, t.hp)
                t.hp = max(0.0, t.hp - damage)
                actual = before - max(0.0, t.hp)
                if actual > 0.0:
                    hits += 1
                    damage_done += actual
        self.troop_damage_taken += damage_done
        if damage_done > 0.0 and source_kind is not None:
            self.splash_damage_taken += damage_done
            if hits >= 2:
                self.multi_hit_splash_events += 1
                self.multi_hit_splash_damage_taken += damage_done

    def _current_splash_cluster_risk(self) -> float:
        troops = self.active_troops
        if len(troops) < 2:
            return 0.0
        risk = 0.0
        for defense in self.buildings:
            if (
                not defense.alive
                or not defense.spec.is_defense
                or defense.spec.splash_radius <= 0.0
                or self._defense_frozen(defense)
            ):
                continue
            cx, cy = defense.center
            threatened = [
                troop
                for troop in troops
                if defense.spec.min_attack_range <= math.hypot(troop.x - cx, troop.y - cy) <= defense.spec.attack_range
            ]
            if len(threatened) < 2:
                continue
            damage_per_tick = self._frozen_threat_damage_per_tick(defense)
            for victim in threatened:
                extra_hits = sum(
                    1
                    for other in troops
                    if other.id != victim.id and math.hypot(other.x - victim.x, other.y - victim.y) <= defense.spec.splash_radius
                )
                risk += damage_per_tick * extra_hits
        return risk

    # ── visualisation helper ─────────────────────────────────────────────
    def current_engagements(self) -> list[Engagement]:
        out: list[Engagement] = []
        for spell in self.active_spells:
            out.append({
                "kind": "spell",
                "source_kind": spell.kind,
                "attacker_id": -1,
                "target_id": -1,
                "from_x": spell.x,
                "from_y": spell.y,
                "to_x": spell.x,
                "to_y": spell.y,
                "radius": spell.spec.radius,
                "duration_remaining": spell.remaining,
            })
        for b in self.buildings:
            if not b.alive or not b.spec.is_defense:
                continue
            victim = self._nearest_troop_in_range(b)
            if victim is None:
                continue
            cx, cy = b.center
            out.append({
                "kind": "defense",
                "source_kind": b.spec.kind,
                "attacker_id": b.id, "target_id": victim.id,
                "from_x": cx, "from_y": cy,
                "to_x": victim.x, "to_y": victim.y,
                "attack_cooldown": b.spec.attack_cooldown,
                "projectile_delay": b.spec.projectile_delay,
                "splash_radius": b.spec.splash_radius,
            })
        for t in self.troops:
            if not t.alive:
                continue
            target = self._target_for(t)
            if target is None:
                continue
            if target.distance_to(t.x, t.y) <= t.spec.attack_range + 0.7:
                tx, ty = target.nearest_point(t.x, t.y)
                out.append({
                    "kind": "troop",
                    "source_kind": t.spec.kind,
                    "attacker_id": t.id, "target_id": target.id,
                    "from_x": t.x, "from_y": t.y,
                    "to_x": tx, "to_y": ty,
                })
        return out

    # ── scoring & termination ────────────────────────────────────────────
    @property
    def damage_pct(self) -> float:
        if self.original_total_hp <= 0:
            return 0.0
        current_hp = sum(max(0.0, b.hp) for b in self.scored_buildings)
        return max(0.0, (self.original_total_hp - current_hp) / self.original_total_hp)

    @property
    def townhall_destroyed(self) -> bool:
        return not any(b.alive and b.spec.kind == "townhall" for b in self.buildings)

    @property
    def stars(self) -> int:
        s = 0
        if self.damage_pct >= 0.5:
            s += 1
        if self.townhall_destroyed:
            s += 1
        if self.damage_pct >= 0.9999:
            s += 1
        return s

    @property
    def score(self) -> float:
        """Canonical scalar attack score the reward is the delta of.

        Stars dominate percentage damage, matching the usual attack ranking:
        more stars always beats fewer stars; damage breaks ties within a star
        count. Bounded in [0, 4]. Sum of per-step deltas equals final score.
        """
        return float(self.stars) + self.damage_pct

    @property
    def is_terminal(self) -> bool:
        """Episode ends from a natural terminal state.

        Either every building is dead, or the army is exhausted and no troops
        remain on the field. Time-limit truncation is NOT terminal.
        """
        if not any(b.alive for b in self.scored_buildings):
            return True
        if self.army_remaining == 0 and not any(t.alive for t in self.troops):
            return True
        return False

    @property
    def is_truncated(self) -> bool:
        """Episode ends because the time limit fired (no other terminal cause)."""
        return self.tick_count >= self.max_ticks and not self.is_terminal

    @property
    def is_done(self) -> bool:
        return self.is_terminal or self.is_truncated
