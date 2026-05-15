from __future__ import annotations
from dataclasses import dataclass
import heapq
import math

from .entities import (
    Building,
    Troop,
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
    x: float
    y: float
    damage: float
    radius: float
    delay: float
    source_id: int
    source_kind: str


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
    ):
        if army_size < 0:
            raise ValueError("army_size must be non-negative")
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
        self.buildings: list[Building] = [
            Building(id=b.id, spec=b.spec, x=b.x, y=b.y) for b in layout
        ]
        self.army_remaining_by_kind: dict[str, int] = composition
        self.army_size: int = sum(composition.values())
        self.troops: list[Troop] = []
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

    @property
    def army_remaining(self) -> int:
        return sum(self.army_remaining_by_kind.values())

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
        self.troops.append(
            Troop(id=self.next_troop_id, spec=TROOP_SPECS[troop_kind],
                  x=float(x), y=float(y))
        )
        self.next_troop_id += 1
        return True

    # ── single tick ──────────────────────────────────────────────────────
    def tick(self) -> None:
        if self.is_done:
            return
        self.visual_events = []
        self.tick_count += 1

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
                self._damage_building(target, t.spec.dps * TICK_SECONDS)
            else:
                self._move_toward(t, target)

        self._update_defenses()

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
                self._damage_building(cell_blocker, troop.spec.dps * TICK_SECONDS)
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
            self._damage_building(blocker, troop.spec.dps * TICK_SECONDS)
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

    def _damage_building(self, building: Building, amount: float) -> None:
        was_alive = building.alive
        building.hp = max(0.0, building.hp - amount)
        if was_alive and not building.alive:
            self.terrain_version += 1
            self._flow_cache.clear()

    def _should_wall_break(self, troop: Troop, target: Building, distance: float) -> bool:
        return (
            troop.spec.explodes_on_wall
            and target.spec.kind == "wall"
            and distance <= troop.spec.explosion_trigger_range
        )

    def _explode_wall_breaker(self, troop: Troop, target: Building) -> None:
        if not troop.alive or target.spec.kind != "wall":
            return
        wall_count = max(0, troop.spec.wall_damage_count)
        if wall_count > 0 and troop.spec.wall_damage_fraction > 0.0:
            for wall in self._connected_walls(target, wall_count):
                self._damage_building(wall, wall.spec.hp * troop.spec.wall_damage_fraction)
        self.visual_events.append({
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

    def _update_defenses(self) -> None:
        for b in self.buildings:
            if not b.alive or not b.spec.is_defense:
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

    def _fire_defense(self, defense: Building, victim: Troop) -> None:
        damage = self._attack_damage(defense)
        radius = defense.spec.splash_radius
        if defense.spec.projectile_delay > 0.0:
            self.pending_impacts.append(PendingImpact(
                x=victim.x,
                y=victim.y,
                damage=damage,
                radius=radius,
                delay=defense.spec.projectile_delay,
                source_id=defense.id,
                source_kind=defense.spec.kind,
            ))
            return
        self._damage_troops_at(victim.x, victim.y, radius, damage, fallback=victim)

    def _update_traps(self) -> None:
        for b in self.buildings:
            if not b.alive or not b.spec.is_trap:
                continue
            if b.triggered:
                b.cooldown_remaining -= TICK_SECONDS
                if b.cooldown_remaining > 1e-9:
                    continue
                self._damage_troops_at(*b.center, b.spec.splash_radius, self._attack_damage(b))
                if b.spec.one_shot:
                    self._damage_building(b, b.hp)
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
            self._damage_troops_at(impact.x, impact.y, impact.radius, impact.damage)
        self.pending_impacts = remaining

    def _damage_troops_at(
        self,
        x: float,
        y: float,
        radius: float,
        damage: float,
        *,
        fallback: Troop | None = None,
    ) -> None:
        if radius <= 0.0:
            if fallback is not None and fallback.alive:
                fallback.hp = max(0.0, fallback.hp - damage)
            return
        for t in self.troops:
            if t.alive and math.hypot(t.x - x, t.y - y) <= radius:
                t.hp = max(0.0, t.hp - damage)

    # ── visualisation helper ─────────────────────────────────────────────
    def current_engagements(self) -> list[Engagement]:
        out: list[Engagement] = list(self.visual_events)
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
        for impact in self.pending_impacts:
            out.append({
                "kind": "impact",
                "source_kind": impact.source_kind,
                "attacker_id": impact.source_id,
                "target_id": -1,
                "from_x": impact.x, "from_y": impact.y,
                "to_x": impact.x, "to_y": impact.y,
                "radius": impact.radius,
                "delay": impact.delay,
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
        return self.tick_count >= MAX_TICKS and not self.is_terminal

    @property
    def is_done(self) -> bool:
        return self.is_terminal or self.is_truncated
