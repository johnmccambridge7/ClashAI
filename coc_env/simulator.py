from __future__ import annotations
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

    def __init__(self, layout: list[Building], army_size: int, seed: int = 0):
        if army_size < 0:
            raise ValueError("army_size must be non-negative")
        self.buildings: list[Building] = [
            Building(id=b.id, spec=b.spec, x=b.x, y=b.y) for b in layout
        ]
        self.army_size: int = int(army_size)
        self.army_remaining: int = int(army_size)
        self.troops: list[Troop] = []
        self.tick_count: int = 0
        self.next_troop_id: int = 0
        self.original_total_hp: float = float(sum(
            b.spec.hp for b in self.buildings if b.spec.counts_for_score
        ))

    # ── view helpers ─────────────────────────────────────────────────────
    @property
    def alive_buildings(self) -> list[Building]:
        return [b for b in self.buildings if b.alive]

    @property
    def active_troops(self) -> list[Troop]:
        return [t for t in self.troops if t.alive]

    @property
    def scored_buildings(self) -> list[Building]:
        return [b for b in self.buildings if b.spec.counts_for_score]

    # ── deploy ───────────────────────────────────────────────────────────
    def deploy(self, x: float, y: float) -> bool:
        if self.army_remaining <= 0:
            return False
        if not (0.0 <= x <= GRID_SIZE - 1 and 0.0 <= y <= GRID_SIZE - 1):
            return False
        if self._building_at(x, y) is not None:
            return False
        self.army_remaining -= 1
        self.troops.append(
            Troop(id=self.next_troop_id, spec=TROOP_SPECS["barbarian"],
                  x=float(x), y=float(y))
        )
        self.next_troop_id += 1
        return True

    # ── single tick ──────────────────────────────────────────────────────
    def tick(self) -> None:
        if self.is_done:
            return
        self.tick_count += 1

        for t in self.troops:
            if not t.alive:
                continue
            if self._target_for(t) is None:
                t.target_id = self._pick_target_id(t)

        for t in self.troops:
            if not t.alive:
                continue
            target = self._target_for(t)
            if target is None:
                continue
            d = target.distance_to(t.x, t.y)
            if d <= t.spec.attack_range + 0.7:
                target.hp = max(0.0, target.hp - t.spec.dps * TICK_SECONDS)
            else:
                self._move_toward(t, target)

        for b in self.buildings:
            if not b.alive or not b.spec.is_defense:
                continue
            victim = self._nearest_troop_in_range(b)
            if victim is not None:
                victim.hp = max(0.0, victim.hp - b.spec.dps * TICK_SECONDS)

    # ── targeting / motion ───────────────────────────────────────────────
    def _pick_target_id(self, troop: Troop) -> int | None:
        best_id: int | None = None
        best_d = math.inf
        for b in self.buildings:
            if not b.alive or not b.spec.counts_for_score:
                continue
            d = b.distance_to(troop.x, troop.y)
            if d < best_d:
                best_d = d
                best_id = b.id
        return best_id

    def _target_for(self, troop: Troop) -> Building | None:
        if troop.target_id is None:
            return None
        for b in self.buildings:
            if b.id == troop.target_id and b.alive:
                return b
        return None

    def _move_toward(self, troop: Troop, target: Building) -> None:
        tx, ty = target.nearest_point(troop.x, troop.y)
        dx = tx - troop.x
        dy = ty - troop.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            return
        # speed is tiles/sec; convert to tiles/tick
        step = min(troop.spec.speed * TICK_SECONDS, dist)
        nx = troop.x + step * dx / dist
        ny = troop.y + step * dy / dist
        wall = self._blocking_wall(nx, ny)
        if wall is not None:
            troop.target_id = wall.id
            wall.hp = max(0.0, wall.hp - troop.spec.dps * TICK_SECONDS)
            return
        troop.x = nx
        troop.y = ny

    def _building_at(self, px: float, py: float) -> Building | None:
        for b in self.buildings:
            if b.alive and b.distance_to(px, py) <= 1e-9:
                return b
        return None

    def _blocking_wall(self, px: float, py: float) -> Building | None:
        for b in self.buildings:
            if b.alive and b.spec.blocks_movement and b.distance_to(px, py) <= 1e-9:
                return b
        return None

    def _nearest_troop_in_range(self, defense: Building) -> Troop | None:
        best: Troop | None = None
        best_d = math.inf
        for t in self.troops:
            if not t.alive:
                continue
            d = defense.distance_to(t.x, t.y)
            if d <= defense.spec.attack_range and d < best_d:
                best_d = d
                best = t
        return best

    # ── visualisation helper ─────────────────────────────────────────────
    def current_engagements(self) -> list[Engagement]:
        out: list[Engagement] = []
        for b in self.buildings:
            if not b.alive or not b.spec.is_defense:
                continue
            victim = self._nearest_troop_in_range(b)
            if victim is None:
                continue
            cx, cy = b.center
            out.append({
                "kind": "defense",
                "attacker_id": b.id, "target_id": victim.id,
                "from_x": cx, "from_y": cy,
                "to_x": victim.x, "to_y": victim.y,
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
        """Canonical scalar score the reward is the delta of.

        score = damage_pct + 0.5·townhall_destroyed + 0.5·full_destruction
        bounded in [0, 2]. Sum of per-step deltas equals final score.
        """
        s = float(self.damage_pct)
        if self.townhall_destroyed:
            s += 0.5
        if self.damage_pct >= 0.9999:
            s += 0.5
        return s

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
