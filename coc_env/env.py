from __future__ import annotations
import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .entities import BUILDING_SPECS, SPELL_SPECS, TROOP_SPECS, GRID_SIZE, MAX_TICKS, DECISION_INTERVAL, Building
from .simulator import Simulator
from .layouts import default_layout
from .generation import LayoutProfile, generate_layout, preset_layout_profile


# Deploy actions map one-to-one to grid cells; WAIT_ACTION is always legal.
DEFAULT_ARMY_SIZE: int = 50
BUILDING_KIND_INDEX: dict[str, int] = {
    kind: i for i, kind in enumerate(BUILDING_SPECS)
}
DEPLOY_TROOP_KINDS: tuple[str, ...] = tuple(TROOP_SPECS)
TROOP_KIND_INDEX: dict[str, int] = {
    kind: i for i, kind in enumerate(DEPLOY_TROOP_KINDS)
}
DEPLOY_SPELL_KINDS: tuple[str, ...] = tuple(SPELL_SPECS)
SPELL_KIND_INDEX: dict[str, int] = {
    kind: i for i, kind in enumerate(DEPLOY_SPELL_KINDS)
}
_DPS_NORM: float = 30.0
_RANGE_NORM: float = float(GRID_SIZE)
_SECONDS_NORM: float = 5.0
TIME_COST_PER_TICK: float = 1e-4
TOWNHALL_DEPLOY_BUFFER: float = 2.0
RAGE_EFFECT_REWARD_COEF: float = 0.15
FREEZE_THREAT_TICK_REWARD: float = 0.002
WASTED_SPELL_CAST_PENALTY: float = 0.02
UNUSED_SPELL_TERMINAL_PENALTY: float = 0.03
MAX_RAGE_EFFECT_REWARD: float = 0.10
MAX_FREEZE_EFFECT_REWARD: float = 0.10
MAX_WASTED_SPELL_PENALTY: float = 0.08
MAX_UNUSED_SPELL_TERMINAL_PENALTY: float = 0.12
RAGE_MASK_TOP_K: int = 32
FREEZE_MASK_TOP_K: int = 24
FREEZE_THREAT_BUFFER: float = 2.0


N_DEPLOY_CELLS: int = GRID_SIZE * GRID_SIZE
N_DEPLOY_ACTIONS: int = N_DEPLOY_CELLS * len(DEPLOY_TROOP_KINDS)
N_SPELL_ACTIONS: int = N_DEPLOY_CELLS * len(DEPLOY_SPELL_KINDS)
WAIT_ACTION: int = N_DEPLOY_ACTIONS
DEPLOY_CELLS: list[tuple[int, int]] = [
    (x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)
]
# Backward-compatible names for older viewer/scripts. These now refer to all
# deployable grid cells, not just the outer perimeter.
PERIMETER: list[tuple[int, int]] = DEPLOY_CELLS
N_PERIMETER: int = N_DEPLOY_CELLS


def cell_to_action(x: int, y: int, troop_kind: str = "barbarian") -> int:
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        raise ValueError(f"cell ({x}, {y}) is outside the grid")
    try:
        troop_offset = TROOP_KIND_INDEX[troop_kind] * N_DEPLOY_CELLS
    except KeyError as exc:
        raise ValueError(f"unknown troop kind {troop_kind!r}") from exc
    return troop_offset + y * GRID_SIZE + x


def action_to_cell(action: int) -> tuple[int, int]:
    if not 0 <= action < N_DEPLOY_CELLS:
        raise ValueError(f"action {action} is not a deploy-cell action")
    return action % GRID_SIZE, action // GRID_SIZE


def decode_deploy_action(action: int) -> tuple[str, int, int]:
    if not 0 <= action < N_DEPLOY_ACTIONS:
        raise ValueError(f"action {action} is not a deploy action")
    troop_kind = DEPLOY_TROOP_KINDS[action // N_DEPLOY_CELLS]
    x, y = action_to_cell(action % N_DEPLOY_CELLS)
    return troop_kind, x, y


def spell_to_action(x: int, y: int, spell_kind: str) -> int:
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        raise ValueError(f"cell ({x}, {y}) is outside the grid")
    try:
        spell_offset = SPELL_KIND_INDEX[spell_kind] * N_DEPLOY_CELLS
    except KeyError as exc:
        raise ValueError(f"unknown spell kind {spell_kind!r}") from exc
    return WAIT_ACTION + 1 + spell_offset + y * GRID_SIZE + x


def decode_spell_action(action: int) -> tuple[str, int, int]:
    spell_action = action - (WAIT_ACTION + 1)
    if not 0 <= spell_action < N_SPELL_ACTIONS:
        raise ValueError(f"action {action} is not a spell action")
    spell_kind = DEPLOY_SPELL_KINDS[spell_action // N_DEPLOY_CELLS]
    x, y = action_to_cell(spell_action % N_DEPLOY_CELLS)
    return spell_kind, x, y


class CoCEnv(gym.Env[dict[str, np.ndarray], int]):
    """Single-troop Clash of Clans MVP.

    Action space: Discrete(GRID_SIZE * GRID_SIZE + 1).
        deploy action y*GRID_SIZE+x -> deploy at cell (x, y) if legal
        action == WAIT_ACTION       -> no-op deploy
    Each step advances DECISION_INTERVAL simulator ticks.

    Observation: Dict with stable per-slot arrays.
        buildings_*  — shape (n_buildings,) where index = padded layout slot
        troops_*     — shape (army_size,)   where index = troop id
    Slot identity is stable for the whole episode. A destroyed building keeps
    present=1, alive=0, hp=0; unused padding slots have present=0. Hidden
    traps use present=0 until revealed.

    Reward: per-step delta of `Simulator.score`, minus a tiny time cost.
        score = stars + damage_pct
        reward_t = score_t − score_{t−1} − 1e−4·ticks_advanced

    Episode end:
        terminated = sim.is_terminal  (base destroyed, or army+troops exhausted)
        truncated  = sim.is_truncated (max ticks reached, no terminal cause)
    """

    metadata = {"render_modes": [], "name": "CoC-MVP-v0"}

    def __init__(
        self,
        army_size: int = DEFAULT_ARMY_SIZE,
        *,
        layout_profile: LayoutProfile | str | None = None,
        max_buildings: int | None = None,
        army_composition: dict[str, int] | None = None,
        spell_composition: dict[str, int] | None = None,
        max_ticks: int = MAX_TICKS,
    ) -> None:
        super().__init__()
        if army_size < 0:
            raise ValueError("army_size must be non-negative")
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        self.army_composition = _coerce_army_composition(army_size, army_composition)
        self.spell_composition = _coerce_spell_composition(spell_composition)
        self.army_size: int = sum(self.army_composition.values())
        self.spells_enabled: bool = sum(self.spell_composition.values()) > 0
        self.max_spells: int = sum(self.spell_composition.values())
        self.wait_action: int = WAIT_ACTION
        self.n_actions: int = WAIT_ACTION + 1 + (N_SPELL_ACTIONS if self.spells_enabled else 0)
        self.max_ticks: int = int(max_ticks)
        self.layout_profile = _coerce_layout_profile(layout_profile)
        self._current_profile_name = self.layout_profile.name if self.layout_profile is not None else "default"

        required_slots = (
            self.layout_profile.max_buildings
            if self.layout_profile is not None
            else len(default_layout())
        )
        self.n_buildings: int = int(max_buildings) if max_buildings is not None else required_slots
        if self.n_buildings < required_slots:
            raise ValueError(
                f"max_buildings={self.n_buildings} is smaller than required layout capacity "
                f"{required_slots}"
            )

        self.action_space = spaces.Discrete(self.n_actions)
        observation_spaces = {
            # buildings — slot index = position in layout
            "buildings_present":      spaces.MultiBinary(self.n_buildings),
            "buildings_alive":        spaces.MultiBinary(self.n_buildings),
            "buildings_kind":         spaces.Box(0, len(BUILDING_KIND_INDEX) - 1,
                                                 shape=(self.n_buildings,), dtype=np.int8),
            "buildings_hp":           spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_pos":          spaces.Box(0.0, 1.0, shape=(self.n_buildings, 2), dtype=np.float32),
            "buildings_size":         spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_is_defense":   spaces.MultiBinary(self.n_buildings),
            "buildings_blocks_move":   spaces.MultiBinary(self.n_buildings),
            "buildings_dps":          spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_attack_range": spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_min_attack_range": spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_splash_radius": spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_cooldown":     spaces.Box(0.0, 1.0, shape=(self.n_buildings,), dtype=np.float32),
            "buildings_is_trap":      spaces.MultiBinary(self.n_buildings),
            # troops — slot index = troop id ∈ [0, army_size)
            "troops_alive":           spaces.MultiBinary(self.army_size),
            "troops_kind":            spaces.Box(0, len(TROOP_KIND_INDEX) - 1,
                                                 shape=(self.army_size,), dtype=np.int8),
            "troops_hp":              spaces.Box(0.0, 1.0, shape=(self.army_size,), dtype=np.float32),
            "troops_pos":             spaces.Box(0.0, 1.0, shape=(self.army_size, 2), dtype=np.float32),
            "troops_target":          spaces.Box(-1.0, 1.0, shape=(self.army_size,), dtype=np.float32),
            # globals
            "army_remaining":         spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "time_remaining":         spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
        if self.spells_enabled:
            observation_spaces.update({
                "spells_remaining": spaces.Box(
                    0.0, 1.0, shape=(len(DEPLOY_SPELL_KINDS),), dtype=np.float32,
                ),
                "active_spells_present": spaces.MultiBinary(self.max_spells),
                "active_spells_kind": spaces.Box(
                    0, len(SPELL_KIND_INDEX) - 1, shape=(self.max_spells,), dtype=np.int8,
                ),
                "active_spells_pos": spaces.Box(
                    0.0, 1.0, shape=(self.max_spells, 2), dtype=np.float32,
                ),
                "active_spells_radius": spaces.Box(
                    0.0, 1.0, shape=(self.max_spells,), dtype=np.float32,
                ),
                "active_spells_time_remaining": spaces.Box(
                    0.0, 1.0, shape=(self.max_spells,), dtype=np.float32,
                ),
            })
        self.observation_space = spaces.Dict(observation_spaces)

        self.sim: Simulator | None = None
        self._prev_score: float = 0.0
        self._rage_effect_reward_total: float = 0.0
        self._freeze_effect_reward_total: float = 0.0
        self._wasted_spell_penalty_total: float = 0.0
        self._unused_spell_penalty_total: float = 0.0
        self._last_reward_components: dict[str, float] = {}
        self._mask_cache_key: tuple[Any, ...] | None = None
        self._mask_cache: np.ndarray | None = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.sim = Simulator(
            layout=self._make_layout(options),
            army_size=self.army_size,
            seed=seed if seed is not None else 0,
            army_composition=self.army_composition,
            spell_composition=self.spell_composition,
            max_ticks=self.max_ticks,
        )
        self._prev_score = 0.0
        self._rage_effect_reward_total = 0.0
        self._freeze_effect_reward_total = 0.0
        self._wasted_spell_penalty_total = 0.0
        self._unused_spell_penalty_total = 0.0
        self._last_reward_components = {}
        self._mask_cache_key = None
        self._mask_cache = None
        return self._obs(), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.sim is not None, "Call reset() before step()."
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside Discrete({self.n_actions})")

        prev_rage_bonus_damage = self.sim.rage_bonus_scored_damage
        prev_frozen_threat_ticks = self.sim.frozen_threat_ticks
        prev_wasted_spell_casts = self.sim.wasted_spell_casts

        if 0 <= action < N_DEPLOY_ACTIONS:
            troop_kind, x, y = decode_deploy_action(action)
            if self._can_deploy_cell(x, y):
                self.sim.deploy(x + 0.5, y + 0.5, troop_kind=troop_kind)
        elif self.spells_enabled and WAIT_ACTION < action < self.n_actions:
            spell_kind, x, y = decode_spell_action(action)
            if self._can_cast_spell_cell(x, y, spell_kind):
                self.sim.cast_spell(x + 0.5, y + 0.5, spell_kind=spell_kind)
        # wait_action is the only explicit no-op action.

        ticks_advanced = 0
        for _ in range(DECISION_INTERVAL):
            if self.sim.is_done:
                break
            self.sim.tick()
            ticks_advanced += 1

        score = self.sim.score
        base_reward = float(score - self._prev_score - TIME_COST_PER_TICK * ticks_advanced)
        self._prev_score = score

        terminated = bool(self.sim.is_terminal)
        truncated = bool(self.sim.is_truncated)
        rage_damage_delta = max(0.0, self.sim.rage_bonus_scored_damage - prev_rage_bonus_damage)
        rage_damage_pct_delta = rage_damage_delta / max(1e-9, self.sim.original_total_hp)
        rage_reward = self._consume_reward_cap(
            RAGE_EFFECT_REWARD_COEF * rage_damage_pct_delta,
            current=self._rage_effect_reward_total,
            cap=MAX_RAGE_EFFECT_REWARD,
        )
        self._rage_effect_reward_total += rage_reward

        frozen_threat_delta = max(0, self.sim.frozen_threat_ticks - prev_frozen_threat_ticks)
        freeze_reward = self._consume_reward_cap(
            FREEZE_THREAT_TICK_REWARD * frozen_threat_delta,
            current=self._freeze_effect_reward_total,
            cap=MAX_FREEZE_EFFECT_REWARD,
        )
        self._freeze_effect_reward_total += freeze_reward

        wasted_delta = max(0, self.sim.wasted_spell_casts - prev_wasted_spell_casts)
        wasted_penalty = self._consume_reward_cap(
            WASTED_SPELL_CAST_PENALTY * wasted_delta,
            current=self._wasted_spell_penalty_total,
            cap=MAX_WASTED_SPELL_PENALTY,
        )
        self._wasted_spell_penalty_total += wasted_penalty

        unused_penalty = 0.0
        if terminated or truncated:
            unused_penalty = min(
                MAX_UNUSED_SPELL_TERMINAL_PENALTY,
                UNUSED_SPELL_TERMINAL_PENALTY * self.sim.spells_remaining,
            )
            self._unused_spell_penalty_total += unused_penalty

        reward = float(base_reward + rage_reward + freeze_reward - wasted_penalty - unused_penalty)
        self._last_reward_components = {
            "reward_base": base_reward,
            "reward_rage_effect": rage_reward,
            "reward_freeze_effect": freeze_reward,
            "reward_wasted_spell_penalty": wasted_penalty,
            "reward_unused_spell_penalty": unused_penalty,
            "reward_spell_shaping": rage_reward + freeze_reward - wasted_penalty - unused_penalty,
            "rage_bonus_damage_pct_delta": rage_damage_pct_delta,
            "frozen_threat_ticks_delta": float(frozen_threat_delta),
            "wasted_spell_casts_delta": float(wasted_delta),
        }
        return self._obs(), reward, terminated, truncated, self._info()

    # ── helpers ──────────────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        if self.sim is None:
            mask = np.zeros(self.n_actions, dtype=bool)
            mask[self.wait_action] = True
            return mask
        key = (
            id(self.sim),
            self.sim.terrain_version,
            self.sim.tick_count,
            tuple(self.sim.army_remaining_by_kind.get(kind, 0) for kind in DEPLOY_TROOP_KINDS),
            tuple(self.sim.spells_remaining_by_kind.get(kind, 0) for kind in DEPLOY_SPELL_KINDS),
            self.sim.is_done,
        )
        if key == self._mask_cache_key and self._mask_cache is not None:
            return self._mask_cache.copy()
        mask = self._build_action_mask()
        self._mask_cache_key = key
        self._mask_cache = mask
        return mask.copy()

    def _build_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.wait_action] = True
        if self.sim is not None and self.sim.army_remaining > 0 and not self.sim.is_done:
            cell_mask = [self._can_deploy_cell(x, y) for x, y in DEPLOY_CELLS]
            for kind_index, troop_kind in enumerate(DEPLOY_TROOP_KINDS):
                if self.sim.army_remaining_by_kind.get(troop_kind, 0) <= 0:
                    continue
                offset = kind_index * N_DEPLOY_CELLS
                for cell_action, ok in enumerate(cell_mask):
                    mask[offset + cell_action] = ok
        if self.sim is not None and self.spells_enabled and not self.sim.is_done:
            for kind_index, spell_kind in enumerate(DEPLOY_SPELL_KINDS):
                if self.sim.spells_remaining_by_kind.get(spell_kind, 0) <= 0:
                    continue
                offset = WAIT_ACTION + 1 + kind_index * N_DEPLOY_CELLS
                mask[offset : offset + N_DEPLOY_CELLS] = self._semantic_spell_cell_mask(spell_kind)
        return mask

    @staticmethod
    def _consume_reward_cap(value: float, *, current: float, cap: float) -> float:
        if value <= 0.0:
            return 0.0
        return min(value, max(0.0, cap - current))

    def _semantic_spell_cell_mask(self, spell_kind: str) -> np.ndarray:
        assert self.sim is not None
        if not self._can_cast_spell_kind(spell_kind):
            return np.zeros(N_DEPLOY_CELLS, dtype=bool)
        scores = (
            self._rage_cell_scores()
            if spell_kind == "rage"
            else self._freeze_cell_scores()
            if spell_kind == "freeze"
            else np.ones(N_DEPLOY_CELLS, dtype=np.float32)
        )
        top_k = RAGE_MASK_TOP_K if spell_kind == "rage" else FREEZE_MASK_TOP_K
        return self._top_positive_cells(scores, top_k)

    def _can_cast_spell_kind(self, spell_kind: str) -> bool:
        assert self.sim is not None
        return (
            self.spells_enabled
            and self.sim.spells_remaining_by_kind.get(spell_kind, 0) > 0
            and bool(self.sim.active_troops)
            and not self.sim.is_done
        )

    @staticmethod
    def _top_positive_cells(scores: np.ndarray, top_k: int) -> np.ndarray:
        mask = np.zeros(N_DEPLOY_CELLS, dtype=bool)
        positive = np.flatnonzero(scores > 0.0)
        if positive.size == 0:
            return mask
        if positive.size <= top_k:
            mask[positive] = True
            return mask
        ranked = positive[np.argsort(scores[positive])[-top_k:]]
        mask[ranked] = True
        return mask

    def _rage_cell_scores(self) -> np.ndarray:
        assert self.sim is not None
        spec = SPELL_SPECS["rage"]
        scores = np.zeros(N_DEPLOY_CELLS, dtype=np.float32)
        troops = self.sim.active_troops
        if not troops:
            return scores
        active_rages = [spell for spell in self.sim.active_spells if spell.kind == "rage"]
        for cell_action, (x, y) in enumerate(DEPLOY_CELLS):
            px, py = x + 0.5, y + 0.5
            if any(math.hypot(spell.x - px, spell.y - py) <= spec.radius * 0.75 for spell in active_rages):
                continue
            score = 0.0
            for troop in troops:
                d = math.hypot(troop.x - px, troop.y - py)
                if d > spec.radius:
                    continue
                weight = 1.0 + max(0.0, troop.spec.dps)
                if troop.spec.explodes_on_wall:
                    weight += 2.0
                score += weight * (1.0 + (spec.radius - d) / max(1e-9, spec.radius))
            scores[cell_action] = score
        return scores

    def _freeze_cell_scores(self) -> np.ndarray:
        assert self.sim is not None
        spec = SPELL_SPECS["freeze"]
        scores = np.zeros(N_DEPLOY_CELLS, dtype=np.float32)
        troops = self.sim.active_troops
        if not troops:
            return scores
        active_freezes = [spell for spell in self.sim.active_spells if spell.kind == "freeze"]
        defenses = [b for b in self.sim.buildings if b.alive and b.spec.is_defense]
        for cell_action, (x, y) in enumerate(DEPLOY_CELLS):
            px, py = x + 0.5, y + 0.5
            if any(math.hypot(spell.x - px, spell.y - py) <= spec.radius * 0.75 for spell in active_freezes):
                continue
            score = 0.0
            for defense in defenses:
                cx, cy = defense.center
                d = math.hypot(cx - px, cy - py)
                if d > spec.radius:
                    continue
                threat = self._defense_threat_weight(defense)
                if threat <= 0.0:
                    continue
                score += threat * (1.0 + (spec.radius - d) / max(1e-9, spec.radius))
            scores[cell_action] = score
        return scores

    def _defense_threat_weight(self, defense: Building) -> float:
        assert self.sim is not None
        cx, cy = defense.center
        nearest_margin: float | None = None
        for troop in self.sim.active_troops:
            d = math.hypot(troop.x - cx, troop.y - cy)
            if d < defense.spec.min_attack_range:
                continue
            margin = defense.spec.attack_range + FREEZE_THREAT_BUFFER - d
            if margin < 0.0:
                continue
            nearest_margin = margin if nearest_margin is None else max(nearest_margin, margin)
        if nearest_margin is None:
            return 0.0
        attack_rate = (
            defense.spec.damage / max(1e-9, defense.spec.attack_cooldown)
            if defense.spec.damage > 0.0
            else defense.spec.dps
        )
        return max(1.0, attack_rate) * (1.0 + nearest_margin / max(1.0, FREEZE_THREAT_BUFFER))

    def _can_deploy_cell(self, x: int, y: int) -> bool:
        assert self.sim is not None
        px, py = x + 0.5, y + 0.5
        for b in self.sim.buildings:
            if not b.alive:
                continue
            if b.spec.blocks_movement and b.distance_to(px, py) <= 1e-9:
                return False
            if b.spec.kind == "townhall" and b.distance_to(px, py) <= TOWNHALL_DEPLOY_BUFFER:
                return False
            if b.spec.is_defense:
                cx, cy = b.center
                d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if b.spec.min_attack_range <= d <= b.spec.attack_range:
                    return False
        return True

    def _can_cast_spell_cell(self, x: int, y: int, spell_kind: str) -> bool:
        assert self.sim is not None
        return (
            self.spells_enabled
            and 0 <= x < GRID_SIZE
            and 0 <= y < GRID_SIZE
            and self.sim.spells_remaining_by_kind.get(spell_kind, 0) > 0
        )

    def _make_layout(self, options: dict[str, Any] | None) -> list[Building]:
        profile = self.layout_profile
        if options is not None and "layout_profile" in options:
            profile = _coerce_layout_profile(options["layout_profile"])
        if options is not None and "profile" in options:
            profile = _coerce_layout_profile(options["profile"])

        self._current_profile_name = profile.name if profile is not None else "default"
        if profile is None:
            layout = default_layout()
        else:
            layout = generate_layout(profile, self.np_random)
        if len(layout) > self.n_buildings:
            raise RuntimeError(
                f"layout produced {len(layout)} buildings but observation capacity is {self.n_buildings}"
            )
        return layout

    def _obs(self) -> dict[str, np.ndarray]:
        assert self.sim is not None
        nb = self.n_buildings
        nt = self.army_size

        b_present = np.zeros(nb, dtype=np.int8)
        b_alive = np.zeros(nb, dtype=np.int8)
        b_kind = np.zeros(nb, dtype=np.int8)
        b_hp = np.zeros(nb, dtype=np.float32)
        b_pos = np.zeros((nb, 2), dtype=np.float32)
        b_size = np.zeros(nb, dtype=np.float32)
        b_def = np.zeros(nb, dtype=np.int8)
        b_block = np.zeros(nb, dtype=np.int8)
        b_dps = np.zeros(nb, dtype=np.float32)
        b_rng = np.zeros(nb, dtype=np.float32)
        b_min_rng = np.zeros(nb, dtype=np.float32)
        b_splash = np.zeros(nb, dtype=np.float32)
        b_cooldown = np.zeros(nb, dtype=np.float32)
        b_trap = np.zeros(nb, dtype=np.int8)

        for i, b in enumerate(self.sim.buildings):
            if b.spec.hidden and not b.revealed:
                continue
            b_present[i] = 1
            b_alive[i] = 1 if b.alive else 0
            b_kind[i] = BUILDING_KIND_INDEX.get(b.spec.kind, 0)
            b_hp[i] = max(0.0, b.hp) / float(b.spec.hp)
            cx, cy = b.center
            b_pos[i] = (cx / GRID_SIZE, cy / GRID_SIZE)
            b_size[i] = b.spec.size / float(GRID_SIZE)
            b_def[i] = 1 if b.spec.is_defense else 0
            b_block[i] = 1 if b.spec.blocks_movement else 0
            b_dps[i] = b.spec.dps / _DPS_NORM
            b_rng[i] = b.spec.attack_range / _RANGE_NORM
            b_min_rng[i] = b.spec.min_attack_range / _RANGE_NORM
            b_splash[i] = b.spec.splash_radius / _RANGE_NORM
            b_cooldown[i] = min(1.0, max(0.0, b.cooldown_remaining / _SECONDS_NORM))
            b_trap[i] = 1 if b.spec.is_trap else 0

        t_alive = np.zeros(nt, dtype=np.int8)
        t_kind = np.zeros(nt, dtype=np.int8)
        t_hp = np.zeros(nt, dtype=np.float32)
        t_pos = np.zeros((nt, 2), dtype=np.float32)
        t_target = np.full(nt, -1.0, dtype=np.float32)
        building_slots = {b.id: i for i, b in enumerate(self.sim.buildings)}

        for tr in self.sim.troops:
            i = tr.id
            if not (0 <= i < nt):
                continue
            t_alive[i] = 1 if tr.alive else 0
            t_kind[i] = TROOP_KIND_INDEX.get(tr.spec.kind, 0)
            t_hp[i] = max(0.0, tr.hp) / float(tr.spec.hp)
            t_pos[i] = (tr.x / GRID_SIZE, tr.y / GRID_SIZE)
            target_slot = building_slots.get(tr.target_id)
            if target_slot is not None:
                t_target[i] = float(target_slot) / max(1, nb - 1)

        obs = {
            "buildings_present":      b_present,
            "buildings_alive":        b_alive,
            "buildings_kind":         b_kind,
            "buildings_hp":           b_hp,
            "buildings_pos":          b_pos,
            "buildings_size":         b_size,
            "buildings_is_defense":   b_def,
            "buildings_blocks_move":   b_block,
            "buildings_dps":          b_dps,
            "buildings_attack_range": b_rng,
            "buildings_min_attack_range": b_min_rng,
            "buildings_splash_radius": b_splash,
            "buildings_cooldown":     b_cooldown,
            "buildings_is_trap":      b_trap,
            "troops_alive":           t_alive,
            "troops_kind":            t_kind,
            "troops_hp":              t_hp,
            "troops_pos":             t_pos,
            "troops_target":          t_target,
            "army_remaining":         np.array(
                [self.sim.army_remaining / max(1, self.army_size)], dtype=np.float32),
            "time_remaining":         np.array(
                [1.0 - self.sim.tick_count / self.max_ticks], dtype=np.float32),
        }
        if self.spells_enabled:
            spells_remaining = np.zeros(len(DEPLOY_SPELL_KINDS), dtype=np.float32)
            for i, kind in enumerate(DEPLOY_SPELL_KINDS):
                total = max(1, self.spell_composition.get(kind, 0))
                spells_remaining[i] = self.sim.spells_remaining_by_kind.get(kind, 0) / total

            active_present = np.zeros(self.max_spells, dtype=np.int8)
            active_kind = np.zeros(self.max_spells, dtype=np.int8)
            active_pos = np.zeros((self.max_spells, 2), dtype=np.float32)
            active_radius = np.zeros(self.max_spells, dtype=np.float32)
            active_time = np.zeros(self.max_spells, dtype=np.float32)
            for i, spell in enumerate(self.sim.active_spells[:self.max_spells]):
                active_present[i] = 1
                active_kind[i] = SPELL_KIND_INDEX[spell.kind]
                active_pos[i] = (spell.x / GRID_SIZE, spell.y / GRID_SIZE)
                active_radius[i] = spell.spec.radius / GRID_SIZE
                active_time[i] = max(0.0, spell.remaining) / max(1e-9, spell.spec.duration)

            obs.update({
                "spells_remaining": spells_remaining,
                "active_spells_present": active_present,
                "active_spells_kind": active_kind,
                "active_spells_pos": active_pos,
                "active_spells_radius": active_radius,
                "active_spells_time_remaining": active_time,
            })
        return obs

    def _info(self) -> dict[str, Any]:
        assert self.sim is not None
        return {
            "profile":        self._current_profile_name,
            "damage_pct":     self.sim.damage_pct,
            "stars":          self.sim.stars,
            "score":          self.sim.score,
            "ticks_elapsed":  self.sim.tick_count,
            "army_remaining": self.sim.army_remaining,
            "army_remaining_by_kind": dict(self.sim.army_remaining_by_kind),
            "spells_remaining": self.sim.spells_remaining,
            "spells_remaining_by_kind": dict(self.sim.spells_remaining_by_kind),
            "spell_casts": self.sim.spell_casts,
            "spell_casts_by_kind": dict(self.sim.spell_casts_by_kind),
            "useful_spell_casts": self.sim.useful_spell_casts,
            "useful_spell_casts_by_kind": dict(self.sim.useful_spell_casts_by_kind),
            "wasted_spell_casts": self.sim.wasted_spell_casts,
            "wasted_spell_casts_by_kind": dict(self.sim.wasted_spell_casts_by_kind),
            "rage_casts": self.sim.spell_casts_by_kind.get("rage", 0),
            "freeze_casts": self.sim.spell_casts_by_kind.get("freeze", 0),
            "useful_rage_casts": self.sim.useful_spell_casts_by_kind.get("rage", 0),
            "useful_freeze_casts": self.sim.useful_spell_casts_by_kind.get("freeze", 0),
            "wasted_rage_casts": self.sim.wasted_spell_casts_by_kind.get("rage", 0),
            "wasted_freeze_casts": self.sim.wasted_spell_casts_by_kind.get("freeze", 0),
            "rage_bonus_damage_pct": self.sim.rage_bonus_scored_damage / max(1e-9, self.sim.original_total_hp),
            "frozen_defense_ticks": self.sim.frozen_defense_ticks,
            "frozen_threat_ticks": self.sim.frozen_threat_ticks,
            "rage_effect_reward_total": self._rage_effect_reward_total,
            "freeze_effect_reward_total": self._freeze_effect_reward_total,
            "wasted_spell_penalty_total": self._wasted_spell_penalty_total,
            "unused_spell_penalty_total": self._unused_spell_penalty_total,
            "spell_shaping_reward_total": (
                self._rage_effect_reward_total
                + self._freeze_effect_reward_total
                - self._wasted_spell_penalty_total
                - self._unused_spell_penalty_total
            ),
            **self._last_reward_components,
            "active_troops":  len(self.sim.active_troops),
            "active_spells":  len(self.sim.active_spells),
            "is_terminal":    self.sim.is_terminal,
            "is_truncated":   self.sim.is_truncated,
        }


def _coerce_layout_profile(profile: object) -> LayoutProfile | None:
    if profile is None:
        return None
    if isinstance(profile, LayoutProfile):
        return profile
    if isinstance(profile, str):
        return preset_layout_profile(profile)
    raise TypeError("layout_profile must be a LayoutProfile, a preset name, or None")


def _coerce_army_composition(
    army_size: int,
    army_composition: dict[str, int] | None,
) -> dict[str, int]:
    if army_composition is None:
        return {"barbarian": int(army_size)}

    out: dict[str, int] = {}
    for kind, count in army_composition.items():
        if kind not in TROOP_SPECS:
            raise ValueError(f"unknown troop kind {kind!r}")
        if count < 0:
            raise ValueError(f"{kind} count must be non-negative")
        out[kind] = int(count)
    if sum(out.values()) <= 0 and army_size > 0:
        raise ValueError("army_composition must contain at least one troop")
    return out


def _coerce_spell_composition(
    spell_composition: dict[str, int] | None,
) -> dict[str, int]:
    if spell_composition is None:
        return {}

    out: dict[str, int] = {}
    for kind, count in spell_composition.items():
        if kind not in SPELL_SPECS:
            raise ValueError(f"unknown spell kind {kind!r}")
        if count < 0:
            raise ValueError(f"{kind} count must be non-negative")
        out[kind] = int(count)
    return out
