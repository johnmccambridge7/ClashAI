from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import random as _random
import sys
import threading
import zlib

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

# Support direct startup via `.venv/bin/python viewer/server.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coc_env.entities import GRID_SIZE, MAX_TICKS, DECISION_INTERVAL, TICK_SECONDS
from coc_env.env import (
    CoCEnv,
    DEPLOY_CELLS,
    DEPLOY_TROOP_KINDS,
    N_DEPLOY_ACTIONS,
    N_DEPLOY_CELLS,
    WAIT_ACTION,
    TIME_COST_PER_TICK,
    decode_deploy_action,
)
from coc_env.generation import PRESET_LAYOUT_PROFILES


VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_PROFILE = "default"
DEFAULT_PROFILE = "hard"
PROFILE_NAMES = [STATIC_PROFILE, *sorted(PRESET_LAYOUT_PROFILES)]
VIEWER_MAX_BUILDINGS = max(profile.max_buildings for profile in PRESET_LAYOUT_PROFILES.values())
VIEWER_ARMY_COMPOSITION = {"barbarian": 40, "wall_breaker": 10}
DEFAULT_ROLLOUT_COUNT = 8
MAX_ROLLOUT_COUNT = 16
ROLLOUT_FRAME_CHUNK_SIZE = 30
ROLLOUT_PROGRESS_TICKS = 30
TRUTHY_ENV = {"1", "true", "yes", "on"}
FALSY_ENV = {"0", "false", "no", "off"}

app = Flask(__name__, static_folder=VIEWER_DIR, static_url_path="")


def _make_viewer_env() -> CoCEnv:
    return CoCEnv(
        layout_profile=DEFAULT_PROFILE,
        max_buildings=VIEWER_MAX_BUILDINGS,
        army_composition=VIEWER_ARMY_COMPOSITION,
    )


env = _make_viewer_env()
env.reset(seed=0)


@dataclass
class SessionState:
    last_action: int | None = None
    last_reward: float = 0.0
    total_reward: float = 0.0
    seed: int = 0
    profile: str = DEFAULT_PROFILE
    prev_score: float = 0.0
    ticks_since_decision: int = 0
    actions: list[dict[str, object]] = field(default_factory=list)


session = SessionState()


@dataclass(frozen=True)
class RolloutBatchRequest:
    seed: int
    count: int
    profile: str
    agent_mode: str
    include_paths: bool


def _json_body() -> dict[str, object]:
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in TRUTHY_ENV:
        return True
    if normalized in FALSY_ENV:
        return False
    return default


def _reload_extra_files() -> list[str]:
    root = Path(VIEWER_DIR).parent
    watched: list[Path] = [
        Path(VIEWER_DIR) / "index.html",
        Path(VIEWER_DIR) / "server.py",
        root / "pyproject.toml",
    ]
    watched.extend((root / "coc_env").glob("*.py"))
    return [str(path) for path in watched if path.exists()]


def _action_label(action: int) -> str:
    if action == WAIT_ACTION:
        return "wait"
    if 0 <= action < N_DEPLOY_ACTIONS:
        troop_kind, x, y = decode_deploy_action(action)
        return f"deploy {troop_kind} @ ({x}, {y})"
    return f"action {action}"


def _record_action(action: int, source: str, applied: bool) -> None:
    sim = env.sim
    assert sim is not None
    entry: dict[str, object] = {
        "tick": sim.tick_count,
        "source": source,
        "action": action,
        "label": _action_label(action),
        "applied": applied,
    }
    if 0 <= action < N_DEPLOY_ACTIONS:
        troop_kind, x, y = decode_deploy_action(action)
        entry["troop_kind"] = troop_kind
        entry["x"] = x
        entry["y"] = y
    session.actions.append(entry)
    session.actions = session.actions[-80:]


def _path_direction(a: tuple[float, float], b: tuple[float, float]) -> tuple[int, int]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (
        0 if abs(dx) <= 1e-9 else (1 if dx > 0 else -1),
        0 if abs(dy) <= 1e-9 else (1 if dy > 0 else -1),
    )


def _simplify_path(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        if _path_direction(simplified[-1], points[i]) != _path_direction(points[i], points[i + 1]):
            simplified.append(points[i])
    simplified.append(points[-1])
    return simplified


_PATH_CACHE: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}


def _path_predictions() -> list[dict[str, object]]:
    sim = env.sim
    assert sim is not None
    terrain_v = sim.terrain_version
    # Drop stale cache entries from earlier terrain versions in one pass.
    if _PATH_CACHE and next(iter(_PATH_CACHE))[3] != terrain_v:
        _PATH_CACHE.clear()

    paths: list[dict[str, object]] = []
    for troop in sim.active_troops:
        target = sim.predicted_target_for(troop)
        if target is None:
            continue
        cell_x = max(0, min(GRID_SIZE - 1, int(troop.x)))
        cell_y = max(0, min(GRID_SIZE - 1, int(troop.y)))
        cache_key = (target.id, cell_x, cell_y, terrain_v)
        cached = _PATH_CACHE.get(cache_key)
        if cached is None:
            raw = sim.predicted_path_for(troop, max_steps=56)
            cached = _simplify_path(raw)
            _PATH_CACHE[cache_key] = cached
        if len(cached) < 2:
            continue
        # Splice the troop's live position onto the cached cell-derived tail so
        # the line starts from where the troop actually is right now.
        points: list[tuple[float, float]] = [(troop.x, troop.y), *cached[1:]]
        paths.append({
            "troop_id": troop.id,
            "target_id": target.id,
            "target_kind": target.spec.kind,
            "points": [{"x": x, "y": y} for x, y in points],
        })
    return paths


def _include_paths_requested(body: dict[str, object] | None = None) -> bool:
    source = body if body is not None else request.get_json(silent=True)
    raw = request.args.get("include_paths", False) if source is None else source.get("include_paths", False)
    return raw is True or raw == 1 or (isinstance(raw, str) and raw.lower() in {"1", "true", "yes"})


def _rollout_batch_request(body: dict[str, object]) -> RolloutBatchRequest:
    try:
        seed = int(body.get("seed", 0))
        count = int(body.get("count", DEFAULT_ROLLOUT_COUNT))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed and count must be integers") from exc
    if not (1 <= count <= MAX_ROLLOUT_COUNT):
        raise ValueError(f"count must be between 1 and {MAX_ROLLOUT_COUNT}")

    profile = str(body.get("profile", session.profile))
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile {profile!r} is not available")

    agent_mode = str(body.get("agent_mode", "random"))
    if agent_mode not in {"off", "random"}:
        raise ValueError(f"agent_mode {agent_mode!r} is not available")

    return RolloutBatchRequest(
        seed=seed,
        count=count,
        profile=profile,
        agent_mode=agent_mode,
        include_paths=_include_paths_requested(body),
    )


def _reset_episode(seed: int, profile: str) -> None:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile {profile!r} is not available")
    options = None if profile == STATIC_PROFILE else {"profile": profile}
    env.reset(seed=seed, options=options)
    _random.seed(seed)
    session.last_action = None
    session.last_reward = 0.0
    session.total_reward = 0.0
    session.seed = seed
    session.profile = profile
    session.prev_score = 0.0
    session.ticks_since_decision = 0
    session.actions.clear()
    _PATH_CACHE.clear()


def _state(
    include_paths: bool = False,
    event_stream: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sim = env.sim
    assert sim is not None
    state: dict[str, object] = {
        "tick": sim.tick_count,
        "max_ticks": MAX_TICKS,
        "tick_seconds": TICK_SECONDS,
        "decision_interval": DECISION_INTERVAL,
        "grid_size": GRID_SIZE,
        "damage_pct": sim.damage_pct,
        "stars": sim.stars,
        "score": sim.score,
        "army_remaining": sim.army_remaining,
        "army_remaining_by_kind": dict(sim.army_remaining_by_kind),
        "army_initial": env.army_size,
        "is_done": sim.is_done,
        "is_terminal": sim.is_terminal,
        "is_truncated": sim.is_truncated,
        "townhall_destroyed": sim.townhall_destroyed,
        "last_action": session.last_action,
        "last_reward": session.last_reward,
        "total_reward": session.total_reward,
        "actions": [dict(action) for action in session.actions],
        "ticks_since_decision": session.ticks_since_decision,
        "seed": session.seed,
        "profile": session.profile,
        "available_profiles": PROFILE_NAMES,
        # Only alive buildings/troops are sent for visualization; observation slots
        # (including dead ones) live behind the RL Gym API, not this JSON.
        "buildings": [
            {
                "id": b.id, "kind": b.spec.kind,
                "x": b.x, "y": b.y, "size": b.spec.size,
                "hp": b.hp, "max_hp": b.spec.hp,
                "is_defense": b.spec.is_defense,
                "blocks_movement": b.spec.blocks_movement,
                "counts_for_score": b.spec.counts_for_score,
                "attack_range": b.spec.attack_range,
                "min_attack_range": b.spec.min_attack_range,
                "splash_radius": b.spec.splash_radius,
                "cooldown_remaining": b.cooldown_remaining,
                "is_trap": b.spec.is_trap,
                "triggered": b.triggered,
                "dps": b.spec.dps,
            }
            for b in sim.visible_buildings
        ],
        "rubble": [
            {
                "id": b.id, "kind": b.spec.kind,
                "x": b.x, "y": b.y, "size": b.spec.size,
            }
            for b in sim.buildings
            if not b.alive and b.revealed and b.spec.kind in {
                "townhall", "cannon", "wizard_tower", "mortar", "storage"
            }
        ],
        "troops": [
            {
                "id": t.id, "kind": t.spec.kind,
                "x": t.x, "y": t.y,
                "hp": t.hp, "max_hp": t.spec.hp,
            }
            for t in sim.active_troops
        ],
        "engagements": sim.current_engagements(),
        "event_stream": event_stream or [],
        "path_predictions": _path_predictions() if include_paths else [],
    }
    return state


def _advance_one_tick() -> tuple[float, list[dict[str, object]]]:
    sim = env.sim
    assert sim is not None
    if sim.is_done:
        return 0.0, []
    sim.tick()
    session.ticks_since_decision += 1
    r = sim.score - session.prev_score - TIME_COST_PER_TICK
    session.prev_score = sim.score
    return float(r), [dict(e) for e in sim.visual_events]


def _random_policy_action() -> int | None:
    sim = env.sim
    if sim is None or sim.is_done:
        return None
    troop_kind_indices = [
        index
        for index, troop_kind in enumerate(DEPLOY_TROOP_KINDS)
        if sim.army_remaining_by_kind.get(troop_kind, 0) > 0
    ]
    if not troop_kind_indices:
        return WAIT_ACTION
    deployable_cells = [
        cell_index
        for cell_index, (x, y) in enumerate(DEPLOY_CELLS)
        if env._can_deploy_cell(x, y)
    ]
    if not deployable_cells:
        return WAIT_ACTION
    return _random.choice(troop_kind_indices) * N_DEPLOY_CELLS + _random.choice(deployable_cells)


def _apply_random_policy_action() -> bool:
    action = _random_policy_action()
    if action is None:
        return False
    applied = _apply_action(action, "policy")
    session.ticks_since_decision = 0
    return applied


def _advance_ticks(count: int, agent_mode: str = "off") -> tuple[float, list[dict[str, object]]]:
    sim = env.sim
    assert sim is not None
    step_reward = 0.0
    event_stream: list[dict[str, object]] = []
    for _ in range(max(0, count)):
        if sim.is_done:
            break
        if (
            agent_mode == "random"
            and sim.army_remaining > 0
            and session.ticks_since_decision >= DECISION_INTERVAL
        ):
            _apply_random_policy_action()
        reward, events = _advance_one_tick()
        step_reward += reward
        event_stream.extend(events)
    return step_reward, event_stream


def _apply_action(action: int, source: str) -> bool:
    if action == WAIT_ACTION:
        session.last_action = WAIT_ACTION
        _record_action(action, source, True)
        return False
    if not (0 <= action < N_DEPLOY_ACTIONS):
        _record_action(action, source, False)
        return False
    troop_kind, cx, cy = decode_deploy_action(action)
    ok = (
        env.sim.deploy(cx + 0.5, cy + 0.5, troop_kind=troop_kind)
        if env.sim is not None and env._can_deploy_cell(cx, cy)
        else False
    )
    if ok:
        session.last_action = action
    _record_action(action, source, ok)
    return ok


def _action_signature(actions: list[dict[str, object]]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            action.get("tick"),
            action.get("source"),
            action.get("action"),
            action.get("applied"),
            action.get("label"),
        )
        for action in actions
    )


def _compact_frame(
    frame: dict[str, object],
    include_actions: bool,
) -> dict[str, object]:
    compact: dict[str, object] = {
        "tick": frame["tick"],
        "damage_pct": frame["damage_pct"],
        "stars": frame["stars"],
        "score": frame["score"],
        "army_remaining": frame["army_remaining"],
        "army_remaining_by_kind": frame["army_remaining_by_kind"],
        "is_done": frame["is_done"],
        "is_terminal": frame["is_terminal"],
        "is_truncated": frame["is_truncated"],
        "townhall_destroyed": frame["townhall_destroyed"],
        "last_action": frame["last_action"],
        "last_reward": frame["last_reward"],
        "total_reward": frame["total_reward"],
        "ticks_since_decision": frame["ticks_since_decision"],
        "buildings": [
            [
                building["id"],
                building["hp"],
                building["cooldown_remaining"],
                1 if building.get("triggered") else 0,
            ]
            for building in frame["buildings"]  # type: ignore[index]
        ],
        "rubble_ids": [building["id"] for building in frame["rubble"]],  # type: ignore[index]
        "troops": [
            [
                troop["id"],
                troop["kind"],
                round(float(troop["x"]), 3),
                round(float(troop["y"]), 3),
                troop["hp"],
                troop["max_hp"],
            ]
            for troop in frame["troops"]  # type: ignore[index]
        ],
        "engagements": frame["engagements"],
        "event_stream": frame["event_stream"],
        "path_predictions": frame["path_predictions"],
    }
    if include_actions:
        compact["actions"] = frame["actions"]
    return compact


class RolloutRunner:
    def __init__(self) -> None:
        self.env = _make_viewer_env()
        self.session = SessionState()
        self.rng = _random.Random()
        self.path_cache: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}

    def reset_episode(self, seed: int, profile: str) -> None:
        if profile not in PROFILE_NAMES:
            raise ValueError(f"profile {profile!r} is not available")
        options = None if profile == STATIC_PROFILE else {"profile": profile}
        self.env.reset(seed=seed, options=options)
        self.rng.seed(seed)
        self.session.last_action = None
        self.session.last_reward = 0.0
        self.session.total_reward = 0.0
        self.session.seed = seed
        self.session.profile = profile
        self.session.prev_score = 0.0
        self.session.ticks_since_decision = 0
        self.session.actions.clear()
        self.path_cache.clear()

    def record_action(self, action: int, source: str, applied: bool) -> None:
        sim = self.env.sim
        assert sim is not None
        entry: dict[str, object] = {
            "tick": sim.tick_count,
            "source": source,
            "action": action,
            "label": _action_label(action),
            "applied": applied,
        }
        if 0 <= action < N_DEPLOY_ACTIONS:
            troop_kind, x, y = decode_deploy_action(action)
            entry["troop_kind"] = troop_kind
            entry["x"] = x
            entry["y"] = y
        self.session.actions.append(entry)
        self.session.actions = self.session.actions[-80:]

    def path_predictions(self) -> list[dict[str, object]]:
        sim = self.env.sim
        assert sim is not None
        terrain_v = sim.terrain_version
        if self.path_cache and next(iter(self.path_cache))[3] != terrain_v:
            self.path_cache.clear()

        paths: list[dict[str, object]] = []
        for troop in sim.active_troops:
            target = sim.predicted_target_for(troop)
            if target is None:
                continue
            cell_x = max(0, min(GRID_SIZE - 1, int(troop.x)))
            cell_y = max(0, min(GRID_SIZE - 1, int(troop.y)))
            cache_key = (target.id, cell_x, cell_y, terrain_v)
            cached = self.path_cache.get(cache_key)
            if cached is None:
                raw = sim.predicted_path_for(troop, max_steps=56)
                cached = _simplify_path(raw)
                self.path_cache[cache_key] = cached
            if len(cached) < 2:
                continue
            points: list[tuple[float, float]] = [(troop.x, troop.y), *cached[1:]]
            paths.append({
                "troop_id": troop.id,
                "target_id": target.id,
                "target_kind": target.spec.kind,
                "points": [{"x": x, "y": y} for x, y in points],
            })
        return paths

    def static_payload(self) -> dict[str, object]:
        sim = self.env.sim
        assert sim is not None
        return {
            "max_ticks": MAX_TICKS,
            "tick_seconds": TICK_SECONDS,
            "decision_interval": DECISION_INTERVAL,
            "grid_size": GRID_SIZE,
            "army_initial": self.env.army_size,
            "seed": self.session.seed,
            "profile": self.session.profile,
            "available_profiles": PROFILE_NAMES,
            "buildings": [
                {
                    "id": b.id,
                    "kind": b.spec.kind,
                    "x": b.x,
                    "y": b.y,
                    "size": b.spec.size,
                    "max_hp": b.spec.hp,
                    "is_defense": b.spec.is_defense,
                    "blocks_movement": b.spec.blocks_movement,
                    "counts_for_score": b.spec.counts_for_score,
                    "attack_range": b.spec.attack_range,
                    "min_attack_range": b.spec.min_attack_range,
                    "splash_radius": b.spec.splash_radius,
                    "is_trap": b.spec.is_trap,
                    "dps": b.spec.dps,
                }
                for b in sim.buildings
            ],
        }

    def state(
        self,
        include_paths: bool = False,
        event_stream: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        sim = self.env.sim
        assert sim is not None
        return {
            "tick": sim.tick_count,
            "max_ticks": MAX_TICKS,
            "tick_seconds": TICK_SECONDS,
            "decision_interval": DECISION_INTERVAL,
            "grid_size": GRID_SIZE,
            "damage_pct": sim.damage_pct,
            "stars": sim.stars,
            "score": sim.score,
            "army_remaining": sim.army_remaining,
            "army_remaining_by_kind": dict(sim.army_remaining_by_kind),
            "army_initial": self.env.army_size,
            "is_done": sim.is_done,
            "is_terminal": sim.is_terminal,
            "is_truncated": sim.is_truncated,
            "townhall_destroyed": sim.townhall_destroyed,
            "last_action": self.session.last_action,
            "last_reward": self.session.last_reward,
            "total_reward": self.session.total_reward,
            "actions": [dict(action) for action in self.session.actions],
            "ticks_since_decision": self.session.ticks_since_decision,
            "seed": self.session.seed,
            "profile": self.session.profile,
            "available_profiles": PROFILE_NAMES,
            "buildings": [
                {
                    "id": b.id, "kind": b.spec.kind,
                    "x": b.x, "y": b.y, "size": b.spec.size,
                    "hp": b.hp, "max_hp": b.spec.hp,
                    "is_defense": b.spec.is_defense,
                    "blocks_movement": b.spec.blocks_movement,
                    "counts_for_score": b.spec.counts_for_score,
                    "attack_range": b.spec.attack_range,
                    "min_attack_range": b.spec.min_attack_range,
                    "splash_radius": b.spec.splash_radius,
                    "cooldown_remaining": b.cooldown_remaining,
                    "is_trap": b.spec.is_trap,
                    "triggered": b.triggered,
                    "dps": b.spec.dps,
                }
                for b in sim.visible_buildings
            ],
            "rubble": [
                {
                    "id": b.id, "kind": b.spec.kind,
                    "x": b.x, "y": b.y, "size": b.spec.size,
                }
                for b in sim.buildings
                if not b.alive and b.revealed and b.spec.kind in {
                    "townhall", "cannon", "wizard_tower", "mortar", "storage"
                }
            ],
            "troops": [
                {
                    "id": t.id, "kind": t.spec.kind,
                    "x": t.x, "y": t.y,
                    "hp": t.hp, "max_hp": t.spec.hp,
                }
                for t in sim.active_troops
            ],
            "engagements": sim.current_engagements(),
            "event_stream": event_stream or [],
            "path_predictions": self.path_predictions() if include_paths else [],
        }

    def advance_one_tick(self) -> tuple[float, list[dict[str, object]]]:
        sim = self.env.sim
        assert sim is not None
        if sim.is_done:
            return 0.0, []
        sim.tick()
        self.session.ticks_since_decision += 1
        reward = sim.score - self.session.prev_score - TIME_COST_PER_TICK
        self.session.prev_score = sim.score
        return float(reward), [dict(e) for e in sim.visual_events]

    def random_policy_action(self) -> int | None:
        sim = self.env.sim
        if sim is None or sim.is_done:
            return None
        troop_kind_indices = [
            index
            for index, troop_kind in enumerate(DEPLOY_TROOP_KINDS)
            if sim.army_remaining_by_kind.get(troop_kind, 0) > 0
        ]
        if not troop_kind_indices:
            return WAIT_ACTION
        deployable_cells = [
            cell_index
            for cell_index, (x, y) in enumerate(DEPLOY_CELLS)
            if self.env._can_deploy_cell(x, y)
        ]
        if not deployable_cells:
            return WAIT_ACTION
        return self.rng.choice(troop_kind_indices) * N_DEPLOY_CELLS + self.rng.choice(deployable_cells)

    def apply_action(self, action: int, source: str) -> bool:
        if action == WAIT_ACTION:
            self.session.last_action = WAIT_ACTION
            self.record_action(action, source, True)
            return False
        if not (0 <= action < N_DEPLOY_ACTIONS):
            self.record_action(action, source, False)
            return False
        troop_kind, cx, cy = decode_deploy_action(action)
        ok = (
            self.env.sim.deploy(cx + 0.5, cy + 0.5, troop_kind=troop_kind)
            if self.env.sim is not None and self.env._can_deploy_cell(cx, cy)
            else False
        )
        if ok:
            self.session.last_action = action
        self.record_action(action, source, ok)
        return ok

    def apply_random_policy_action(self) -> bool:
        action = self.random_policy_action()
        if action is None:
            return False
        applied = self.apply_action(action, "policy")
        self.session.ticks_since_decision = 0
        return applied

    def advance_ticks(self, count: int, agent_mode: str = "off") -> tuple[float, list[dict[str, object]]]:
        sim = self.env.sim
        assert sim is not None
        step_reward = 0.0
        event_stream: list[dict[str, object]] = []
        for _ in range(max(0, count)):
            if sim.is_done:
                break
            if (
                agent_mode == "random"
                and sim.army_remaining > 0
                and self.session.ticks_since_decision >= DECISION_INTERVAL
            ):
                self.apply_random_policy_action()
            reward, events = self.advance_one_tick()
            step_reward += reward
            event_stream.extend(events)
        return step_reward, event_stream

    def rollout_payload(
        self,
        seed: int,
        profile: str,
        agent_mode: str,
        include_paths: bool,
    ) -> dict[str, object]:
        self.reset_episode(seed, profile)
        frames: list[dict[str, object]] = [self.state(include_paths)]
        sim = self.env.sim
        assert sim is not None
        while not sim.is_done:
            reward, events = self.advance_ticks(1, agent_mode)
            self.session.last_reward = reward
            self.session.total_reward += reward
            frames.append(self.state(include_paths, events))

        return {
            "seed": self.session.seed,
            "profile": self.session.profile,
            "agent_mode": agent_mode,
            "frame_count": len(frames),
            "available_profiles": PROFILE_NAMES,
            "frames": frames,
        }

    def stream_messages(
        self,
        offset: int,
        options: RolloutBatchRequest,
    ) -> Iterator[dict[str, object]]:
        self.reset_episode(options.seed + offset, options.profile)
        sim = self.env.sim
        assert sim is not None

        frame_index = 0
        chunk_start = 0
        chunk: list[dict[str, object]] = []
        last_progress_tick = sim.tick_count
        last_actions_key: tuple[tuple[object, ...], ...] | None = None

        def append_frame(frame: dict[str, object]) -> None:
            nonlocal last_actions_key
            actions = frame["actions"]  # type: ignore[assignment]
            actions_key = _action_signature(actions)  # type: ignore[arg-type]
            chunk.append(_compact_frame(frame, actions_key != last_actions_key))
            last_actions_key = actions_key

        append_frame(self.state(options.include_paths))
        yield {
            "type": "frames",
            "index": offset,
            "count": options.count,
            "start": chunk_start,
            "static": self.static_payload(),
            "frames": chunk,
        }
        frame_index += len(chunk)
        chunk = []
        chunk_start = frame_index

        while not sim.is_done:
            reward, events = self.advance_ticks(1, options.agent_mode)
            self.session.last_reward = reward
            self.session.total_reward += reward
            append_frame(self.state(options.include_paths, events))
            if len(chunk) >= ROLLOUT_FRAME_CHUNK_SIZE:
                yield {
                    "type": "frames",
                    "index": offset,
                    "count": options.count,
                    "start": chunk_start,
                    "frames": chunk,
                }
                frame_index += len(chunk)
                chunk = []
                chunk_start = frame_index
            if (
                not sim.is_done
                and sim.tick_count - last_progress_tick >= ROLLOUT_PROGRESS_TICKS
            ):
                last_progress_tick = sim.tick_count
                yield {
                    "type": "progress",
                    "index": offset,
                    "completed": offset,
                    "count": options.count,
                    "tick": sim.tick_count,
                    "max_ticks": MAX_TICKS,
                    "message": (
                        f"computing rollout {offset + 1} / {options.count} "
                        f"(tick {sim.tick_count} / {MAX_TICKS})"
                    ),
                }

        if chunk:
            yield {
                "type": "frames",
                "index": offset,
                "count": options.count,
                "start": chunk_start,
                "frames": chunk,
            }
            frame_index += len(chunk)

        yield {
            "type": "rollout_done",
            "index": offset,
            "completed": offset + 1,
            "count": options.count,
            "frame_count": frame_index,
        }


def _rollout_stream_messages(
    offset: int,
    options: RolloutBatchRequest,
) -> Iterator[dict[str, object]]:
    return RolloutRunner().stream_messages(offset, options)


def _rollout_worker(
    offset: int,
    options: RolloutBatchRequest,
    messages: queue.Queue[dict[str, object]],
) -> None:
    try:
        for message in _rollout_stream_messages(offset, options):
            messages.put(message)
    except Exception as exc:  # pragma: no cover - surfaced through stream.
        messages.put({
            "type": "error",
            "index": offset,
            "completed": 0,
            "count": options.count,
            "error": str(exc),
        })
    finally:
        messages.put({"type": "worker_done", "index": offset})


def _parallel_rollout_messages(options: RolloutBatchRequest) -> Iterator[dict[str, object]]:
    messages: queue.Queue[dict[str, object]] = queue.Queue()
    threads = [
        threading.Thread(
            target=_rollout_worker,
            args=(offset, options, messages),
            daemon=True,
        )
        for offset in range(options.count)
    ]
    for thread in threads:
        thread.start()

    done = 0
    completed = 0
    while done < options.count:
        message = messages.get()
        if message.get("type") == "worker_done":
            done += 1
            continue
        if message.get("type") == "rollout_done":
            completed += 1
            message["completed"] = completed
        yield message


@app.route("/")
def index():
    return send_from_directory(VIEWER_DIR, "index.html")


@app.post("/reset")
def reset():
    body = _json_body()
    seed = int(body.get("seed", 0))
    profile = str(body.get("profile", session.profile))
    try:
        _reset_episode(seed, profile)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_state(_include_paths_requested(body)))


@app.post("/rollout")
def rollout():
    body = _json_body()
    try:
        seed = int(body.get("seed", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "seed must be an integer"}), 400
    profile = str(body.get("profile", session.profile))
    agent_mode = str(body.get("agent_mode", "random"))
    if agent_mode not in {"off", "random"}:
        return jsonify({"error": f"agent_mode {agent_mode!r} is not available"}), 400
    if profile not in PROFILE_NAMES:
        return jsonify({"error": f"profile {profile!r} is not available"}), 400
    return jsonify(_rollout_payload(seed, profile, agent_mode, _include_paths_requested(body)))


def _rollout_payload(
    seed: int,
    profile: str,
    agent_mode: str,
    include_paths: bool,
) -> dict[str, object]:
    return RolloutRunner().rollout_payload(seed, profile, agent_mode, include_paths)


def _gzip_chunks(chunks: Iterator[str]) -> Iterator[bytes]:
    compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    for chunk in chunks:
        data = compressor.compress(chunk.encode("utf-8"))
        data += compressor.flush(zlib.Z_SYNC_FLUSH)
        if data:
            yield data
    tail = compressor.flush(zlib.Z_FINISH)
    if tail:
        yield tail


@app.post("/rollouts")
def rollouts():
    try:
        options = _rollout_batch_request(_json_body())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "seed": options.seed,
        "count": options.count,
        "profile": options.profile,
        "agent_mode": options.agent_mode,
        "available_profiles": PROFILE_NAMES,
        "rollouts": [
            _rollout_payload(
                options.seed + offset,
                options.profile,
                options.agent_mode,
                options.include_paths,
            )
            for offset in range(options.count)
        ],
    })


@app.post("/rollouts/stream")
def rollouts_stream():
    try:
        options = _rollout_batch_request(_json_body())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    def line(payload: dict[str, object]) -> str:
        return json.dumps(payload, separators=(",", ":")) + "\n"

    @stream_with_context
    def generate() -> Iterator[str]:
        yield line({
            "type": "start",
            "seed": options.seed,
            "count": options.count,
            "profile": options.profile,
            "agent_mode": options.agent_mode,
            "available_profiles": PROFILE_NAMES,
        })
        for message in _parallel_rollout_messages(options):
            yield line(message)
            if message.get("type") == "error":
                return
        yield line({
            "type": "done",
            "completed": options.count,
            "count": options.count,
        })

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    raw_chunks = generate()
    chunks: Iterator[str] | Iterator[bytes] = raw_chunks
    if "gzip" in request.headers.get("Accept-Encoding", "").lower():
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
        chunks = _gzip_chunks(raw_chunks)

    return Response(
        chunks,
        mimetype="application/x-ndjson",
        headers=headers,
    )


@app.get("/state")
def state():
    return jsonify(_state(_include_paths_requested()))


def main() -> None:
    port = int(os.environ.get("PORT", "5173"))
    hot_reload = _env_flag("VIEWER_RELOAD", True)
    print(f"Open http://127.0.0.1:{port}/ in your browser")
    print(f"Hot reload: {'on' if hot_reload else 'off'}")
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=hot_reload,
        extra_files=_reload_extra_files() if hot_reload else None,
    )


if __name__ == "__main__":
    main()
