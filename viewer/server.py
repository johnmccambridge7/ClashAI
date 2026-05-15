from __future__ import annotations
from dataclasses import dataclass, field
import os
import random as _random
import sys

from flask import Flask, jsonify, request, send_from_directory

# Support direct startup via `.venv/bin/python viewer/server.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coc_env.entities import GRID_SIZE, MAX_TICKS, DECISION_INTERVAL, TICK_SECONDS
from coc_env.env import (
    CoCEnv, DEPLOY_CELLS, DEPLOY_TROOP_KINDS, N_DEPLOY_ACTIONS, N_DEPLOY_CELLS,
    WAIT_ACTION, TIME_COST_PER_TICK, decode_deploy_action,
)
from coc_env.generation import PRESET_LAYOUT_PROFILES


VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_PROFILE = "default"
DEFAULT_PROFILE = "hard"
PROFILE_NAMES = [STATIC_PROFILE, *sorted(PRESET_LAYOUT_PROFILES)]
VIEWER_MAX_BUILDINGS = max(profile.max_buildings for profile in PRESET_LAYOUT_PROFILES.values())
VIEWER_ARMY_COMPOSITION = {"barbarian": 40, "wall_breaker": 10}

app = Flask(__name__, static_folder=VIEWER_DIR, static_url_path="")

env = CoCEnv(
    layout_profile=DEFAULT_PROFILE,
    max_buildings=VIEWER_MAX_BUILDINGS,
    army_composition=VIEWER_ARMY_COMPOSITION,
)
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


def _request_action() -> int | None:
    body = request.get_json(silent=True) or {}
    try:
        return int(body["action"])
    except (KeyError, TypeError, ValueError):
        return None


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


def _state(
    include_paths: bool = False,
    event_stream: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sim = env.sim
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
        "army_initial": env.army_size,
        "is_done": sim.is_done,
        "is_terminal": sim.is_terminal,
        "is_truncated": sim.is_truncated,
        "townhall_destroyed": sim.townhall_destroyed,
        "last_action": session.last_action,
        "last_reward": session.last_reward,
        "total_reward": session.total_reward,
        "actions": session.actions,
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
        "deploy_cells": DEPLOY_CELLS,
        "deploy_troop_kinds": DEPLOY_TROOP_KINDS,
        "wait_action": WAIT_ACTION,
        "n_actions": N_DEPLOY_ACTIONS + 1,
        "action_mask": env.action_masks().tolist(),
        "engagements": sim.current_engagements(),
        "event_stream": event_stream or [],
        "path_predictions": _path_predictions() if include_paths else [],
    }


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
    mask = env.action_masks()
    valid = [i for i, ok in enumerate(mask) if ok]
    if not valid:
        return None
    return _random.choice(valid)


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


@app.route("/")
def index():
    return send_from_directory(VIEWER_DIR, "index.html")


@app.post("/reset")
def reset():
    body = request.get_json(silent=True) or {}
    seed = int(body.get("seed", 0))
    profile = str(body.get("profile", session.profile))
    if profile not in PROFILE_NAMES:
        return jsonify({"error": f"profile {profile!r} is not available"}), 400
    options = None if profile == STATIC_PROFILE else {"profile": profile}
    env.reset(seed=seed, options=options)
    session.last_action = None
    session.last_reward = 0.0
    session.total_reward = 0.0
    session.seed = seed
    session.profile = profile
    session.prev_score = 0.0
    session.ticks_since_decision = 0
    session.actions.clear()
    return jsonify(_state(_include_paths_requested(body)))


@app.post("/tick")
def tick():
    r, events = _advance_ticks(1)
    session.last_reward = r
    session.total_reward += r
    return jsonify(_state(_include_paths_requested(), events))


@app.post("/deploy")
def deploy():
    body = request.get_json(silent=True) or {}
    action = _request_action()
    if action is None:
        return jsonify({"error": "integer action is required"}), 400
    if not (0 <= action <= WAIT_ACTION):
        return jsonify({"error": f"action {action} out of range"}), 400
    mask = env.action_masks()
    if not mask[action]:
        return jsonify({"error": f"action {action} is masked"}), 400
    _apply_action(action, "manual")
    return jsonify(_state(_include_paths_requested(body)))


@app.post("/agent_random")
def agent_random():
    body = request.get_json(silent=True) or {}
    _apply_random_policy_action()
    return jsonify(_state(_include_paths_requested(body)))


@app.post("/step")
def step():
    action = _request_action()
    if action is None:
        return jsonify({"error": "integer action is required"}), 400
    if not (0 <= action <= WAIT_ACTION):
        return jsonify({"error": f"action {action} out of range"}), 400
    mask = env.action_masks()
    if not mask[action]:
        return jsonify({"error": f"action {action} is masked"}), 400
    _apply_action(action, "manual")
    step_reward, events = _advance_ticks(DECISION_INTERVAL)
    session.last_reward = step_reward
    session.total_reward += step_reward
    return jsonify(_state(_include_paths_requested(), events))


@app.post("/step_random")
def step_random():
    _apply_random_policy_action()
    step_reward, events = _advance_ticks(DECISION_INTERVAL)
    session.last_reward = step_reward
    session.total_reward += step_reward
    return jsonify(_state(_include_paths_requested(), events))


@app.post("/advance")
def advance():
    body = request.get_json(silent=True) or {}
    try:
        ticks = int(body.get("ticks", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "ticks must be an integer"}), 400
    if ticks < 0:
        return jsonify({"error": "ticks must be non-negative"}), 400
    agent_mode = str(body.get("agent_mode", "off"))
    if agent_mode not in {"off", "random"}:
        return jsonify({"error": f"agent_mode {agent_mode!r} is not available"}), 400
    step_reward, events = _advance_ticks(ticks, agent_mode)
    session.last_reward = step_reward
    session.total_reward += step_reward
    return jsonify(_state(_include_paths_requested(body), events))


@app.get("/state")
def state():
    return jsonify(_state(_include_paths_requested()))


def main() -> None:
    port = int(os.environ.get("PORT", "5173"))
    print(f"Open http://127.0.0.1:{port}/ in your browser")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
