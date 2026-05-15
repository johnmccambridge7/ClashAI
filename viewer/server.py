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
    CoCEnv, DEPLOY_CELLS, N_DEPLOY_CELLS, WAIT_ACTION, TIME_COST_PER_TICK,
    action_to_cell,
)
from coc_env.generation import PRESET_LAYOUT_PROFILES


VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE = "default"
PROFILE_NAMES = [DEFAULT_PROFILE, *sorted(PRESET_LAYOUT_PROFILES)]
VIEWER_MAX_BUILDINGS = max(profile.max_buildings for profile in PRESET_LAYOUT_PROFILES.values())

app = Flask(__name__, static_folder=VIEWER_DIR, static_url_path="")

env = CoCEnv(max_buildings=VIEWER_MAX_BUILDINGS)
env.reset(seed=0)


@dataclass
class SessionState:
    last_action: int | None = None
    last_reward: float = 0.0
    total_reward: float = 0.0
    seed: int = 0
    profile: str = DEFAULT_PROFILE
    prev_score: float = 0.0
    actions: list[dict[str, object]] = field(default_factory=list)


session = SessionState()


def _action_label(action: int) -> str:
    if action == WAIT_ACTION:
        return "wait"
    if 0 <= action < N_DEPLOY_CELLS:
        x, y = DEPLOY_CELLS[action]
        return f"deploy @ ({x}, {y})"
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
    if 0 <= action < N_DEPLOY_CELLS:
        x, y = DEPLOY_CELLS[action]
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


def _state() -> dict[str, object]:
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
        "army_initial": env.army_size,
        "is_done": sim.is_done,
        "is_terminal": sim.is_terminal,
        "is_truncated": sim.is_truncated,
        "townhall_destroyed": sim.townhall_destroyed,
        "last_action": session.last_action,
        "last_reward": session.last_reward,
        "total_reward": session.total_reward,
        "actions": session.actions,
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
                "dps": b.spec.dps,
            }
            for b in sim.alive_buildings
        ],
        "rubble": [
            {
                "id": b.id, "kind": b.spec.kind,
                "x": b.x, "y": b.y, "size": b.spec.size,
            }
            for b in sim.buildings
            if not b.alive and b.spec.kind in {"townhall", "cannon", "storage"}
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
        "perimeter": DEPLOY_CELLS,
        "wait_action": WAIT_ACTION,
        "n_actions": N_DEPLOY_CELLS + 1,
        "action_mask": env.action_masks().tolist(),
        "engagements": sim.current_engagements(),
    }


def _advance_one_tick() -> float:
    sim = env.sim
    assert sim is not None
    if sim.is_done:
        return 0.0
    sim.tick()
    r = sim.score - session.prev_score - TIME_COST_PER_TICK
    session.prev_score = sim.score
    return float(r)


def _apply_action(action: int, source: str) -> bool:
    if action == WAIT_ACTION:
        session.last_action = WAIT_ACTION
        _record_action(action, source, True)
        return False
    if not (0 <= action < N_DEPLOY_CELLS):
        _record_action(action, source, False)
        return False
    cx, cy = action_to_cell(action)
    ok = (
        env.sim.deploy(cx + 0.5, cy + 0.5)
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
    options = None if profile == DEFAULT_PROFILE else {"profile": profile}
    env.reset(seed=seed, options=options)
    session.last_action = None
    session.last_reward = 0.0
    session.total_reward = 0.0
    session.seed = seed
    session.profile = profile
    session.prev_score = 0.0
    session.actions.clear()
    return jsonify(_state())


@app.post("/tick")
def tick():
    r = _advance_one_tick()
    session.last_reward = r
    session.total_reward += r
    return jsonify(_state())


@app.post("/deploy")
def deploy():
    action = _request_action()
    if action is None:
        return jsonify({"error": "integer action is required"}), 400
    if not (0 <= action <= WAIT_ACTION):
        return jsonify({"error": f"action {action} out of range"}), 400
    mask = env.action_masks()
    if not mask[action]:
        return jsonify({"error": f"action {action} is masked"}), 400
    _apply_action(action, "manual")
    return jsonify(_state())


@app.post("/agent_random")
def agent_random():
    mask = env.action_masks()
    valid = [i for i, ok in enumerate(mask) if ok]
    if not valid:
        return jsonify(_state())
    action = _random.choice(valid)
    _apply_action(action, "policy")
    return jsonify(_state())


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
    step_reward = 0.0
    for _ in range(DECISION_INTERVAL):
        if env.sim is None or env.sim.is_done:
            break
        step_reward += _advance_one_tick()
    session.last_reward = step_reward
    session.total_reward += step_reward
    return jsonify(_state())


@app.post("/step_random")
def step_random():
    mask = env.action_masks()
    valid = [i for i, ok in enumerate(mask) if ok]
    if not valid:
        return jsonify(_state())
    action = _random.choice(valid)
    _apply_action(action, "policy")
    step_reward = 0.0
    for _ in range(DECISION_INTERVAL):
        if env.sim is None or env.sim.is_done:
            break
        step_reward += _advance_one_tick()
    session.last_reward = step_reward
    session.total_reward += step_reward
    return jsonify(_state())


@app.get("/state")
def state():
    return jsonify(_state())


def main() -> None:
    port = int(os.environ.get("PORT", "5173"))
    print(f"Open http://127.0.0.1:{port}/ in your browser")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
