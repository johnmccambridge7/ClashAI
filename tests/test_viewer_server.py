from __future__ import annotations

import json

from coc_env.entities import BUILDING_SPECS, Building
from coc_env.simulator import Simulator
import viewer.server as server


def test_advance_ticks_accumulates_transient_events_across_batch() -> None:
    sim = Simulator([Building(0, BUILDING_SPECS["wizard_tower"], 10, 10)], army_size=1)
    assert sim.deploy(15.0, 11.5)

    original_sim = server.env.sim
    original_prev_score = server.session.prev_score
    original_ticks_since_decision = server.session.ticks_since_decision
    try:
        server.env.sim = sim
        server.session.prev_score = sim.score
        server.session.ticks_since_decision = 0

        _, events = server._advance_ticks(3)

        assert sim.tick_count == 3
        assert sim.visual_events == []
        assert any(event["kind"] == "defense_fire" and event["tick"] == 1 for event in events)
        assert any(event["kind"] == "impact" and event["tick"] == 1 for event in events)
    finally:
        server.env.sim = original_sim
        server.session.prev_score = original_prev_score
        server.session.ticks_since_decision = original_ticks_since_decision


def test_rollout_returns_precomputed_frame_timeline() -> None:
    client = server.app.test_client()

    response = client.post("/rollout", json={
        "seed": 123,
        "profile": server.STATIC_PROFILE,
        "agent_mode": "random",
        "include_paths": False,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    frames = payload["frames"]
    assert len(frames) == payload["frame_count"]
    assert frames[0]["tick"] == 0
    assert frames[-1]["is_done"]
    assert all("action_mask" not in frame for frame in frames)
    assert all("deploy_cells" not in frame for frame in frames)
    assert any(frame["event_stream"] for frame in frames)


def test_rollouts_returns_eight_precomputed_timelines(monkeypatch) -> None:
    client = server.app.test_client()
    calls: list[tuple[int, str, str, bool]] = []

    def fake_rollout_payload(
        seed: int,
        profile: str,
        agent_mode: str,
        include_paths: bool,
    ) -> dict[str, object]:
        calls.append((seed, profile, agent_mode, include_paths))
        return {
            "seed": seed,
            "profile": profile,
            "agent_mode": agent_mode,
            "frame_count": 2,
            "available_profiles": server.PROFILE_NAMES,
            "frames": [{"tick": 0, "is_done": False}, {"tick": 1, "is_done": True}],
        }

    monkeypatch.setattr(server, "_rollout_payload", fake_rollout_payload)

    response = client.post("/rollouts", json={
        "seed": 200,
        "count": 8,
        "profile": server.STATIC_PROFILE,
        "agent_mode": "random",
        "include_paths": False,
    })

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["count"] == 8
    assert len(payload["rollouts"]) == 8
    assert [rollout["seed"] for rollout in payload["rollouts"]] == list(range(200, 208))
    assert calls == [
        (seed, server.STATIC_PROFILE, "random", False)
        for seed in range(200, 208)
    ]
    assert all(rollout["frames"][0]["tick"] == 0 for rollout in payload["rollouts"])
    assert all(rollout["frames"][-1]["is_done"] for rollout in payload["rollouts"])


def test_rollouts_stream_returns_incremental_ndjson(monkeypatch) -> None:
    client = server.app.test_client()
    calls: list[tuple[int, int, int, str, str, bool]] = []

    def fake_rollout_stream_messages(
        offset: int,
        options: server.RolloutBatchRequest,
    ):
        calls.append((
            offset,
            options.seed,
            options.count,
            options.profile,
            options.agent_mode,
            options.include_paths,
        ))
        yield {
            "type": "frames",
            "index": offset,
            "count": options.count,
            "start": 0,
            "static": {
                "max_ticks": server.MAX_TICKS,
                "tick_seconds": server.TICK_SECONDS,
                "decision_interval": server.DECISION_INTERVAL,
                "grid_size": server.GRID_SIZE,
                "army_initial": 1,
                "seed": options.seed + offset,
                "profile": options.profile,
                "available_profiles": server.PROFILE_NAMES,
                "buildings": [],
            },
            "frames": [{
                "tick": 0,
                "damage_pct": 0.0,
                "stars": 0,
                "score": 0.0,
                "army_remaining": 0,
                "army_remaining_by_kind": {},
                "is_done": True,
                "is_terminal": True,
                "is_truncated": False,
                "townhall_destroyed": False,
                "last_action": None,
                "last_reward": 0.0,
                "total_reward": 0.0,
                "ticks_since_decision": 0,
                "buildings": [],
                "rubble_ids": [],
                "troops": [],
                "engagements": [],
                "event_stream": [],
                "path_predictions": [],
                "actions": [],
            }],
        }
        yield {
            "type": "progress",
            "index": offset,
            "completed": offset,
            "count": options.count,
            "tick": 30,
            "max_ticks": server.MAX_TICKS,
            "message": f"computing rollout {offset + 1} / {options.count}",
        }
        yield {
            "type": "rollout_done",
            "index": offset,
            "completed": offset + 1,
            "count": options.count,
            "frame_count": 1,
        }

    monkeypatch.setattr(server, "_rollout_stream_messages", fake_rollout_stream_messages)

    response = client.post("/rollouts/stream", json={
        "seed": 300,
        "count": 2,
        "profile": server.STATIC_PROFILE,
        "agent_mode": "random",
        "include_paths": False,
    })

    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    messages = [
        json.loads(line)
        for line in response.data.decode("utf-8").splitlines()
    ]
    assert messages[0]["type"] == "start"
    assert messages[-1]["type"] == "done"
    frame_messages = [message for message in messages if message["type"] == "frames"]
    assert sorted(message["static"]["seed"] for message in frame_messages) == [300, 301]
    assert all(message["frames"][0]["rubble_ids"] == [] for message in frame_messages)
    assert any(
        message["type"] == "progress" and message.get("tick") == 30
        for message in messages
    )
    assert sorted(calls) == [
        (0, 300, 2, server.STATIC_PROFILE, "random", False),
        (1, 300, 2, server.STATIC_PROFILE, "random", False),
    ]


def test_manual_viewer_step_routes_are_not_exposed() -> None:
    client = server.app.test_client()

    for path in ["/tick", "/deploy", "/agent_random", "/step", "/step_random", "/advance"]:
        assert client.post(path, json={}).status_code in {404, 405}
