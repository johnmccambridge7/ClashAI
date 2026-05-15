from __future__ import annotations

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
