from __future__ import annotations
import math

import pytest

from coc_env.entities import BUILDING_SPECS, GRID_SIZE, MAX_TICKS, TICK_SECONDS, TROOP_SPECS, Building
from coc_env.layouts import default_layout
from coc_env.simulator import Simulator


def _full_episode_sim(seed: int = 7, army: int = 10) -> Simulator:
    sim = Simulator(layout=default_layout(), army_size=army, seed=seed)
    for i in range(army):
        sim.deploy(i % 44, 43)
    while not sim.is_done:
        sim.tick()
    return sim


def test_initial_state() -> None:
    sim = Simulator(layout=default_layout(), army_size=10)
    assert sim.tick_count == 0
    assert sim.damage_pct == 0.0
    assert sim.score == 0.0
    assert sim.army_remaining == 10
    assert not sim.is_done
    assert not sim.is_terminal
    assert not sim.is_truncated


def test_deploy() -> None:
    sim = Simulator(layout=default_layout(), army_size=5)
    assert sim.deploy(0, 22)
    assert sim.army_remaining == 4
    assert len(sim.active_troops) == 1
    assert sim.active_troops[0].x == 0.0 and sim.active_troops[0].y == 22.0


def test_deploy_fails_when_army_empty() -> None:
    sim = Simulator(layout=default_layout(), army_size=0)
    assert not sim.deploy(0, 0)


def test_deploy_fails_off_grid() -> None:
    sim = Simulator(layout=default_layout(), army_size=1)
    assert not sim.deploy(-1, 22)
    assert not sim.deploy(44, 22)
    assert sim.army_remaining == 1


def test_deploy_accepts_cell_centers_on_last_row_and_column() -> None:
    sim = Simulator(layout=default_layout(), army_size=1)
    assert sim.deploy(43.5, 43.5)
    assert sim.army_remaining == 0


def test_deploy_fails_on_building_or_wall() -> None:
    sim = Simulator(layout=default_layout(), army_size=2)
    assert not sim.deploy(20.5, 20.5)  # townhall
    assert not sim.deploy(18.5, 22.5)  # wall
    assert sim.army_remaining == 2


def test_simulator_clones_layout_state() -> None:
    layout = default_layout()
    sim = Simulator(layout=layout, army_size=1)
    sim.buildings[0].hp = 1.0
    assert layout[0].hp == float(layout[0].spec.hp)
    assert Simulator(layout=layout, army_size=1).buildings[0].hp == float(layout[0].spec.hp)


def test_determinism() -> None:
    a = _full_episode_sim(seed=7)
    b = _full_episode_sim(seed=7)
    assert (a.damage_pct, a.tick_count, a.score) == (b.damage_pct, b.tick_count, b.score)


def test_episode_terminates() -> None:
    sim = _full_episode_sim()
    assert sim.is_done
    assert sim.tick_count <= MAX_TICKS


def test_score_monotonic() -> None:
    sim = Simulator(layout=default_layout(), army_size=10)
    for i in range(10):
        sim.deploy(i % 44, 43)
    prev = 0.0
    while not sim.is_done:
        sim.tick()
        assert sim.score + 1e-9 >= prev
        prev = sim.score


def test_score_stars_dominate_damage_percentage() -> None:
    one_star = Simulator(layout=default_layout(), army_size=0)
    for b in one_star.buildings:
        b.hp = 0.0
    townhall = next(b for b in one_star.buildings if b.spec.kind == "townhall")
    townhall.hp = one_star.original_total_hp * 0.01
    assert one_star.stars == 1

    two_star = Simulator(layout=default_layout(), army_size=0)
    for b in two_star.buildings:
        b.hp = 0.0
    remaining = two_star.original_total_hp * 0.5
    for b in two_star.buildings:
        if b.spec.kind != "townhall" and b.spec.counts_for_score and remaining > 0:
            b.hp = min(float(b.spec.hp), remaining)
            remaining -= b.hp
    assert two_star.stars == 2
    assert two_star.score > one_star.score


# ── invariants targeted by the review ────────────────────────────────────
def test_dead_buildings_keep_their_slot() -> None:
    """A destroyed building remains in sim.buildings at its original index."""
    sim = Simulator(layout=default_layout(), army_size=0)
    initial_ids = [b.id for b in sim.buildings]
    initial_len = len(sim.buildings)
    sim.buildings[0].hp = 0.0

    assert [b.id for b in sim.buildings] == initial_ids
    assert len(sim.buildings) == initial_len
    assert not sim.buildings[0].alive


def test_dead_troops_keep_their_slot() -> None:
    sim = Simulator(layout=default_layout(), army_size=10)
    for i in range(10):
        sim.deploy(i % 44, 43)
    # tick until at least one troop has died
    for _ in range(120):
        sim.tick()
    # troop ids must be 0..n-1 in order (no removal shifted them)
    assert [t.id for t in sim.troops] == list(range(len(sim.troops)))


def test_truncation_vs_termination() -> None:
    """Forcing time to exceed MAX_TICKS yields truncated, not terminated."""
    sim = Simulator(layout=default_layout(), army_size=1)
    sim.deploy(20, 0)  # one troop, far from buildings
    # Drive ticks past MAX_TICKS without destroying everything
    while sim.tick_count < MAX_TICKS and not sim.is_terminal:
        sim.tick()
    if sim.is_truncated:
        assert not sim.is_terminal
        assert sim.tick_count >= MAX_TICKS


def test_terminal_when_buildings_all_dead() -> None:
    """Manually kill all buildings; must be terminal, not truncated."""
    sim = Simulator(layout=default_layout(), army_size=1)
    for b in sim.buildings:
        b.hp = 0
    sim.tick()
    assert sim.is_terminal
    assert not sim.is_truncated


def test_movement_unit_is_tiles_per_second() -> None:
    """Troop should traverse speed * TICK_SECONDS tiles per tick, not speed."""
    sim = Simulator(layout=default_layout(), army_size=1)
    sim.deploy(0, 22)
    barb = sim.active_troops[0]
    barb.target_id = sim.buildings[0].id  # lock target so it moves
    x0 = barb.x
    y0 = barb.y
    sim.tick()
    expected = TROOP_SPECS["barbarian"].speed * TICK_SECONDS
    moved = math.hypot(barb.x - x0, barb.y - y0)
    assert moved <= expected * 1.05  # within rounding


def test_damage_pct_clamps() -> None:
    """damage_pct stays in [0, 1] even if hp would overshoot."""
    sim = Simulator(layout=default_layout(), army_size=10)
    for i in range(10):
        sim.deploy(i % 44, 43)
    while not sim.is_done:
        sim.tick()
        assert 0.0 <= sim.damage_pct <= 1.0


def test_walls_do_not_count_for_score_or_terminal() -> None:
    sim = Simulator(layout=default_layout(), army_size=1)
    for b in sim.buildings:
        if b.spec.counts_for_score:
            b.hp = 0.0
    assert sim.damage_pct == 1.0
    assert sim.is_terminal
    assert any(b.alive and b.spec.kind == "wall" for b in sim.buildings)


def test_destroying_only_walls_does_not_reward_score() -> None:
    layout = [
        Building(0, BUILDING_SPECS["townhall"], 20, 20),
        Building(1, BUILDING_SPECS["wall"], 18, 22),
    ]
    sim = Simulator(layout=layout, army_size=0)
    wall = next(b for b in sim.buildings if b.spec.kind == "wall")
    wall.hp = 0.0

    assert sim.damage_pct == 0.0
    assert sim.stars == 0
    assert sim.score == 0.0


def test_damage_counters_ignore_walls_and_track_objectives() -> None:
    layout = [
        Building(0, BUILDING_SPECS["townhall"], 20, 20),
        Building(1, BUILDING_SPECS["cannon"], 10, 10),
        Building(2, BUILDING_SPECS["wall"], 18, 22),
    ]
    sim = Simulator(layout=layout, army_size=0)
    townhall, cannon, wall = sim.buildings

    sim._damage_building(wall, wall.spec.hp)
    assert sim.defense_damage == 0.0
    assert sim.townhall_damage == 0.0
    assert sim.defenses_destroyed == 0
    assert sim.townhall_destroyed == 0

    sim._damage_building(cannon, cannon.spec.hp)
    sim._damage_building(townhall, townhall.spec.hp)

    assert sim.defense_damage == cannon.spec.hp
    assert sim.townhall_damage == townhall.spec.hp
    assert sim.defenses_destroyed == 1
    assert sim.townhall_destroyed == 1


def test_wall_blocks_troop_and_becomes_target() -> None:
    sim = Simulator(layout=default_layout(), army_size=1)
    assert sim.deploy(17.5, 22.5)
    troop = sim.troops[0]
    troop.target_id = next(b.id for b in sim.buildings if b.spec.kind == "townhall")
    for _ in range(6):
        sim.tick()
        target = next((b for b in sim.buildings if b.id == troop.target_id), None)
        if target is not None and target.spec.kind == "wall":
            assert target.hp < target.spec.hp
            return
    raise AssertionError("troop did not attack the wall before crossing it")


def test_wizard_tower_splash_damages_grouped_troops() -> None:
    layout = [Building(0, BUILDING_SPECS["wizard_tower"], 10, 10)]
    sim = Simulator(layout=layout, army_size=2)
    assert sim.deploy(15.0, 11.5)
    assert sim.deploy(15.6, 11.5)

    sim.tick()

    assert len(sim.troops) == 2
    assert all(t.hp < t.spec.hp for t in sim.troops)
    assert sim.troop_damage_taken > 0.0
    assert sim.splash_damage_taken > 0.0
    assert sim.multi_hit_splash_damage_taken > 0.0
    assert sim.splash_cluster_risk > 0.0
    assert sim.multi_hit_splash_events == 1


def test_defense_fire_and_impact_are_explicit_visual_events() -> None:
    layout = [Building(0, BUILDING_SPECS["wizard_tower"], 10, 10)]
    sim = Simulator(layout=layout, army_size=1)
    assert sim.deploy(15.0, 11.5)

    sim.tick()

    assert [event["kind"] for event in sim.visual_events] == ["defense_fire", "impact"]
    fire, impact = sim.visual_events
    assert fire["tick"] == sim.tick_count
    assert fire["attacker_id"] == 0
    assert fire["target_id"] == 0
    assert fire["source_kind"] == "wizard_tower"
    assert impact["tick"] == sim.tick_count
    assert impact["source_kind"] == "wizard_tower"
    assert all(event["kind"] not in {"defense_fire", "impact"} for event in sim.current_engagements())


def test_mortar_uses_blind_spot_and_delayed_splash() -> None:
    layout = [Building(0, BUILDING_SPECS["mortar"], 20, 20)]
    sim = Simulator(layout=layout, army_size=2)
    assert sim.deploy(21.5, 18.5)  # inside minimum range
    assert sim.deploy(21.5, 12.5)  # inside maximum range

    sim.tick()

    assert len(sim.pending_impacts) == 1
    impact = sim.pending_impacts[0]
    assert impact.source_kind == "mortar"
    assert impact.radius == BUILDING_SPECS["mortar"].splash_radius
    assert math.hypot(impact.x - sim.troops[1].x, impact.y - sim.troops[1].y) < 1e-9


def test_mortar_delayed_impact_emits_visual_event_on_impact_tick() -> None:
    layout = [Building(0, BUILDING_SPECS["mortar"], 20, 20)]
    sim = Simulator(layout=layout, army_size=1)
    assert sim.deploy(21.5, 12.5)

    sim.tick()
    fire = next(event for event in sim.visual_events if event["kind"] == "defense_fire")
    impact_tick = int(fire["impact_tick"])
    assert impact_tick > sim.tick_count

    while sim.tick_count < impact_tick:
        sim.tick()

    impact = next(event for event in sim.visual_events if event["kind"] == "impact")
    assert impact["tick"] == impact_tick
    assert impact["source_kind"] == "mortar"
    assert impact["impact_tick"] == impact_tick


def test_hidden_bomb_reveals_triggers_and_explodes_once() -> None:
    layout = [
        Building(0, BUILDING_SPECS["storage"], 6, 5),
        Building(1, BUILDING_SPECS["bomb"], 5, 5),
    ]
    sim = Simulator(layout=layout, army_size=1)
    assert [b.spec.kind for b in sim.visible_buildings] == ["storage"]
    assert sim.deploy(5.5, 5.5)

    sim.tick()
    bomb = sim.buildings[1]
    assert bomb.revealed
    assert bomb.triggered
    assert bomb.alive

    for _ in range(math.ceil(bomb.spec.trigger_delay / TICK_SECONDS) + 1):
        sim.tick()

    assert not sim.troops[0].alive
    assert not bomb.alive


def test_troop_paths_around_wall_gap() -> None:
    layout = [Building(0, BUILDING_SPECS["townhall"], 10, 10)]
    wall_id = 1
    for y in range(GRID_SIZE):
        if y == 12:
            continue
        layout.append(Building(wall_id, BUILDING_SPECS["wall"], 8, y))
        wall_id += 1
    sim = Simulator(layout=layout, army_size=1)
    assert sim.deploy(5.5, 12.5)
    troop = sim.troops[0]
    troop.target_id = 0

    for _ in range(10):
        sim.tick()

    assert troop.x > 8.5
    assert all(b.hp == b.spec.hp for b in sim.buildings if b.spec.kind == "wall")


def test_troop_attacks_wall_when_no_gap_exists() -> None:
    layout = [Building(0, BUILDING_SPECS["townhall"], 10, 10)]
    for i, y in enumerate(range(GRID_SIZE), start=1):
        layout.append(Building(i, BUILDING_SPECS["wall"], 8, y))
    sim = Simulator(layout=layout, army_size=1)
    assert sim.deploy(5.5, 12.5)
    troop = sim.troops[0]
    troop.target_id = 0

    for _ in range(10):
        sim.tick()
        target = next((b for b in sim.buildings if b.id == troop.target_id), None)
        if target is not None and target.spec.kind == "wall":
            assert target.hp < target.spec.hp
            return

    raise AssertionError("troop did not target the wall")


def test_troop_uses_diagonal_gap() -> None:
    """8-directional movement: first step is diagonal, then troop slips through the gap."""
    # Wall line at x=8 covering all y except y=11 (gap one row above deploy point).
    # With 4-directional pathfinding, the first step from (5.5, 12.5) is purely
    # cardinal (only x or only y changes). 8-directional routing steps NE diagonally
    # toward the gap cell (8, 11), changing both x AND y on the very first tick.
    layout = [Building(0, BUILDING_SPECS["townhall"], 12, 8)]
    wall_id = 1
    for y in range(GRID_SIZE):
        if y == 11:
            continue
        layout.append(Building(wall_id, BUILDING_SPECS["wall"], 8, y))
        wall_id += 1
    sim = Simulator(layout=layout, army_size=1)
    assert sim.deploy(5.5, 12.5)
    troop = sim.troops[0]
    troop.target_id = 0

    # First tick must show simultaneous east + north motion — strict 8-dir signature.
    sim.tick()
    assert troop.x > 5.5, f"troop did not move east on tick 1 (x={troop.x})"
    assert troop.y < 12.5, (
        f"troop did not move north on tick 1 (y={troop.y}) — 4-dir routing in use"
    )

    # And the troop must traverse the gap without damaging walls.
    for _ in range(13):
        sim.tick()
    assert troop.x > 8.5, f"troop stalled at x={troop.x}, did not pass through gap"
    assert all(b.hp == b.spec.hp for b in sim.buildings if b.spec.kind == "wall"), \
        "wall was damaged — troop did not use the gap"


def test_rule_of_three_prefers_open_building() -> None:
    """Rule of Three: a farther-but-open building beats a closer wall-surrounded one."""
    # Storage at (5, 5) — closer by Euclidean distance, sealed inside a *double-thick*
    # wall ring (mirrors real CoC base designs where a townhall sits behind two layers
    # of walls). Townhall at (30, 5) — farther but fully open.
    # With WALL_STEP_COST=20 the walled-storage path costs ~5+40=45 vs the open
    # townhall at ~30. Path-cost targeting should pick the townhall.
    storage_id = 0
    townhall_id = 1
    layout: list[Building] = [
        Building(storage_id,  BUILDING_SPECS["storage"],  5, 5),
        Building(townhall_id, BUILDING_SPECS["townhall"], 30, 5),
    ]
    wall_id = 2
    # Inner ring at offset 1 around storage (5,5) size 3 — perimeter (4..8, 4..8)
    for x in range(4, 9):
        for y in (4, 8):
            layout.append(Building(wall_id, BUILDING_SPECS["wall"], x, y))
            wall_id += 1
    for y in range(5, 8):
        layout.append(Building(wall_id, BUILDING_SPECS["wall"], 4, y))
        wall_id += 1
        layout.append(Building(wall_id, BUILDING_SPECS["wall"], 8, y))
        wall_id += 1
    # Outer ring at offset 2 — perimeter (3..9, 3..9)
    for x in range(3, 10):
        for y in (3, 9):
            layout.append(Building(wall_id, BUILDING_SPECS["wall"], x, y))
            wall_id += 1
    for y in range(4, 9):
        layout.append(Building(wall_id, BUILDING_SPECS["wall"], 3, y))
        wall_id += 1
        layout.append(Building(wall_id, BUILDING_SPECS["wall"], 9, y))
        wall_id += 1

    sim = Simulator(layout=layout, army_size=1)
    assert sim.deploy(0.5, 5.5)
    troop = sim.troops[0]

    sim.tick()
    assert troop.target_id == townhall_id, (
        f"expected target={townhall_id} (open townhall) but got {troop.target_id} (walled storage)"
    )


def test_wall_breaker_explodes_on_nearest_connected_walls() -> None:
    walls = [Building(i, BUILDING_SPECS["wall"], 10 + i, 10) for i in range(5)]
    layout = [
        Building(100, BUILDING_SPECS["storage"], 22, 10),
        *walls,
    ]
    sim = Simulator(
        layout=layout,
        army_size=1,
        army_composition={"wall_breaker": 1},
    )

    assert sim.deploy(5.5, 10.5, troop_kind="wall_breaker")
    for _ in range(40):
        sim.tick()
        if not sim.active_troops:
            break

    damaged_walls = [
        b for b in sim.buildings
        if b.spec.kind == "wall" and b.hp == b.spec.hp * 0.5
    ]
    explosion_events = [
        e for e in sim.visual_events
        if e["kind"] == "wall_breaker_explosion"
    ]
    assert len(damaged_walls) == TROOP_SPECS["wall_breaker"].wall_damage_count
    assert len(explosion_events) == 1
    assert explosion_events[0]["source_kind"] == "wall_breaker"
    assert sim.wall_breaker_explosions == 1
    assert sim.wall_breaker_damaged_wall_segments == TROOP_SPECS["wall_breaker"].wall_damage_count
    assert sim.wall_breaker_destroyed_wall_segments == 0
    assert sim.wall_breakers_dead_without_explosion == 0
    assert not sim.active_troops


def test_scored_damage_is_attributed_to_deployment_origin_quadrants() -> None:
    sim = Simulator(
        layout=[Building(0, BUILDING_SPECS["storage"], 10, 10)],
        army_size=2,
    )
    assert sim.deploy(0.5, 0.5)
    assert sim.deploy(35.5, 35.5)

    sim._damage_building(sim.buildings[0], 10.0, attacker=sim.troops[0])
    sim._damage_building(sim.buildings[0], 20.0, attacker=sim.troops[1])

    assert sim.scored_damage_by_origin_quadrant[0] == pytest.approx(10.0)
    assert sim.scored_damage_by_origin_quadrant[3] == pytest.approx(20.0)


def test_rage_spell_multiplies_troop_damage_for_18_seconds() -> None:
    layout = [Building(0, BUILDING_SPECS["storage"], 10, 10)]
    normal = Simulator(layout=layout, army_size=1)
    raged = Simulator(layout=layout, army_size=1, spell_composition={"rage": 1})

    assert normal.deploy(9.5, 11.5)
    assert raged.deploy(9.5, 11.5)
    assert raged.cast_spell(9.5, 11.5, "rage")
    normal.troops[0].target_id = 0
    raged.troops[0].target_id = 0

    normal.tick()
    raged.tick()

    normal_damage = BUILDING_SPECS["storage"].hp - normal.buildings[0].hp
    raged_damage = BUILDING_SPECS["storage"].hp - raged.buildings[0].hp
    assert math.isclose(raged_damage, normal_damage * 1.3, rel_tol=1e-6)

    for _ in range(math.ceil(18.0 / TICK_SECONDS)):
        raged.tick()
    assert not raged.active_spells


def test_spell_usefulness_requires_immediate_tactical_value() -> None:
    rage_runner = Simulator(
        layout=[Building(0, BUILDING_SPECS["storage"], 10, 10)],
        army_size=1,
        spell_composition={"rage": 1},
    )
    assert rage_runner.deploy(5.5, 5.5)
    assert rage_runner.cast_spell(5.5, 5.5, "rage")
    assert rage_runner.useful_spell_casts_by_kind["rage"] == 0
    assert rage_runner.wasted_spell_casts_by_kind["rage"] == 1

    rage_attacker = Simulator(
        layout=[Building(0, BUILDING_SPECS["storage"], 10, 10)],
        army_size=1,
        spell_composition={"rage": 1},
    )
    assert rage_attacker.deploy(9.5, 11.5)
    assert rage_attacker.cast_spell(9.5, 11.5, "rage")
    assert rage_attacker.useful_spell_casts_by_kind["rage"] == 1
    assert rage_attacker.wasted_spell_casts_by_kind["rage"] == 0

    freeze_quiet = Simulator(
        layout=[Building(0, BUILDING_SPECS["cannon"], 10, 10)],
        army_size=1,
        spell_composition={"freeze": 1},
    )
    assert freeze_quiet.deploy(35.5, 35.5)
    assert freeze_quiet.cast_spell(*freeze_quiet.buildings[0].center, "freeze")
    assert freeze_quiet.useful_spell_casts_by_kind["freeze"] == 0
    assert freeze_quiet.wasted_spell_casts_by_kind["freeze"] == 1

    freeze_threat = Simulator(
        layout=[Building(0, BUILDING_SPECS["cannon"], 10, 10)],
        army_size=1,
        spell_composition={"freeze": 1},
    )
    assert freeze_threat.deploy(15.5, 11.5)
    assert freeze_threat.cast_spell(*freeze_threat.buildings[0].center, "freeze")
    assert freeze_threat.useful_spell_casts_by_kind["freeze"] == 1
    assert freeze_threat.wasted_spell_casts_by_kind["freeze"] == 0


def test_freeze_spell_only_stops_defenses_for_6_seconds() -> None:
    layout = [Building(0, BUILDING_SPECS["cannon"], 10, 10)]
    sim = Simulator(layout=layout, army_size=1, spell_composition={"freeze": 1})

    assert sim.deploy(15.5, 11.5)
    assert sim.cast_spell(*sim.buildings[0].center, "freeze")
    troop = sim.troops[0]

    for _ in range(math.ceil(6.0 / TICK_SECONDS)):
        sim.tick()
        assert troop.hp == troop.spec.hp

    assert not sim.active_spells
    sim.tick()
    assert troop.hp < troop.spec.hp
    assert sim.buildings[0].hp < sim.buildings[0].spec.hp
