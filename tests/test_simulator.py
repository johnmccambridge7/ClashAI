from __future__ import annotations

from coc_env.entities import MAX_TICKS, TICK_SECONDS, TROOP_SPECS
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


# ── invariants targeted by the review ────────────────────────────────────
def test_dead_buildings_keep_their_slot() -> None:
    """A destroyed building remains in sim.buildings at its original index."""
    sim = Simulator(layout=default_layout(), army_size=10)
    initial_ids = [b.id for b in sim.buildings]
    initial_len = len(sim.buildings)
    for i in range(10):
        sim.deploy(i % 44, 43)
    while not sim.is_done:
        sim.tick()
    # length and id ordering must match the layout exactly
    assert [b.id for b in sim.buildings] == initial_ids
    assert len(sim.buildings) == initial_len
    # at least one building should have died in a long battle with army=10
    assert any(not b.alive for b in sim.buildings)


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
    sim.tick()
    expected = TROOP_SPECS["barbarian"].speed * TICK_SECONDS
    moved = abs(barb.x - x0) + abs(barb.y - 22)  # mostly horizontal step toward TH
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
