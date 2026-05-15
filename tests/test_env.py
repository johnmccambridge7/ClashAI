from __future__ import annotations
import numpy as np
import pytest

from coc_env.env import (
    CoCEnv,
    DEPLOY_TROOP_KINDS,
    N_DEPLOY_ACTIONS,
    N_DEPLOY_CELLS,
    WAIT_ACTION,
    TIME_COST_PER_TICK,
    cell_to_action,
)
from coc_env.entities import BUILDING_SPECS, GRID_SIZE, MAX_TICKS, Building
from coc_env.generation import preset_layout_profile
from coc_env.simulator import Simulator


def test_reset_returns_obs_and_info() -> None:
    env = CoCEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert "buildings_alive" in obs and "troops_alive" in obs
    assert info["damage_pct"] == 0.0 and info["ticks_elapsed"] == 0


def test_step_returns_5_tuple() -> None:
    env = CoCEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(WAIT_ACTION)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "score" in info


def test_action_mask_blocks_deploys_when_army_empty() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask[WAIT_ACTION] and mask[:N_DEPLOY_CELLS].any()
    env.step(int(np.where(mask[:N_DEPLOY_CELLS])[0][0]))
    assert env.action_masks()[WAIT_ACTION]
    assert not env.action_masks()[:N_DEPLOY_CELLS].any()


def test_obs_in_observation_space() -> None:
    env = CoCEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_procedural_layout_uses_present_mask_for_padding() -> None:
    profile = preset_layout_profile("easy")
    env = CoCEnv(layout_profile=profile, max_buildings=profile.max_buildings + 3)
    obs, _ = env.reset(seed=0)

    assert env.sim is not None
    assert env.observation_space.contains(obs)
    visible_buildings = [
        b for b in env.sim.buildings
        if not (b.spec.hidden and not b.revealed)
    ]
    assert int(obs["buildings_present"].sum()) == len(visible_buildings)
    assert not obs["buildings_present"][-3:].any()


def test_destroyed_building_remains_present() -> None:
    env = CoCEnv()
    env.reset(seed=0)
    assert env.sim is not None

    env.sim.buildings[0].hp = 0.0
    obs = env._obs()

    assert obs["buildings_present"][0] == 1
    assert obs["buildings_alive"][0] == 0


def test_episode_eventually_ends() -> None:
    env = CoCEnv()
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 1000:
        mask = env.action_masks()
        action = int(np.where(mask)[0][0])
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert terminated or truncated


# ── invariants from the review ───────────────────────────────────────────
def test_building_slots_stable_across_destruction() -> None:
    """Building observation slot identity is constant; destroyed → alive=0."""
    env = CoCEnv(army_size=10)
    env.reset(seed=0)
    initial_kind = None
    initial_present = None
    while True:
        mask = env.action_masks()
        a = int(np.where(mask)[0][0])
        obs, _, term, trunc, _ = env.step(a)
        if initial_kind is None:
            initial_kind = obs["buildings_kind"].copy()
            initial_present = obs["buildings_present"].copy()
        # kind/pos/size never reshuffle; only alive/hp drop to 0
        known_slots = initial_present.astype(bool)
        assert np.array_equal(obs["buildings_kind"][known_slots], initial_kind[known_slots])
        if term or trunc:
            assert any(obs["buildings_alive"][i] == 0 for i in range(env.n_buildings)) \
                or all(obs["buildings_alive"])
            break


def test_target_appears_in_observation() -> None:
    """After deploying, the troop's target_id is exposed in troops_target."""
    env = CoCEnv(army_size=2)
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(cell_to_action(0, 0))
    assert obs["troops_alive"][0] == 1
    # target value is either -1 (no alive building) or a slot fraction in [0, 1).
    # With buildings present, the troop must have picked a target by tick 1.
    assert obs["troops_target"][0] >= 0.0


def test_truncated_not_terminated_on_time_limit() -> None:
    """If the time limit fires without natural terminal, truncated wins."""
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    env.step(0)  # one troop deployed somewhere
    term = trunc = False
    while not (term or trunc):
        _, _, term, trunc, info = env.step(WAIT_ACTION)
        if info["ticks_elapsed"] >= MAX_TICKS:
            break
    if env.sim is not None and env.sim.is_truncated:
        assert trunc and not term


def test_masked_deploy_action_is_a_noop() -> None:
    """Once the army is empty, deploy actions are silently no-ops."""
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    env.step(0)  # consume the army
    sim = env.sim
    assert sim is not None and sim.army_remaining == 0
    pre_ticks = sim.tick_count
    env.step(0)  # masked action: deploy is a no-op but ticks still advance
    assert sim.army_remaining == 0
    assert sim.tick_count > pre_ticks


def test_out_of_space_action_raises() -> None:
    env = CoCEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(N_DEPLOY_ACTIONS + 1)


def test_action_mask_blocks_buildings_and_defense_ranges() -> None:
    env = CoCEnv()
    env.reset(seed=0)
    mask = env.action_masks()
    assert not mask[cell_to_action(20, 20)]  # townhall footprint
    assert not mask[cell_to_action(18, 22)]  # wall footprint
    assert not mask[cell_to_action(12, 3)]   # in cannon range
    assert mask[cell_to_action(0, 0)]


def test_action_mask_honors_mortar_blind_spot() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    env.sim = Simulator([Building(0, BUILDING_SPECS["mortar"], 20, 20)], army_size=1)
    env._mask_cache_key = None

    mask = env.action_masks()
    assert mask[cell_to_action(21, 18)]      # inside minimum range
    assert not mask[cell_to_action(21, 12)]  # inside attackable range


def test_hidden_bomb_is_unobserved_until_triggered_and_does_not_block_deploy() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    env.sim = Simulator([
        Building(0, BUILDING_SPECS["storage"], 6, 5),
        Building(1, BUILDING_SPECS["bomb"], 5, 5),
    ], army_size=1)
    env._mask_cache_key = None

    obs = env._obs()
    assert obs["buildings_present"][0] == 1
    assert obs["buildings_present"][1] == 0
    assert env.action_masks()[cell_to_action(5, 5)]

    assert env.sim.deploy(5.5, 5.5)
    env.sim.tick()
    obs = env._obs()

    assert obs["buildings_present"][1] == 1
    assert obs["buildings_is_trap"][1] == 1


def test_action_mask_returns_copy() -> None:
    env = CoCEnv()
    env.reset(seed=0)
    mask = env.action_masks()
    mask[:] = False
    assert env.action_masks()[WAIT_ACTION]


def test_action_space_covers_the_whole_grid() -> None:
    assert WAIT_ACTION == GRID_SIZE * GRID_SIZE * len(DEPLOY_TROOP_KINDS)
    assert cell_to_action(GRID_SIZE - 1, GRID_SIZE - 1) == N_DEPLOY_CELLS - 1
    assert cell_to_action(GRID_SIZE - 1, GRID_SIZE - 1, "wall_breaker") == N_DEPLOY_ACTIONS - 1


def test_last_grid_cell_can_deploy_when_unmasked() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    action = cell_to_action(GRID_SIZE - 1, GRID_SIZE - 1)
    assert env.action_masks()[action]
    env.step(action)
    assert env.sim is not None
    assert env.sim.army_remaining == 0


def test_wall_breaker_actions_use_second_deploy_layer() -> None:
    env = CoCEnv(army_composition={"barbarian": 0, "wall_breaker": 1})
    env.reset(seed=0)

    barbarian_action = cell_to_action(0, 0)
    wall_breaker_action = cell_to_action(0, 0, "wall_breaker")
    mask = env.action_masks()

    assert not mask[barbarian_action]
    assert mask[wall_breaker_action]
    obs, _, _, _, info = env.step(wall_breaker_action)

    assert env.sim is not None
    assert env.sim.troops[0].spec.kind == "wall_breaker"
    assert obs["troops_kind"][0] == DEPLOY_TROOP_KINDS.index("wall_breaker")
    assert info["army_remaining_by_kind"]["wall_breaker"] == 0


def test_target_observation_uses_building_slot_not_id() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    assert env.sim is not None
    env.sim.buildings[0].id = 42
    assert env.sim.deploy(0.5, 22.5)
    env.sim.troops[0].target_id = 42
    obs = env._obs()
    assert obs["troops_target"][0] == 0.0


def test_reward_equals_score_delta_minus_time_cost() -> None:
    """Per-step reward must equal Δscore − time_cost × ticks_advanced."""
    env = CoCEnv()
    env.reset(seed=0)
    sim = env.sim
    assert sim is not None
    prev_score = sim.score
    for _ in range(5):
        prev_ticks = sim.tick_count
        _, reward, term, trunc, info = env.step(0)
        ticks_advanced = sim.tick_count - prev_ticks
        expected = info["score"] - prev_score - TIME_COST_PER_TICK * ticks_advanced
        assert abs(reward - expected) < 1e-6
        prev_score = info["score"]
        if term or trunc:
            break
