from __future__ import annotations
import numpy as np
import pytest

from coc_env.env import (
    CoCEnv,
    DEPLOY_SPELL_KINDS,
    DEPLOY_TROOP_KINDS,
    N_DEPLOY_ACTIONS,
    N_DEPLOY_CELLS,
    N_SPELL_ACTIONS,
    SPELL_KIND_INDEX,
    WAIT_ACTION,
    TIME_COST_PER_TICK,
    cell_to_action,
    spell_to_action,
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
    assert info["profile"] == "default"


def test_info_reports_active_layout_profile() -> None:
    env = CoCEnv(layout_profile="easy", max_buildings=preset_layout_profile("hard").max_buildings)
    _, info = env.reset(seed=0)
    assert info["profile"] == "easy"

    _, info = env.reset(seed=1, options={"profile": "hard"})
    assert info["profile"] == "hard"


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


def test_action_mask_blocks_townhall_buffer() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    env.sim = Simulator([Building(0, BUILDING_SPECS["townhall"], 20, 20)], army_size=1)
    env._mask_cache_key = None

    mask = env.action_masks()
    assert not mask[cell_to_action(19, 21)]  # adjacent to townhall footprint
    assert mask[cell_to_action(16, 21)]      # outside the small townhall buffer


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


def test_spell_actions_are_optional_extra_layers() -> None:
    plain = CoCEnv()
    spell_env = CoCEnv(army_size=1, spell_composition={"rage": 2, "freeze": 2})
    assert plain.action_space.n == N_DEPLOY_ACTIONS + 1
    assert spell_env.action_space.n == N_DEPLOY_ACTIONS + N_SPELL_ACTIONS + 1
    assert spell_env.wait_action == WAIT_ACTION

    obs, _ = spell_env.reset(seed=0)
    before = spell_env.sim.spells_remaining if spell_env.sim is not None else 0
    spell_env.step(WAIT_ACTION)
    assert spell_env.sim is not None
    assert spell_env.sim.spells_remaining == before
    assert not spell_env.action_masks()[spell_to_action(20, 20, "rage")]

    spell_env.sim = Simulator(
        [Building(0, BUILDING_SPECS["storage"], 10, 10)],
        army_size=1,
        spell_composition={"rage": 2, "freeze": 2},
    )
    assert spell_env.sim.deploy(9.5, 11.5)
    spell_env._mask_cache_key = None
    action = spell_to_action(9, 11, "rage")
    assert spell_env.action_masks()[action]
    obs, _, _, _, info = spell_env.step(action)

    assert "spells_remaining" in obs
    assert obs["active_spells_present"].sum() == 1
    assert info["spells_remaining_by_kind"]["rage"] == 1
    assert info["active_spells"] == 1
    assert DEPLOY_SPELL_KINDS == ("rage", "freeze")


def test_spell_action_masks_are_semantic() -> None:
    env = CoCEnv(army_size=1, spell_composition={"rage": 2, "freeze": 2})
    env.reset(seed=0)
    rage_offset = WAIT_ACTION + 1 + SPELL_KIND_INDEX["rage"] * N_DEPLOY_CELLS
    freeze_offset = WAIT_ACTION + 1 + SPELL_KIND_INDEX["freeze"] * N_DEPLOY_CELLS

    mask = env.action_masks()
    assert not mask[rage_offset:rage_offset + N_DEPLOY_CELLS].any()
    assert not mask[freeze_offset:freeze_offset + N_DEPLOY_CELLS].any()

    env.sim = Simulator([Building(0, BUILDING_SPECS["storage"], 10, 10)], army_size=1, spell_composition={"rage": 2, "freeze": 2})
    assert env.sim.deploy(8.5, 8.5)
    env._mask_cache_key = None
    mask = env.action_masks()
    assert not mask[rage_offset:rage_offset + N_DEPLOY_CELLS].any()
    assert not mask[freeze_offset:freeze_offset + N_DEPLOY_CELLS].any()

    env.sim = Simulator([Building(0, BUILDING_SPECS["storage"], 10, 10)], army_size=1, spell_composition={"rage": 2, "freeze": 2})
    assert env.sim.deploy(9.5, 11.5)
    env._mask_cache_key = None
    mask = env.action_masks()
    assert 0 < int(mask[rage_offset:rage_offset + N_DEPLOY_CELLS].sum()) <= 32
    assert not mask[freeze_offset:freeze_offset + N_DEPLOY_CELLS].any()

    env.sim = Simulator([Building(0, BUILDING_SPECS["cannon"], 10, 10)], army_size=1, spell_composition={"rage": 2, "freeze": 2})
    assert env.sim.deploy(15.5, 11.5)
    env._mask_cache_key = None
    mask = env.action_masks()
    assert 0 < int(mask[freeze_offset:freeze_offset + N_DEPLOY_CELLS].sum()) <= 24
    assert mask[spell_to_action(11, 11, "freeze")]


def test_terminal_unused_spell_penalty() -> None:
    env = CoCEnv(army_size=1, spell_composition={"rage": 2, "freeze": 2})
    env.reset(seed=0)
    assert env.sim is not None
    env.sim.army_remaining_by_kind["barbarian"] = 0

    _, reward, terminated, truncated, info = env.step(WAIT_ACTION)

    assert terminated and not truncated
    assert reward == pytest.approx(-0.12)
    assert info["unused_spell_penalty_total"] == pytest.approx(0.12)
    assert info["spells_remaining"] == 4


def test_wasted_spell_cast_is_penalized_when_forced() -> None:
    env = CoCEnv(army_size=1, spell_composition={"rage": 1})
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(spell_to_action(0, 0, "rage"))

    assert not terminated and not truncated
    assert reward < -0.02
    assert info["wasted_spell_casts"] == 1
    assert info["wasted_spell_penalty_total"] == pytest.approx(0.02)


def test_useful_spell_shaping_rewards_effective_rage_and_freeze() -> None:
    rage_env = CoCEnv(army_size=1, spell_composition={"rage": 1})
    rage_env.reset(seed=0)
    rage_env.sim = Simulator([Building(0, BUILDING_SPECS["storage"], 10, 10)], army_size=1, spell_composition={"rage": 1})
    assert rage_env.sim.deploy(9.5, 11.5)
    rage_env._mask_cache_key = None
    _, _, _, _, rage_info = rage_env.step(spell_to_action(9, 11, "rage"))

    assert rage_info["useful_rage_casts"] == 1
    assert rage_info["wasted_spell_casts"] == 0
    assert rage_info["reward_rage_effect"] > 0.0
    assert rage_info["rage_bonus_damage_pct"] > 0.0

    freeze_env = CoCEnv(army_size=1, spell_composition={"freeze": 1})
    freeze_env.reset(seed=0)
    freeze_env.sim = Simulator([Building(0, BUILDING_SPECS["cannon"], 10, 10)], army_size=1, spell_composition={"freeze": 1})
    assert freeze_env.sim.deploy(15.5, 11.5)
    freeze_env._mask_cache_key = None
    _, _, _, _, freeze_info = freeze_env.step(spell_to_action(11, 11, "freeze"))

    assert freeze_info["useful_freeze_casts"] == 1
    assert freeze_info["reward_freeze_effect"] > 0.0
    assert freeze_info["frozen_threat_ticks"] > 0
    assert freeze_info["frozen_threat_damage_pct"] > 0.0


def test_reward_v3_objective_shaping_tracks_defense_and_townhall_damage() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    env.sim = Simulator([
        Building(0, BUILDING_SPECS["cannon"], 10, 10),
        Building(1, BUILDING_SPECS["townhall"], 14, 10),
    ], army_size=1)
    assert env.sim.deploy(9.5, 11.5)
    env._mask_cache_key = None

    for _ in range(20):
        _, reward, terminated, truncated, info = env.step(WAIT_ACTION)
        if info["defense_damage_pct"] > 0.0:
            assert reward > info["reward_base"]
            assert info["reward_defense_damage"] > 0.0
            assert info["objective_shaping_reward_total"] > 0.0
            break
        if terminated or truncated:
            break
    else:
        raise AssertionError("troop did not damage defense")


def test_freeze_reward_requires_threatened_defense_not_troop_in_radius() -> None:
    env = CoCEnv(army_size=1, spell_composition={"freeze": 1})
    env.reset(seed=0)
    env.sim = Simulator([Building(0, BUILDING_SPECS["cannon"], 10, 10)], army_size=1, spell_composition={"freeze": 1})
    assert env.sim.deploy(35.5, 35.5)
    env._mask_cache_key = None

    _, _, _, _, info = env.step(spell_to_action(11, 11, "freeze"))

    assert info["useful_freeze_casts"] == 0
    assert info["wasted_freeze_casts"] == 1
    assert info["frozen_defense_ticks"] > 0
    assert info["frozen_threat_ticks"] == 0
    assert info["frozen_threat_damage_pct"] == 0.0
    assert info["reward_freeze_effect"] == 0.0


def test_deployment_and_splash_risk_diagnostics_are_reported() -> None:
    env = CoCEnv(army_size=2)
    env.reset(seed=0)
    env.sim = Simulator([Building(0, BUILDING_SPECS["wizard_tower"], 10, 10)], army_size=2)
    assert env.sim.deploy(15.0, 11.5)
    assert env.sim.deploy(15.6, 11.5)
    env._mask_cache_key = None

    _, reward, _, _, info = env.step(WAIT_ACTION)

    assert info["deployments"] == 2
    assert info["deployment_unique_cells"] == 1
    assert info["deployment_top_cell_fraction"] == pytest.approx(1.0)
    assert info["deployment_quadrant_entropy"] == pytest.approx(0.0)
    assert info["splash_cluster_risk_pct"] > 0.0
    assert info["splash_damage_taken_pct"] > 0.0
    assert info["multi_hit_splash_damage_pct"] > 0.0
    assert info["multi_hit_splash_events"] > 0
    assert info["reward_splash_risk_penalty"] > 0.0
    assert info["reward_multi_hit_splash_penalty"] > 0.0
    assert reward < (
        info["reward_base"]
        + info["reward_rage_effect"]
        + info["reward_freeze_effect"]
        + info["reward_objective_shaping"]
    )


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


def test_repeated_barbarian_deployments_are_mildly_penalized() -> None:
    env = CoCEnv(army_size=2)
    env.reset(seed=0)
    env.sim = Simulator([Building(0, BUILDING_SPECS["storage"], 10, 10)], army_size=2)
    env._mask_cache_key = None

    env.step(cell_to_action(0, 0))
    _, _, _, _, info = env.step(cell_to_action(0, 0))

    assert info["deploy_crowding_count"] == 1.0
    assert info["reward_deploy_crowding_penalty"] > 0.0
    assert info["deploy_crowding_penalty_total"] == info["reward_deploy_crowding_penalty"]


def test_target_observation_uses_building_slot_not_id() -> None:
    env = CoCEnv(army_size=1)
    env.reset(seed=0)
    assert env.sim is not None
    env.sim.buildings[0].id = 42
    assert env.sim.deploy(0.5, 22.5)
    env.sim.troops[0].target_id = 42
    obs = env._obs()
    assert obs["troops_target"][0] == 0.0


def test_reward_components_sum_to_step_reward() -> None:
    """Per-step reward must equal base score delta plus explicit shaping terms."""
    env = CoCEnv()
    env.reset(seed=0)
    sim = env.sim
    assert sim is not None
    prev_score = sim.score
    for _ in range(5):
        prev_ticks = sim.tick_count
        _, reward, term, trunc, info = env.step(0)
        ticks_advanced = sim.tick_count - prev_ticks
        expected_base = info["score"] - prev_score - TIME_COST_PER_TICK * ticks_advanced
        expected = (
            info["reward_base"]
            + info["reward_rage_effect"]
            + info["reward_freeze_effect"]
            + info["reward_objective_shaping"]
            + info["reward_tactical_shaping"]
            - info["reward_wasted_spell_penalty"]
            - info["reward_unused_spell_penalty"]
        )
        assert info["reward_base"] == pytest.approx(expected_base)
        assert abs(reward - expected) < 1e-6
        prev_score = info["score"]
        if term or trunc:
            break
