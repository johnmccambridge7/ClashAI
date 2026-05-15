# ClashAI

A Clash of Clans-style battle simulator wrapped as a Gymnasium environment,
plus a browser viewer for watching episodes. The simulator is deterministic
once seeded and fast enough to use as the inner loop of an RL training run.

The agent's job: pick when and where to drop troops on a 44x44 base, and
maximise the final score (stars plus damage percent).

## What's in the repo

`coc_env/simulator.py` is the battle core. Pure Python, no rendering, no
hidden randomness. The viewer and the Gym wrapper both call into it.

`coc_env/env.py` wraps the simulator as a `gymnasium.Env`. It exposes an
action mask, a stable observation space, and a per-step reward.

`coc_env/generation.py` builds bases procedurally. There are five preset
profiles (`easy`, `medium`, `hard`, `resource_bait`, `split`) that the env
can sample from on reset.

`viewer/` is a Flask app with a sprite-based browser UI. Useful for stepping
through episodes by hand, deploying troops manually, or running the bundled
random policy.

`coc_env/render_pygame.py` plus `scripts/hand_play.py` is the local pygame
alternative if you'd rather script a deploy schedule than click through a
browser.

## Simulator model

The grid is 44 by 44. Time advances in 0.25 second ticks (`TICK_SECONDS`),
capped at 720 ticks (`MAX_TICKS`), which is three minutes of game time. The
Gym wrapper hides 10 ticks per agent decision (`DECISION_INTERVAL`).

Buildings (`coc_env/entities.py`):

| Kind          | Role                                          | Notable stats |
| ------------- | --------------------------------------------- | ------------- |
| `townhall`    | Score anchor; killing it grants one star      | 1500 hp, size 4 |
| `cannon`      | Single-target defence                         | dps 11, range 9 |
| `wizard_tower`| Splash defence                                | dps 11, range 7, splash 1.0 |
| `mortar`      | Long-range splash with a min-range blind spot | dps 5, range 4 to 11, projectile delay 1.0 |
| `storage`     | Counts toward damage percent, no offence      | 900 hp |
| `wall`        | Blocks movement, does not score               | 300 hp, size 1 |
| `bomb`        | Hidden one-shot trap, 1.5 tile splash         | trigger 1.5, delay 1.5 |

Troops:

- `barbarian`: melee, picks targets via a flow field.
- `wall_breaker`: fast and fragile, targets walls, detonates on contact and
  damages a connected wall segment
  (`wall_damage_fraction * wall_damage_count`).

Each tick, troops without a target pick one of the three closest scored
buildings, breaking ties by flow-field cost (Dijkstra from the building's
attack-adjacent cells). Walls aren't impassable. They cost `WALL_STEP_COST =
20` per tile to walk through, so troops route around walls when there's
space and cut through them when there isn't. Flow fields are cached and
invalidated when a building dies (`terrain_version` bumps).

`score = stars + damage_pct`, bounded in `[0, 4]`. Stars: one for 50 percent
damage, one for the townhall, one for 100 percent damage. Per-step reward
is the score delta minus a small time penalty (`TIME_COST_PER_TICK = 1e-4`),
so the cumulative reward over an episode equals the final score minus a few
hundredths.

The episode terminates when every scored building is dead, or when the army
is exhausted with no troops left on the field. It truncates when the tick
limit fires without a terminal cause.

## Gym API

```python
from coc_env import CoCEnv

env = CoCEnv(
    army_composition={"barbarian": 40, "wall_breaker": 10},
    layout_profile="hard",
)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

Action space: `Discrete(GRID_SIZE * GRID_SIZE * len(troops) + 1)`. The first
`N_DEPLOY_CELLS` actions deploy a barbarian at a given cell. The next
`N_DEPLOY_CELLS` deploy a wall breaker. The final action (`WAIT_ACTION`) is
a no-op, and is always legal.

`env.action_masks()` returns a boolean mask of legal actions. It removes
cells inside building footprints, cells inside a live defence's attack
range, and troop kinds you've already used up. The mask is cached against
`(sim_id, terrain_version, army_by_kind, is_done)`, so it is effectively
free in the inner loop.

Observations are a `Dict` of per-slot arrays. Slot identity is stable for
the whole episode: a destroyed building keeps `present=1, alive=0, hp=0`.
Hidden traps stay at `present=0` until they trigger. Padding slots (when
you set `max_buildings` higher than the layout actually contains) also stay
at `present=0`. The keys are:

- Buildings: `present`, `alive`, `kind`, `hp`, `pos`, `size`, `is_defense`,
  `blocks_move`, `dps`, `attack_range`, `min_attack_range`,
  `splash_radius`, `cooldown`, `is_trap`.
- Troops: `alive`, `kind`, `hp`, `pos`, `target`. Target is a slot fraction
  in `[0, 1]`, or `-1` if there is no target.
- Globals: `army_remaining`, `time_remaining`.

## Training

The env was built around the assumption that the agent would be a
discrete-action, action-masked PPO. The action space is large
(`44 * 44 + 1 = 1937`) and most of it is illegal at any given moment,
which is the case action masking exists for.

### Step 1: baseline

Run the random-legal-action baseline first:

```
.venv/bin/python -m scripts.random_baseline --profile hard --seconds 60
```

It prints maps per minute, mean damage and stars, terminated vs. truncated
counts, and threshold success rates (>=25 percent, >=50 percent, and so
on). If a learned policy doesn't beat this by a wide margin, the run is
broken.

For rollout-throughput checks, run the same random policy across multiple
processes:

```
.venv/bin/python -B -m scripts.random_baseline_parallel --profile hard --seconds 60 --workers 8
```

This reports aggregate maps/min, decisions/sec, sim ticks/sec, and per-worker
episode counts. Use it to choose the worker count for hosted training.

### Step 2: maskable PPO

`sb3-contrib` ships `MaskablePPO`, which consumes `action_masks()` directly
and restricts the categorical distribution to legal actions. Install with
`pip install -e .[train]` and start from something like:

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from coc_env import CoCEnv

def make_env(profile, seed):
    def _f():
        env = CoCEnv(
            layout_profile=profile,
            army_composition={"barbarian": 40, "wall_breaker": 10},
        )
        env.reset(seed=seed)
        return env
    return _f

env = SubprocVecEnv([make_env("hard", seed=i) for i in range(8)])
model = MaskablePPO(
    MaskableMultiInputActorCriticPolicy,
    env,
    n_steps=512,
    batch_size=2048,
    learning_rate=3e-4,
    gamma=0.995,
    gae_lambda=0.95,
    ent_coef=0.01,
    tensorboard_log="./runs",
)
model.learn(total_timesteps=5_000_000)
```

Two reasons for the high `gamma`: episodes are long (up to 720 ticks, or
72 agent steps), and most of the reward signal lands at the tick a
building dies, not the tick the relevant deploy happened. The observation
is a `Dict`, which is why this needs the multi-input policy.

A curriculum tends to help. Train on `easy` until the agent reliably hits
two stars, then switch to `medium` or `hard` by passing
`options={"profile": "..."}` to `env.reset()`. The five preset profiles
work as a difficulty ladder.

### What the reward looks like

`score = stars + damage_pct` is not dense in time. Most of the damage
percent accrues at the tick a building hits zero hp. Two practical
consequences:

- The time-cost term (1e-4 per tick) discourages stalling but is small
  enough that it can't outweigh a single star.
- Value targets are noisy unless you train with enough parallel envs to
  see multiple terminal transitions per update. Eight envs is the floor.
  Sixteen is more comfortable.

### Things that have bitten me

- Don't add a shaping reward for picking an illegal action. The mask
  already removes them, so shaping just spreads bias.
- If you include wall breakers in the composition but the policy never
  picks them, check `info["army_remaining_by_kind"]` per episode. The
  second deploy layer is real but easy to ignore.
- In a custom feature extractor, condition every per-slot read on
  `buildings_present`. Padding slots and unrevealed traps are zeros, not
  missing values.

### Watching a policy play

Two viewers exist:

- Browser: `coc-viewer` (or `python viewer/server.py`), then open
  `http://127.0.0.1:5173/`. The UI can step ticks, run the bundled random
  agent, or accept manual deploys. The `/state` JSON endpoint is a
  reasonable scripting surface if you'd rather not go through Gymnasium.
- Pygame: `python scripts/hand_play.py` runs a hand-coded deploy schedule
  against the default layout. This is what I use when I am changing
  simulator internals.

The pytest suite covers slot stability, mask correctness, hidden-trap
reveal, the mortar blind spot, wall-breaker wiring, and the
`reward == score_delta - time_cost` invariant:

```
.venv/bin/python -m pytest -q
```

## Layout

```
coc/
  coc_env/
    entities.py       Building/Troop dataclasses, grid and tick constants
    simulator.py      Battle core: pathing, defences, traps
    env.py            Gymnasium wrapper, action mask
    generation.py     Procedural layouts, preset profiles
    layouts.py        Hand-authored default layout
    render_pygame.py  Local pygame renderer
  viewer/
    server.py         Flask JSON API
    index.html        Sprite-based browser UI
    img/              Sprite assets
  scripts/
    random_baseline.py           Random-policy evaluation harness
    random_baseline_parallel.py  Multi-process random-policy throughput harness
    bench_env.py                 Throughput benchmark for masks/tick/step
    hand_play.py                 Scripted-deploy pygame demo
  tests/                Pytest suite
```

## Install

```
python -m venv .venv
.venv/bin/pip install -e .[viewer,dev]
.venv/bin/pip install -e .[train]
```

Python 3.11+, numpy 2.x, gymnasium 1.x.

## Commands

| Task                    | Command |
| ----------------------- | ------- |
| Tests                   | `.venv/bin/python -m pytest -q` |
| Random baseline, 1 min  | `.venv/bin/python -m scripts.random_baseline --profile hard --seconds 60` |
| Parallel random baseline | `.venv/bin/python -B -m scripts.random_baseline_parallel --profile hard --seconds 60 --workers 8` |
| All profiles, 30 s total | `.venv/bin/python -m scripts.random_baseline --profile all --seconds 30` |
| Throughput              | `.venv/bin/python -m scripts.bench_env --profile hard --iterations 20000` |
| Browser viewer          | `.venv/bin/coc-viewer` then `http://127.0.0.1:5173/` |
| Pygame demo             | `.venv/bin/python scripts/hand_play.py` |
