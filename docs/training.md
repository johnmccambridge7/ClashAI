# Training A Clash Agent With Reinforcement Learning

This document explains the training setup from first principles, then maps
each concept to the current `coc_env` implementation. The goal is not just
to run PPO; it is to define the problem carefully enough that reward design,
action masking, curriculum, and evaluation all line up with the simulator.

Notation: `s` is state, `o` is observation, `a` is action, `r` is reward,
`pi` is the policy, `V` is the value function, and `gamma` is the discount
factor. Subscript `t` means "at decision step `t`".

## 1. Problem Formulation

A reinforcement learning task is usually modelled as a Markov decision
process (MDP):

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- `S`: all possible simulator states.
- `A`: all possible actions.
- `P(s' | s, a)`: the transition rule from state `s` to next state `s'`.
- `R(s, a, s')`: the reward produced by that transition.
- `gamma`: how much future reward matters relative to immediate reward.

The agent samples an action from its policy:

$$
a_t \sim \pi(\cdot \mid o_t)
$$

The environment applies that action, advances the simulator, and returns a
new observation and reward. The objective is to maximise expected discounted
return:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

where `T` is the episode length.

In this environment:

- The full simulator state is Markov: buildings, troops, hidden traps, army
  counts, cooldowns, and tick count.
- The policy receives a compact observation of that state. Because hidden
  traps are not present in the observation until revealed, the agent's view
  is technically partially observable. The simulator state is still Markov;
  the policy must learn to act under incomplete information.
- Transitions are deterministic once the layout and seed are fixed.
  Stochasticity comes from layout sampling at reset and from the policy's
  sampled actions.
- One `env.step(action)` advances `DECISION_INTERVAL = 10` simulator ticks,
  unless the episode ends earlier.
- With `MAX_TICKS = 720`, an episode lasts at most `72` agent decisions.

## 2. Observation

The observation is a fixed-shape `gymnasium.spaces.Dict`. Fixed shapes matter
because neural networks need the same tensor structure every step, even
though each generated base may have a different number of buildings.

With `N_B` building slots and `N_T` troop slots, the observation contains:

| Group | Key | Shape | Meaning |
| --- | --- | --- | --- |
| Buildings | `buildings_present` | `(N_B,)` | Slot contains a visible building |
| | `buildings_alive` | `(N_B,)` | Visible building has hp > 0 |
| | `buildings_kind` | `(N_B,)` | Integer building type |
| | `buildings_hp` | `(N_B,)` | Hp fraction in `[0, 1]` |
| | `buildings_pos` | `(N_B, 2)` | Normalised top-left position |
| | `buildings_size` | `(N_B,)` | Normalised footprint size |
| | `buildings_is_defense` | `(N_B,)` | Cannon, wizard tower, mortar, etc. |
| | `buildings_blocks_move` | `(N_B,)` | Blocks troop movement |
| | `buildings_dps` | `(N_B,)` | Damage per second, normalised |
| | `buildings_attack_range` | `(N_B,)` | Maximum range, normalised |
| | `buildings_min_attack_range` | `(N_B,)` | Dead-zone radius, normalised |
| | `buildings_splash_radius` | `(N_B,)` | Splash radius, normalised |
| | `buildings_cooldown` | `(N_B,)` | Attack cooldown, normalised |
| | `buildings_is_trap` | `(N_B,)` | Trap flag |
| Troops | `troops_alive` | `(N_T,)` | Slot contains a live troop |
| | `troops_kind` | `(N_T,)` | Integer troop type |
| | `troops_hp` | `(N_T,)` | Hp fraction in `[0, 1]` |
| | `troops_pos` | `(N_T, 2)` | Normalised position |
| | `troops_target` | `(N_T,)` | Target building slot fraction, or `-1` |
| Globals | `army_remaining` | `(1,)` | Fraction of undeployed army left |
| | `time_remaining` | `(1,)` | Fraction of episode time left |

Slot identity is stable for the whole episode. If building slot `i` holds
the townhall at reset, slot `i` continues to refer to that townhall after it
is destroyed. A destroyed building has `present=1`, `alive=0`, and `hp=0`.
An unrevealed hidden trap has `present=0`. Padding slots also have
`present=0`.

That invariant is important for credit assignment: the policy can learn that
"the destroyed building in slot 4" was previously a defence, storage, or
townhall instead of seeing the observation list reshuffle after every kill.

## 3. Actions And Masks

The action space is discrete:

```text
Discrete(N_DEPLOY_ACTIONS + 1)
```

The deploy actions are one layer per deployable troop kind:

$$
a = k \cdot N_C + y \cdot W + x
$$

where:

- `k` is the index into `DEPLOY_TROOP_KINDS`.
- `W = H = GRID_SIZE = 44`.
- `N_C = W * H = 1936` deploy cells.
- Current `DEPLOY_TROOP_KINDS` are `barbarian` and `wall_breaker`.

So the current action count is:

```text
44 * 44 * 2 + 1 = 3873
```

Actions `0..1935` deploy barbarians. Actions `1936..3871` deploy wall
breakers. `WAIT_ACTION = 3872` is the explicit no-op and is always legal.

Most actions are illegal at any given step. A deployment is illegal if:

- the cell overlaps a live blocking building footprint;
- the cell is inside a live defence's attack ring;
- the selected troop kind has no remaining units;
- the episode is already done.

`env.action_masks()` returns a boolean vector with one entry per action.
The mask is not a reward signal. It defines the valid action set.

For policy learning, the logits are masked before the categorical
distribution is formed:

$$
\pi_\theta(a \mid o_t, m_t) =
\frac{m_{t,a} \exp(\ell_a(o_t))}
{\sum_{a'} m_{t,a'} \exp(\ell_{a'}(o_t))}
$$

Illegal actions get probability zero. Legal actions are renormalised. This
is why we should use `sb3-contrib`'s `MaskablePPO`, not vanilla PPO: the
policy should optimise over legal deploy choices, not waste capacity learning
that impossible actions fail.

The action mask must also be used during evaluation. Otherwise the trained
policy and evaluation policy are not solving the same problem.

## 4. Reward

The simulator score is:

$$
\text{score}_t = S_t + D_t
$$

where `D_t` is the fraction of original scored building hp destroyed and
`S_t` is the star count:

$$
S_t =
\mathbf{1}[D_t \ge 0.5]
+ \mathbf{1}[\text{townhall destroyed}]
+ \mathbf{1}[D_t \ge 1.0]
$$

Walls and bombs are not scored buildings. They can matter strategically, but
they do not directly add damage percentage or stars.

The per-step reward is the score delta minus a small time cost:

$$
r_t = \text{score}_t - \text{score}_{t-1} - c \Delta_t
$$

with `c = 1e-4` and `Delta_t` equal to the number of simulator ticks advanced
by the step.

This reward telescopes over an episode:

$$
\sum_t r_t = \text{score}_T - c \sum_t \Delta_t
$$

The final score is in `[0, 4]`: up to `3` stars plus up to `1.0` damage. The
maximum time penalty is `720 * 1e-4 = 0.072`, so speed matters, but it cannot
outweigh a meaningful score improvement.

This is intentionally sparse-ish. The agent gets reward when buildings die
and stars are earned, not for every pixel of movement. That keeps the
objective aligned with the game outcome instead of teaching a hand-shaped
heuristic.

## 5. What The Policy Learns

The value function estimates how much reward remains from an observation:

$$
V^\pi(o_t) =
\mathbb{E}_\pi\left[\sum_{l=0}^{T-t} \gamma^l r_{t+l}
\mid o_t\right]
$$

The action value estimates the same return after choosing a particular
action:

$$
Q^\pi(o_t, a_t) =
\mathbb{E}_\pi\left[\sum_{l=0}^{T-t} \gamma^l r_{t+l}
\mid o_t, a_t\right]
$$

The advantage is:

$$
A^\pi(o_t, a_t) = Q^\pi(o_t, a_t) - V^\pi(o_t)
$$

Positive advantage means the action performed better than the policy's
normal expectation for that observation. Negative advantage means it
performed worse. Policy-gradient methods increase the probability of
positive-advantage actions and decrease the probability of negative-advantage
actions.

## 6. From Policy Gradient To PPO

The basic policy-gradient estimator is:

$$
\nabla_\theta J(\pi_\theta) \approx
\hat{\mathbb{E}}_t[
\nabla_\theta \log \pi_\theta(a_t \mid o_t, m_t) \hat{A}_t
]
$$

The mask appears inside the policy distribution. The logged probability must
be the probability after illegal actions have been removed.

PPO improves the basic estimator by limiting how far the policy can move
from the policy that collected the data. Define:

$$
\rho_t(\theta) =
\frac{\pi_\theta(a_t \mid o_t, m_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid o_t, m_t)}
$$

PPO's clipped policy objective is:

$$
L^{\text{CLIP}}(\theta) =
\hat{\mathbb{E}}_t\left[
\min\left(
\rho_t(\theta)\hat{A}_t,
\text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
$$

We use `epsilon = 0.2` as the default clip range.

## 7. Advantage Estimation

The critic learns `V(o_t)`. Generalised advantage estimation (GAE) uses the
critic to trade bias against variance.

For a true terminal transition, there is no next-state value. With
`done_t = 1` for terminal and `0` otherwise, the one-step residual is:

$$
\delta_t =
r_t + \gamma (1 - done_t) V_\phi(o_{t+1}) - V_\phi(o_t)
$$

GAE then mixes residuals over multiple horizons:

$$
\hat{A}_t =
\sum_{l=0}^{T-t}(\gamma\lambda)^l \delta_{t+l}
$$

- `lambda = 0` gives low-variance, high-bias one-step estimates.
- `lambda = 1` approaches Monte Carlo returns with lower bias and higher
  variance.

We use `gae_lambda = 0.95`.

For time-limit truncations, be consistent with the training library's
timeout handling. The key rule is simple: do not bootstrap across a true
terminal state where the attack has actually ended.

## 8. Optimisation Loss

PPO trains a shared policy/value network with this loss:

$$
L(\theta, \phi) =
-L^{\text{CLIP}}(\theta)
+ c_v L^{\text{VF}}(\phi)
- c_e L^{\text{ENT}}(\theta)
$$

The terms are:

- Policy loss: the negative clipped PPO objective.
- Value loss: mean squared error between `V(o_t)` and the return target.
- Entropy bonus: masked categorical entropy over legal actions only.

Default coefficients:

- `vf_coef = 0.5`
- `ent_coef = 0.01`
- `max_grad_norm = 0.5`

The entropy term is important early because there are many legal deployment
choices. If entropy collapses before reward improves, the policy has become
too deterministic too soon.

## 9. Training Regime

Use `MaskablePPO` with `MaskableMultiInputActorCriticPolicy`. The observation
is a dict, so a multi-input policy is required. The action space is large and
masked, so maskable PPO is required.

A first serious run should use:

| Setting | Starting value |
| --- | --- |
| Algorithm | `MaskablePPO` |
| Policy | `MaskableMultiInputActorCriticPolicy` |
| Workers | `8` to `16` |
| `n_steps` | `512` per worker |
| `batch_size` | `2048` |
| `n_epochs` | `10` |
| `gamma` | `0.995` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `learning_rate` | `3e-4`, preferably linearly decayed |
| First budget | `5_000_000` environment steps |

`n_steps = 512` is longer than one episode. That is expected: each worker
collects several attacks into one rollout buffer before PPO updates.

`gamma = 0.995` is high because rewards can arrive many decisions after the
deployment that caused them. Across the full 72-step horizon,
`0.995 ** 72 ~= 0.70`, so late building kills still receive meaningful
credit.

Advantage normalisation should happen over the rollout buffer before
minibatch optimisation. This is the default in PPO implementations such as
Stable Baselines.

Use local throughput measurements instead of assuming wall-clock training
time:

```bash
.venv/bin/python -m scripts.bench_env --profile hard --iterations 20000
```

## 10. Curriculum

Start with a profile that lets the policy discover the basic causal chain:
deploy troops, destroy buildings, earn reward. Then increase difficulty.

A practical curriculum:

1. Train on `easy` until mean stars over a recent evaluation window is near
   `2`.
2. Move to `medium`.
3. Move to `hard`.
4. Mix in `resource_bait` and `split` as challenge archetypes once the base
   policy is competent.
5. Finish on a sampled mixture of profiles so the policy does not overfit to
   one layout family.

`easy`, `medium`, and `hard` are the main difficulty ladder.
`resource_bait` and `split` are not just harder versions of the same task;
they test different deployment priorities and pathing behaviour.

Keep evaluation seeds separate from training seeds. Otherwise the policy may
look better than it is because it has adapted to the generator's repeated
layouts.

## 11. End-To-End PPO Loop

```text
for update in range(num_updates):
    rollout = collect_rollouts(
        envs,
        policy=pi_theta_old,
        masks=env.action_masks(),
        n_steps=512,
    )

    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.true_terminals,
        gamma=0.995,
        lambda=0.95,
    )

    advantages = normalize(advantages)

    for epoch in range(n_epochs):
        for batch in shuffled_minibatches(rollout, batch_size=2048):
            logp_new = masked_log_prob(pi_theta, batch.obs, batch.masks, batch.actions)
            ratio = exp(logp_new - batch.logp_old)

            unclipped = ratio * batch.advantages
            clipped = clip(ratio, 1 - 0.2, 1 + 0.2) * batch.advantages
            policy_loss = -mean(min(unclipped, clipped))

            value_loss = mean((V_phi(batch.obs) - batch.returns) ** 2)
            entropy = mean(masked_entropy(pi_theta, batch.obs, batch.masks))

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, 0.5)
            optimizer.step()
```

The important implementation detail is that rollout collection, log
probabilities, entropy, and evaluation all use the same action mask
semantics.

## 12. Diagnostics

Track these first:

- `rollout/ep_rew_mean`: should beat the random legal-action baseline.
- Mean stars: easier to reason about than reward.
- Mean damage percent: shows progress even before stars improve.
- Star distribution: `0`, `1`, `2`, and `3` star rates.
- `train/entropy_loss` or entropy: detects premature collapse.
- `train/clip_fraction`: if consistently above about `0.3`, updates are too
  aggressive.
- `train/approx_kl`: spikes usually precede instability.
- `train/explained_variance`: near or below `0` means the critic is not
  explaining returns.

If reward is flat and entropy remains high, exploration is not turning into
useful behaviour. If entropy collapses early and reward is poor, reduce the
learning rate, increase entropy pressure, or slow the curriculum.

## 13. Evaluation

First measure the random legal-action floor:

```bash
.venv/bin/python -m scripts.random_baseline --profile hard --seconds 60
```

Evaluate trained policies on the same profile set and a fixed holdout seed
range. Report at least:

- mean and median damage percent;
- mean score;
- star distribution;
- `P(damage >= 50%)`, `P(damage >= 90%)`, and `P(stars >= 2)`;
- terminated vs truncated counts;
- average ticks used.

Speed only matters through the time penalty. A faster clear is slightly
better than an equally scoring slower clear, but the total speed bonus is
bounded by `0.072` over the whole attack.

The pytest suite is the simulator regression gate. It proves mechanics are
stable; it does not prove policy quality. Policy quality is measured by
holdout evaluation against the random baseline and previous trained
checkpoints.
