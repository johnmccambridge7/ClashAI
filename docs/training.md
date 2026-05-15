# How the RL side works

This is a walk-through of the reinforcement learning problem that `coc_env`
poses, and the training recipe we use to solve it. The math is written from
first principles so you can follow it without prior RL background, but it
moves fast.

Notation: $s$ for state, $a$ for action, $r$ for reward, $\pi$ for policy,
$V$ for value function. Subscripts $t$ are timesteps. Hat ($\hat{\cdot}$)
means empirical estimate. $\theta$ are the policy network's parameters.

## 1. The problem as a Markov decision process

A Markov decision process (MDP) is the tuple $(\mathcal{S}, \mathcal{A}, P,
R, \gamma)$:

- $\mathcal{S}$: the set of possible states.
- $\mathcal{A}$: the set of possible actions.
- $P(s' \mid s, a)$: the transition kernel. Given state $s$ and action $a$,
  the probability of landing in state $s'$.
- $R(s, a)$: the reward function.
- $\gamma \in [0, 1]$: the discount factor.

The agent picks actions $a_t \sim \pi(\cdot \mid s_t)$. The environment
returns $r_t$ and $s_{t+1}$. The agent's job is to find a policy that
maximises the expected discounted return:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ is a trajectory and $T$
is the (random) episode length.

In our case:

- $\mathcal{S}$ is the set of all simulator states (building list, troop
  list, army counts, tick count).
- $\mathcal{A}$ is `Discrete(GRID_SIZE * GRID_SIZE * len(troops) + 1)` =
  $\{0, 1, \dots, 3872\}$. Index $a$ decodes to "deploy troop kind $k$ at
  cell $(x, y)$" or, if $a = 3872$, "do nothing this step".
- $P$ is deterministic given the seed used at `env.reset()`. The simulator
  has no stochastic transitions once initialised. Stochasticity comes from
  the policy and from layout sampling at reset.
- $R(s, a)$ is the per-step reward, defined below.
- $\gamma$ is a hyperparameter, set to $0.995$ in our default recipe.

Each $\texttt{env.step()}$ advances `DECISION_INTERVAL = 10` simulator ticks.
So one "agent step" $t$ corresponds to ten game ticks. An episode lasts at
most $720 / 10 = 72$ agent steps.

## 2. The state, as the agent sees it

The simulator's full state is large, but the observation handed to the
policy is a compact `Dict` of fixed-shape arrays. With $N_B$ building slots
and $N_T$ troop slots, the observation $s_t$ contains:

| Group       | Key                       | Shape           | Notes |
| ----------- | ------------------------- | --------------- | ----- |
| Buildings   | `buildings_present`       | $(N_B,)$        | 1 if slot is observable |
|             | `buildings_alive`         | $(N_B,)$        | 1 if hp > 0 |
|             | `buildings_kind`          | $(N_B,)$        | int in `[0, K_B)` |
|             | `buildings_hp`            | $(N_B,)$        | $\in [0, 1]$ |
|             | `buildings_pos`           | $(N_B, 2)$      | $\in [0, 1]^2$ |
|             | `buildings_size`          | $(N_B,)$        | $\in [0, 1]$ |
|             | `buildings_is_defense`    | $(N_B,)$        | binary |
|             | `buildings_blocks_move`   | $(N_B,)$        | binary |
|             | `buildings_dps`           | $(N_B,)$        | $\in [0, 1]$, normalised by 30 |
|             | `buildings_attack_range`  | $(N_B,)$        | $\in [0, 1]$, normalised by grid |
|             | `buildings_min_attack_range` | $(N_B,)$     | same |
|             | `buildings_splash_radius` | $(N_B,)$        | same |
|             | `buildings_cooldown`      | $(N_B,)$        | $\in [0, 1]$, seconds / 5 |
|             | `buildings_is_trap`       | $(N_B,)$        | binary |
| Troops      | `troops_alive`            | $(N_T,)$        | binary |
|             | `troops_kind`             | $(N_T,)$        | int in `[0, K_T)` |
|             | `troops_hp`               | $(N_T,)$        | $\in [0, 1]$ |
|             | `troops_pos`              | $(N_T, 2)$      | $\in [0, 1]^2$ |
|             | `troops_target`           | $(N_T,)$        | building slot fraction, or $-1$ |
| Globals     | `army_remaining`          | $(1,)$          | $\in [0, 1]$ |
|             | `time_remaining`          | $(1,)$          | $\in [0, 1]$ |

The slot invariant matters for credit assignment. Slot $i$ refers to the
same building (or troop) for the whole episode. A destroyed building keeps
`present=1, alive=0, hp=0`. Hidden traps that have not triggered yet keep
`present=0`. Padding slots, when `max_buildings` exceeds the actual layout
size, also keep `present=0`. So the network can attend to "the building
that used to be at slot 4" and read off useful information about why
nothing is there anymore.

## 3. The action space and masking

Actions decode as

$$
a = k \cdot N_C + y \cdot W + x
\quad\text{for}\quad
k \in [0, K_T),\ x \in [0, W),\ y \in [0, H)
$$

with $W = H = 44$, $N_C = W \cdot H = 1936$, $K_T = 2$ (barbarian, wall
breaker), plus a single wait action at $a = K_T \cdot N_C = 3872$. The
total action count is $|\mathcal{A}| = K_T \cdot N_C + 1 = 3873$.

Most of those actions are illegal at any given moment. A cell is illegal if
it overlaps a building footprint or sits inside a live defence's attack
ring (between `min_attack_range` and `attack_range`). A troop kind is
illegal if you have already used all of that kind. The wait action is
always legal.

Let $m_t \in \{0, 1\}^{|\mathcal{A}|}$ be the mask at step $t$, with
$m_{t,a} = 1$ iff action $a$ is legal. The masked policy is

$$
\pi_\theta(a \mid s_t, m_t) = \frac{m_{t,a} \cdot \exp(\ell_a(s_t))}{\sum_{a'} m_{t,a'} \cdot \exp(\ell_{a'}(s_t))}
$$

where $\ell_a(s_t)$ is the logit the network emits for action $a$. Illegal
actions receive zero probability, and the remaining mass is renormalised
over the legal set $\mathcal{A}_t = \{a : m_{t,a} = 1\}$. The entropy used
inside the loss is also computed over $\mathcal{A}_t$ only:

$$
H[\pi_\theta(\cdot \mid s_t, m_t)] = -\sum_{a \in \mathcal{A}_t} \pi_\theta(a \mid s_t, m_t) \log \pi_\theta(a \mid s_t, m_t).
$$

This is what `sb3-contrib`'s `MaskablePPO` does internally. The agent never
sees, scores, or gets gradients for illegal actions, so the policy doesn't
have to learn to avoid them.

## 4. The reward

Let $D_t \in [0, 1]$ be the fraction of original total hp destroyed across
scored buildings (so walls and bombs are excluded). Let $S_t \in \{0,1,2,3\}$
be the star count, with

$$
S_t = \mathbb{1}[D_t \geq 0.5] + \mathbb{1}[\text{townhall destroyed}] + \mathbb{1}[D_t \geq 1].
$$

The scalar score the agent is rewarded for is

$$
\text{score}_t = S_t + D_t \in [0, 4].
$$

The reward at step $t$ is the score delta, minus a small time cost
proportional to the number of simulator ticks the step advanced:

$$
r_t = \text{score}_t - \text{score}_{t-1} - c \cdot \Delta_t
$$

with $c = 10^{-4}$ and $\Delta_t \leq 10$. Because $r_t$ telescopes, the
sum over an episode is

$$
\sum_{t=0}^{T} r_t = \text{score}_T - c \cdot \sum_t \Delta_t.
$$

That is: the undiscounted return equals the final score minus a tiny time
penalty (at most $720 \cdot 10^{-4} = 0.072$). The time term keeps the
policy from stalling but is small enough not to outweigh a single star.

## 5. What we want to learn

The optimal value of being in state $s$ under policy $\pi$ is

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{l=0}^{\infty} \gamma^l r_{t+l} \;\middle|\; s_t = s\right].
$$

The optimal action-value is

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{l=0}^{\infty} \gamma^l r_{t+l} \;\middle|\; s_t = s, a_t = a\right].
$$

The advantage is how much better a particular action is than the average:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s).
$$

Positive advantage means "this action did better than baseline"; negative
means worse. We want to push the policy toward positive-advantage actions
and away from negative-advantage actions. That is what policy gradient
methods do.

## 6. Policy gradient, from REINFORCE to PPO

### REINFORCE

The policy gradient theorem says

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]
$$

where $G_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l}$ is the discounted return
from step $t$ onward. So the simplest policy gradient algorithm
(REINFORCE) is "sample a trajectory, compute returns, take a gradient step
in the direction of $\log \pi$ weighted by $G_t$."

This works, but $G_t$ is very high variance: it depends on every later
action the policy took, not just $a_t$.

### Actor-critic

Subtracting a baseline $b(s_t)$ that depends only on the state doesn't bias
the gradient (because $\mathbb{E}[\nabla \log \pi \cdot b(s)] = 0$). The
best variance-reducing baseline is $V^\pi(s_t)$. Replacing $G_t$ with
$G_t - V(s_t)$, and then approximating that with an empirical advantage
$\hat{A}_t$, gives the actor-critic gradient:

$$
\nabla_\theta J \approx \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t\right].
$$

The "critic" is a learned $V_\phi(s)$, trained by regression against
observed returns.

### Generalised advantage estimation (GAE)

The bias-variance tradeoff for $\hat{A}_t$ is parameterised by $\lambda \in
[0, 1]$. Define the one-step TD residual

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

GAE then mixes residuals at every horizon:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}.
$$

- $\lambda = 0$: $\hat{A}_t = \delta_t$, low variance, high bias (relies on
  $V_\phi$ being good).
- $\lambda = 1$: $\hat{A}_t = G_t - V_\phi(s_t)$, high variance, low bias.

We use $\lambda = 0.95$ as the standard middle ground.

### PPO: the trust region issue

Stepping in the direction of the gradient with a large learning rate can
move the policy so far that the data used to compute the gradient is no
longer representative. Trust region methods cap the per-update change in
policy. PPO's clipped surrogate objective is a cheap, practical version of
this. Define the importance ratio

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

where $\theta_{\text{old}}$ is the policy at the time the data was collected
(once per rollout). The PPO surrogate is

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\Big(\rho_t(\theta) \hat{A}_t,\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_t\Big)\right].
$$

The clip says: if the new policy disagrees with the old policy by more
than a factor of $1 \pm \epsilon$ on an action that had positive
advantage, the gradient flattens, so the optimiser stops pushing. Same on
the other side for negative advantage. We use $\epsilon = 0.2$.

## 7. The full loss

The complete loss minimised at each minibatch is

$$
L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_v L^{\text{VF}}(\phi) - c_e L^{\text{ENT}}(\theta).
$$

The three terms:

- **Policy term.** $-L^{\text{CLIP}}$ is the negated PPO surrogate. We
  minimise the negative because gradient descent works on a loss, not a
  reward.

- **Value term.** A standard regression loss on the value head:

  $$
  L^{\text{VF}}(\phi) = \hat{\mathbb{E}}_t\left[(V_\phi(s_t) - \hat{R}_t)^2\right]
  $$

  where $\hat{R}_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$ is the target.
  Coefficient $c_v = 0.5$ by default.

- **Entropy bonus.** Encourages exploration by rewarding high-entropy
  policies:

  $$
  L^{\text{ENT}}(\theta) = \hat{\mathbb{E}}_t\big[H[\pi_\theta(\cdot \mid s_t, m_t)]\big].
  $$

  The entropy is computed over the masked action set (Section 3).
  Coefficient $c_e$ starts at $0.01$.

## 8. Training regime

The schedule we run is standard PPO with a few choices specific to this
env.

**Vectorised rollouts.** Use eight to sixteen `SubprocVecEnv` workers, each
running its own copy of `CoCEnv`. Every worker rolls out for
`n_steps = 512` agent steps, giving a rollout buffer of $8 \cdot 512 =
4096$ transitions per update at the floor and $16 \cdot 512 = 8192$ at
sixteen workers.

**Per-update optimisation.** Each rollout is broken into minibatches of
$2048$ transitions and passed through the loss $n_{\text{epochs}} = 10$
times. So each transition is reused ten times before being thrown away,
which is what the PPO clip lets us get away with safely.

**Discounting.** $\gamma = 0.995$. Episodes can run 72 agent steps with
most reward arriving late, so we want $\gamma^{72} \approx 0.70$, not
$\gamma^{72} \approx 0$.

**Advantage normalisation.** $\hat{A}_t$ values are standardised
(zero-mean, unit-variance) per minibatch before being used in
$L^{\text{CLIP}}$. This is on by default in `MaskablePPO` and removes a
common cause of unstable updates.

**Learning rate.** $3 \cdot 10^{-4}$ with the Adam optimiser. A linear
schedule down to $0$ over the run helps once the policy stops improving.

**Curriculum.** Start on `layout_profile="easy"`. When the mean star count
over the last $N$ episodes reaches $\approx 2$, advance to `medium`, then
`hard`. The five preset profiles
(`easy`, `medium`, `hard`, `resource_bait`, `split`) form a difficulty
ladder. Use `env.reset(options={"profile": "..."})` to switch.

**Total budget.** A reasonable first run is $5 \cdot 10^6$ environment
steps. With eight workers and the simulator's throughput, that is on the
order of an hour on a recent CPU; with sixteen, half that.

### One full training step, end to end

```
for update in range(num_updates):
    # 1. Roll out fresh data with theta_old.
    rollout = collect_rollouts(env, policy_theta_old, n_steps=512)

    # 2. Compute advantages with GAE(gamma=0.995, lambda=0.95).
    advantages = compute_gae(rollout.rewards, rollout.values, gamma, lam)
    returns = advantages + rollout.values

    # 3. Normalise advantages per minibatch.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Train for n_epochs over shuffled minibatches.
    for epoch in range(n_epochs):
        for batch in shuffled_minibatches(rollout, batch_size=2048):
            # Importance ratio under the new theta.
            ratio = exp(log_pi_theta(a|s, m) - log_pi_theta_old(a|s, m))

            # PPO clipped surrogate.
            unclipped = ratio * batch.advantages
            clipped   = clip(ratio, 1-eps, 1+eps) * batch.advantages
            L_clip    = mean(min(unclipped, clipped))

            # Value regression.
            L_vf = mean((V_phi(s) - batch.returns) ** 2)

            # Masked entropy bonus.
            L_ent = mean(masked_entropy(pi_theta, s, m))

            loss = -L_clip + c_v * L_vf - c_e * L_ent

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, max_norm=0.5)
            optimizer.step()
```

## 9. What to watch during training

The diagnostic signals that matter most in this env:

- **`rollout/ep_rew_mean`.** Mean undiscounted episode reward. Should
  climb from around the random baseline (run
  `scripts/random_baseline.py` for the number on your chosen profile) up
  toward $3.0$+.
- **Mean star count.** A more interpretable number than reward. Pulled from
  `info["stars"]` at terminal transitions.
- **`train/clip_fraction`.** The fraction of samples where the importance
  ratio was clipped. If this is consistently above $\sim 0.3$, the
  learning rate is too high or `n_epochs` is too high.
- **`train/approx_kl`.** Approximate KL divergence between old and new
  policy. Spikes here precede instability. A common trick is to early-stop
  the inner epoch loop when $\hat{\text{KL}} > 0.02$.
- **`train/explained_variance`.** $1 - \frac{\text{Var}(\hat{R}_t -
  V_\phi(s_t))}{\text{Var}(\hat{R}_t)}$. If this is near zero or negative,
  the critic is not learning. If close to $1$, the critic is fitting well.

If `ep_rew_mean` is flat but `entropy` is high, the policy is exploring
without committing. If `entropy` collapses early but reward is still low,
the policy got trapped. Adjust `ent_coef` in either case.

## 10. Evaluation

For a final number, run:

```
.venv/bin/python -m scripts.random_baseline --profile hard --seconds 60
```

to get the random baseline, then evaluate the trained policy on the same
profile and seed range. Useful summary statistics:

- Mean and median damage percent.
- Star distribution (how many runs got 0, 1, 2, 3 stars).
- Threshold success rates: $P(D \geq 0.5)$, $P(D \geq 0.9)$, $P(S \geq 2)$.
- Time-to-terminal among non-truncated runs (a fast clear is worth more
  than a slow one once you cap the reward at $3.0$).

The pytest suite (`tests/`) is the regression gate for simulator changes,
not for policy quality. Treat the random baseline as the policy-quality
floor.

## Quick reference

| Symbol | Meaning | Default |
| ------ | ------- | ------- |
| $\gamma$ | Discount factor | $0.995$ |
| $\lambda$ | GAE bias-variance knob | $0.95$ |
| $\epsilon$ | PPO clip range | $0.2$ |
| $c_v$ | Value loss coefficient | $0.5$ |
| $c_e$ | Entropy bonus coefficient | $0.01$ |
| `n_steps` | Rollout length per worker | $512$ |
| `batch_size` | Minibatch size | $2048$ |
| `n_epochs` | Optimisation passes per rollout | $10$ |
| Workers | Number of parallel envs | $8$ to $16$ |
| Total steps | Environment steps for a first run | $5 \cdot 10^6$ |
| Learning rate | Adam, linear schedule to $0$ | $3 \cdot 10^{-4}$ |
