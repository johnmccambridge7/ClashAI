from __future__ import annotations

import numpy as np

from coc_env.env import CoCEnv


def run(n_episodes: int = 200, seed: int = 42) -> np.ndarray:
    env = CoCEnv()
    rng = np.random.default_rng(seed)
    results: list[float] = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = int(rng.choice(valid))
            _, _, terminated, truncated, info = env.step(action)
        results.append(float(info["damage_pct"]))
    return np.array(results)


if __name__ == "__main__":
    arr = run(n_episodes=200, seed=42)
    print(
        f"N={len(arr)}  "
        f"mean={arr.mean():.3f}  "
        f"median={np.median(arr):.3f}  "
        f"min={arr.min():.3f}  "
        f"max={arr.max():.3f}  "
        f"std={arr.std():.3f}"
    )
    print("Damage% thresholds: ", end="")
    for lo in (0, 25, 50, 75, 90, 100):
        frac = (arr * 100 >= lo).mean() * 100
        print(f">={lo}%: {frac:5.1f}%  ", end="")
    print()
