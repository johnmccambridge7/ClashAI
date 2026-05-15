from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from coc_env.env import CoCEnv, WAIT_ACTION


def _rate(name: str, count: int, elapsed: float) -> str:
    return f"{name:<18} {count / max(elapsed, 1e-9):>12,.0f}/s  ({elapsed:.3f}s)"


def _time(count: int, fn: Callable[[], None]) -> float:
    start = time.perf_counter()
    for _ in range(count):
        fn()
    return time.perf_counter() - start


def bench_env(profile: str | None, seed: int, iterations: int) -> None:
    env = CoCEnv(layout_profile=profile)
    env.reset(seed=seed)

    mask_elapsed = _time(iterations, lambda: env.action_masks())
    print(_rate("action_masks", iterations, mask_elapsed))

    assert env.sim is not None
    tick_elapsed = _time(iterations, env.sim.tick)
    print(_rate("sim.tick", iterations, tick_elapsed))

    rng = np.random.default_rng(seed)

    def step_valid() -> None:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        action = int(rng.choice(valid)) if len(valid) else WAIT_ACTION
        terminated = truncated = False
        try:
            _, _, terminated, truncated, _ = env.step(action)
        finally:
            if terminated or truncated:
                env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    step_elapsed = _time(iterations, step_valid)
    print(_rate("env.step(valid)", iterations, step_elapsed))

    mask = env.action_masks()
    print(f"valid actions       {int(mask.sum()):>12,}/{len(mask):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    bench_env(profile=args.profile, seed=args.seed, iterations=args.iterations)


if __name__ == "__main__":
    main()
