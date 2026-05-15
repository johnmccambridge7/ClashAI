from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import numpy as np

from coc_env.env import CoCEnv
from coc_env.generation import PRESET_LAYOUT_PROFILES


@dataclass(frozen=True)
class EpisodeResult:
    profile: str
    seed: int
    damage_pct: float
    score: float
    stars: int
    steps: int
    ticks: int
    terminated: bool
    truncated: bool


def _profile_sequence(profile: str) -> list[str]:
    if profile == "all":
        return sorted(PRESET_LAYOUT_PROFILES)
    if profile not in PRESET_LAYOUT_PROFILES:
        known = ", ".join(["all", *sorted(PRESET_LAYOUT_PROFILES)])
        raise ValueError(f"unknown profile {profile!r}; expected one of: {known}")
    return [profile]


def run_time_budget(seconds: float = 60.0, seed: int = 42, profile: str = "hard") -> tuple[list[EpisodeResult], float]:
    profiles = _profile_sequence(profile)
    max_buildings = max(p.max_buildings for p in PRESET_LAYOUT_PROFILES.values())
    env = CoCEnv(layout_profile=profiles[0], max_buildings=max_buildings)
    rng = np.random.default_rng(seed)
    results: list[EpisodeResult] = []

    start = time.perf_counter()
    deadline = start + seconds
    episode = 0
    while time.perf_counter() < deadline:
        episode_seed = seed + episode
        episode_profile = profiles[episode % len(profiles)]
        env.reset(seed=episode_seed, options={"profile": episode_profile})
        terminated = truncated = False
        steps = 0
        info = {"damage_pct": 0.0, "score": 0.0, "stars": 0, "ticks_elapsed": 0}
        while not (terminated or truncated):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
        results.append(EpisodeResult(
            profile=episode_profile,
            seed=episode_seed,
            damage_pct=float(info["damage_pct"]),
            score=float(info["score"]),
            stars=int(info["stars"]),
            steps=steps,
            ticks=int(info["ticks_elapsed"]),
            terminated=terminated,
            truncated=truncated,
        ))
        episode += 1

    return results, time.perf_counter() - start


def run(n_episodes: int = 200, seed: int = 42, profile: str = "hard") -> np.ndarray:
    profiles = _profile_sequence(profile)
    max_buildings = max(p.max_buildings for p in PRESET_LAYOUT_PROFILES.values())
    env = CoCEnv(layout_profile=profiles[0], max_buildings=max_buildings)
    rng = np.random.default_rng(seed)
    damage: list[float] = []
    for ep in range(n_episodes):
        episode_profile = profiles[ep % len(profiles)]
        env.reset(seed=seed + ep, options={"profile": episode_profile})
        terminated = truncated = False
        info = {"damage_pct": 0.0}
        while not (terminated or truncated):
            valid = np.flatnonzero(env.action_masks())
            action = int(rng.choice(valid))
            _, _, terminated, truncated, info = env.step(action)
        damage.append(float(info["damage_pct"]))
    return np.array(damage)


def summarize(results: list[EpisodeResult], elapsed: float) -> str:
    if not results:
        return f"N=0 elapsed={elapsed:.2f}s"
    damage = np.array([r.damage_pct for r in results], dtype=np.float64)
    scores = np.array([r.score for r in results], dtype=np.float64)
    stars = np.array([r.stars for r in results], dtype=np.int64)
    steps = np.array([r.steps for r in results], dtype=np.float64)
    ticks = np.array([r.ticks for r in results], dtype=np.float64)
    terminated = sum(r.terminated for r in results)
    truncated = sum(r.truncated for r in results)
    maps_per_min = len(results) / max(elapsed, 1e-9) * 60.0
    decisions_per_sec = steps.sum() / max(elapsed, 1e-9)
    ticks_per_sec = ticks.sum() / max(elapsed, 1e-9)
    lines = [
        f"N={len(results)} elapsed={elapsed:.2f}s maps/min={maps_per_min:.1f} "
        f"decisions/sec={decisions_per_sec:.1f} sim_ticks/sec={ticks_per_sec:.1f}",
        f"damage mean={damage.mean():.3f} median={np.median(damage):.3f} "
        f"min={damage.min():.3f} max={damage.max():.3f} std={damage.std():.3f}",
        f"score  mean={scores.mean():.3f} median={np.median(scores):.3f} "
        f"min={scores.min():.3f} max={scores.max():.3f}",
        f"stars  mean={stars.mean():.3f} dist="
        + " ".join(f"{s}:{int((stars == s).sum())}" for s in range(4)),
        f"steps  mean={steps.mean():.1f} median={np.median(steps):.1f} "
        f"ticks_mean={ticks.mean():.1f}",
        f"ends   terminated={terminated} truncated={truncated}",
    ]
    lines.append(
        "damage thresholds "
        + " ".join(f">={lo}%:{(damage * 100 >= lo).mean() * 100:4.1f}%" for lo in (25, 50, 75, 90, 100))
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a uniformly random legal-action policy.")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="hard", help="Preset profile name, or 'all'.")
    args = parser.parse_args()

    results, elapsed = run_time_budget(seconds=args.seconds, seed=args.seed, profile=args.profile)
    print(f"profile={args.profile} seed={args.seed}")
    print(summarize(results, elapsed))


if __name__ == "__main__":
    main()
