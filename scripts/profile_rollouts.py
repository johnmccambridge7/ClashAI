from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import io
from pathlib import Path
import pstats
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from coc_env.env import CoCEnv
from scripts.random_baseline import EpisodeResult, summarize
from scripts.train_maskable_ppo import (
    DEFAULT_ARMY_COMPOSITION,
    DEFAULT_EVAL_SEED_START,
    import_training_deps,
    max_preset_buildings,
    parse_army_composition,
    profile_sequence,
    resolve_model_path,
    result_metrics,
    write_json,
)


def run_rollouts(
    *,
    profiles: list[str],
    episodes: int,
    seed_start: int,
    max_buildings: int,
    army_composition: dict[str, int],
    checkpoint: Path | None,
    device: str,
    deterministic: bool,
) -> tuple[list[EpisodeResult], float]:
    model: Any | None = None
    if checkpoint is not None:
        deps = import_training_deps()
        model = deps.MaskablePPO.load(str(resolve_model_path(checkpoint)), device=device)

    rng = np.random.default_rng(seed_start)
    env = CoCEnv(
        layout_profile=profiles[0],
        max_buildings=max_buildings,
        army_composition=army_composition,
    )
    results: list[EpisodeResult] = []
    start = time.perf_counter()
    for episode in range(episodes):
        profile = profiles[episode % len(profiles)]
        seed = seed_start + episode
        obs, _ = env.reset(seed=seed, options={"profile": profile})
        terminated = truncated = False
        steps = 0
        info: dict[str, Any] = {"damage_pct": 0.0, "score": 0.0, "stars": 0, "ticks_elapsed": 0}
        while not (terminated or truncated):
            action_masks = env.action_masks()
            if model is None:
                legal = np.flatnonzero(action_masks)
                action = int(rng.choice(legal))
            else:
                prediction, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=action_masks,
                )
                action = int(np.asarray(prediction).item())
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
        results.append(EpisodeResult(
            profile=profile,
            seed=seed,
            damage_pct=float(info["damage_pct"]),
            score=float(info["score"]),
            stars=int(info["stars"]),
            steps=steps,
            ticks=int(info["ticks_elapsed"]),
            terminated=bool(terminated),
            truncated=bool(truncated),
        ))
    env.close()
    return results, time.perf_counter() - start


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile ClashAI rollout collection hotspots.")
    parser.add_argument("--layout-profile", default="hard", help="Preset profile, comma-list, or 'all'.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--army-composition", default=",".join(f"{k}={v}" for k, v in DEFAULT_ARMY_COMPOSITION.items()))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional policy checkpoint; omitted means random legal actions.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/profile"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument("--limit", type=int, default=80)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    profiles = profile_sequence(args.layout_profile)
    army_composition = parse_army_composition(args.army_composition)
    max_buildings = max_preset_buildings()
    name = args.name or f"{'-'.join(profiles)}_n{args.episodes}_{'policy' if args.checkpoint else 'random'}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    results, elapsed = run_rollouts(
        profiles=profiles,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_buildings=max_buildings,
        army_composition=army_composition,
        checkpoint=args.checkpoint,
        device=args.device,
        deterministic=not args.stochastic,
    )
    profiler.disable()

    profile_path = args.output_dir / f"{name}.prof"
    text_path = args.output_dir / f"{name}.txt"
    json_path = args.output_dir / f"{name}.json"
    profiler.dump_stats(str(profile_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.limit)
    text_path.write_text(stream.getvalue(), encoding="utf-8")

    summary = result_metrics(results, elapsed)
    write_json(json_path, {
        "config": {
            "profiles": profiles,
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "army_composition": army_composition,
            "max_buildings": max_buildings,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "deterministic": not args.stochastic,
            "sort": args.sort,
            "limit": args.limit,
        },
        "summary": summary,
        "episodes": [asdict(result) for result in results],
    })
    print(summarize(results, elapsed), flush=True)
    print(f"wrote: {profile_path}", flush=True)
    print(f"wrote: {text_path}", flush=True)
    print(f"wrote: {json_path}", flush=True)


if __name__ == "__main__":
    main()
