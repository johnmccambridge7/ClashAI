from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from coc_env.env import CoCEnv
from coc_env.generation import PRESET_LAYOUT_PROFILES


DEFAULT_ARMY_COMPOSITION = {"barbarian": 40, "wall_breaker": 10}
DEFAULT_SPELL_COMPOSITION = {"rage": 2, "freeze": 2}


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
    values = [part.strip() for part in profile.split(",") if part.strip()]
    if not values:
        raise ValueError("profile must not be empty")
    if values == ["all"]:
        return sorted(PRESET_LAYOUT_PROFILES)
    if "all" in values:
        raise ValueError("'all' cannot be combined with explicit profile names")
    unknown = [name for name in values if name not in PRESET_LAYOUT_PROFILES]
    if unknown:
        known = ", ".join(["all", *sorted(PRESET_LAYOUT_PROFILES)])
        raise ValueError(f"unknown profile(s) {unknown!r}; expected one of: {known}")
    return values


def _parse_composition(value: str, *, allow_none: bool = False) -> dict[str, int]:
    normalized = value.strip().lower()
    if allow_none and normalized in {"", "none", "no", "false", "0"}:
        return {}
    out: dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("composition entries must look like kind=count")
        kind, count_text = [piece.strip() for piece in part.split("=", 1)]
        if not kind:
            raise ValueError("composition contains an empty kind")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"{kind} count must be non-negative")
        out[kind] = count
    if not out and not allow_none:
        raise ValueError("composition must contain at least one kind")
    return out


def _format_composition(composition: dict[str, int]) -> str:
    return ",".join(f"{kind}={count}" for kind, count in composition.items())


def _max_buildings() -> int:
    return max(p.max_buildings for p in PRESET_LAYOUT_PROFILES.values())


def _run_episode(
    env: CoCEnv,
    *,
    profile: str,
    seed: int,
    rng: np.random.Generator,
) -> EpisodeResult:
    env.reset(seed=seed, options={"profile": profile})
    terminated = truncated = False
    steps = 0
    info = {"damage_pct": 0.0, "score": 0.0, "stars": 0, "ticks_elapsed": 0}
    while not (terminated or truncated):
        valid = np.flatnonzero(env.action_masks())
        action = int(rng.choice(valid))
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
    return EpisodeResult(
        profile=profile,
        seed=seed,
        damage_pct=float(info["damage_pct"]),
        score=float(info["score"]),
        stars=int(info["stars"]),
        steps=steps,
        ticks=int(info["ticks_elapsed"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def run_time_budget(
    seconds: float = 60.0,
    seed: int = 42,
    profile: str = "hard",
    *,
    army_composition: dict[str, int] | None = None,
    spell_composition: dict[str, int] | None = None,
    max_ticks: int = 720,
) -> tuple[list[EpisodeResult], float]:
    profiles = _profile_sequence(profile)
    env = CoCEnv(
        layout_profile=profiles[0],
        max_buildings=_max_buildings(),
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=max_ticks,
    )
    rng = np.random.default_rng(seed)
    results: list[EpisodeResult] = []

    start = time.perf_counter()
    deadline = start + seconds
    episode = 0
    while time.perf_counter() < deadline:
        episode_seed = seed + episode
        episode_profile = profiles[episode % len(profiles)]
        results.append(_run_episode(env, profile=episode_profile, seed=episode_seed, rng=rng))
        episode += 1

    return results, time.perf_counter() - start


def _worker(task: tuple[int, int, int, list[str], int, dict[str, int], dict[str, int], int]) -> tuple[int, list[EpisodeResult]]:
    offset, count, seed, profiles, max_buildings, army_composition, spell_composition, max_ticks = task
    env = CoCEnv(
        layout_profile=profiles[offset % len(profiles)],
        max_buildings=max_buildings,
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=max_ticks,
    )
    rng = np.random.default_rng(seed + offset * 104_729 + 17)
    results = [
        _run_episode(
            env,
            profile=profiles[(offset + i) % len(profiles)],
            seed=seed + offset + i,
            rng=rng,
        )
        for i in range(count)
    ]
    return offset, results


def run_episodes(
    n_episodes: int,
    seed: int,
    profile: str,
    *,
    army_composition: dict[str, int],
    spell_composition: dict[str, int],
    max_ticks: int,
    workers: int = 1,
) -> tuple[list[EpisodeResult], float]:
    if n_episodes <= 0:
        raise ValueError("episodes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    profiles = _profile_sequence(profile)
    max_buildings = _max_buildings()
    start = time.perf_counter()

    if workers <= 1:
        _, results = _worker((
            0,
            n_episodes,
            seed,
            profiles,
            max_buildings,
            army_composition,
            spell_composition,
            max_ticks,
        ))
        return results, time.perf_counter() - start

    workers = min(workers, n_episodes)
    base = n_episodes // workers
    extra = n_episodes % workers
    tasks = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < extra else 0)
        if count <= 0:
            continue
        tasks.append((
            offset,
            count,
            seed,
            profiles,
            max_buildings,
            army_composition,
            spell_composition,
            max_ticks,
        ))
        offset += count

    chunks: list[tuple[int, list[EpisodeResult]]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            chunks.append(future.result())

    results: list[EpisodeResult] = []
    for _, chunk in sorted(chunks, key=lambda item: item[0]):
        results.extend(chunk)
    return results, time.perf_counter() - start


def run(n_episodes: int = 200, seed: int = 42, profile: str = "hard") -> np.ndarray:
    profiles = _profile_sequence(profile)
    results, _ = run_episodes(
        n_episodes,
        seed,
        ",".join(profiles),
        army_composition={"barbarian": 50},
        spell_composition={},
        max_ticks=720,
        workers=1,
    )
    return np.array([result.damage_pct for result in results])


def result_metrics(results: list[EpisodeResult], elapsed: float) -> dict[str, Any]:
    if not results:
        return {"episodes": 0, "elapsed_seconds": float(elapsed)}
    damage = np.array([r.damage_pct for r in results], dtype=np.float64)
    scores = np.array([r.score for r in results], dtype=np.float64)
    stars = np.array([r.stars for r in results], dtype=np.int64)
    steps = np.array([r.steps for r in results], dtype=np.float64)
    ticks = np.array([r.ticks for r in results], dtype=np.float64)
    return {
        "episodes": len(results),
        "elapsed_seconds": float(elapsed),
        "maps_per_min": float(len(results) / max(elapsed, 1e-9) * 60.0),
        "decisions_per_sec": float(steps.sum() / max(elapsed, 1e-9)),
        "sim_ticks_per_sec": float(ticks.sum() / max(elapsed, 1e-9)),
        "damage_mean": float(damage.mean()),
        "damage_median": float(np.median(damage)),
        "damage_min": float(damage.min()),
        "damage_max": float(damage.max()),
        "damage_std": float(damage.std()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "stars_mean": float(stars.mean()),
        "stars_0": int((stars == 0).sum()),
        "stars_1": int((stars == 1).sum()),
        "stars_2": int((stars == 2).sum()),
        "stars_3": int((stars == 3).sum()),
        "p_stars_ge_2": float((stars >= 2).mean()),
        "p_damage_ge_50": float((damage >= 0.50).mean()),
        "p_damage_ge_75": float((damage >= 0.75).mean()),
        "p_damage_ge_90": float((damage >= 0.90).mean()),
        "steps_mean": float(steps.mean()),
        "ticks_mean": float(ticks.mean()),
        "terminated": int(sum(r.terminated for r in results)),
        "truncated": int(sum(r.truncated for r in results)),
    }


def write_outputs(
    output_dir: Path,
    name: str,
    results: list[EpisodeResult],
    elapsed: float,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(result) for result in results]
    profile_summaries = {}
    for profile in sorted({result.profile for result in results}):
        profile_results = [result for result in results if result.profile == profile]
        profile_elapsed = elapsed * len(profile_results) / max(1, len(results))
        profile_summaries[profile] = result_metrics(profile_results, profile_elapsed)

    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "summary": result_metrics(results, elapsed),
            "profile_summaries": profile_summaries,
            "episodes": records,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


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
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="hard", help="Preset profile, comma-list, or 'all'.")
    parser.add_argument("--army-composition", default=_format_composition(DEFAULT_ARMY_COMPOSITION))
    parser.add_argument("--spell-composition", default=_format_composition(DEFAULT_SPELL_COMPOSITION))
    parser.add_argument("--max-ticks", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baselines"))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    army_composition = _parse_composition(args.army_composition)
    spell_composition = _parse_composition(args.spell_composition, allow_none=True)
    config = {
        "policy": "uniform_random_legal_action",
        "profile": args.profile,
        "seed": args.seed,
        "episodes": args.episodes,
        "seconds": args.seconds,
        "workers": args.workers,
        "army_composition": army_composition,
        "spell_composition": spell_composition,
        "max_ticks": args.max_ticks,
        "max_buildings": _max_buildings(),
    }
    if args.episodes is None:
        results, elapsed = run_time_budget(
            seconds=args.seconds,
            seed=args.seed,
            profile=args.profile,
            army_composition=army_composition,
            spell_composition=spell_composition,
            max_ticks=args.max_ticks,
        )
    else:
        results, elapsed = run_episodes(
            args.episodes,
            args.seed,
            args.profile,
            army_composition=army_composition,
            spell_composition=spell_composition,
            max_ticks=args.max_ticks,
            workers=args.workers,
        )

    name = args.name or f"random_{args.profile.replace(',', '-')}_seed{args.seed}_n{len(results)}"
    json_path, csv_path = write_outputs(args.output_dir, name, results, elapsed, config)
    print(json.dumps(config, indent=2, sort_keys=True))
    print(summarize(results, elapsed))
    for profile in sorted({result.profile for result in results}):
        profile_results = [result for result in results if result.profile == profile]
        profile_elapsed = elapsed * len(profile_results) / max(1, len(results))
        print(f"\n[{profile}]")
        print(summarize(profile_results, profile_elapsed))
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
